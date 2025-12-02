import asyncio
import os
import tempfile
import time
from datetime import datetime
from typing import Optional
import threading
import queue
import gc
import math
import re
from collections import deque
import json

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
from pydantic import BaseModel

# AI Models
import nemo.collections.asr as nemo_asr
from transformers import AutoTokenizer, AutoModelForCausalLM

# WebRTC VAD (optional)
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtcvad = None

class VoiceRecorder:
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2"):
        print("🔄 Loading ASR model...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        print("✅ ASR model loaded!")
        
        # Audio settings
        self.samplerate = 16000
        self.channels = 1
        self.chunk_duration = 0.1
        self.chunk_size = int(self.chunk_duration * self.samplerate)
        
        # VAD settings
        self.silence_threshold = 0.008
        self.speech_threshold = 0.015
        self.min_speech_duration = 0.5
        self.max_silence_duration = 2.5
        
        # State
        self.is_recording = False
        self.transcript_text = ""
        self.full_transcript = ""
        self.audio_buffer = deque(maxlen=int(60.0 / self.chunk_duration))
        self.buffer_timestamps = deque(maxlen=int(60.0 / self.chunk_duration))
        self.speech_history = deque(maxlen=10)
        self.speech_confidence_threshold = 0.3
        
        # Recording state
        self.speech_state = "idle"
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.current_recording_start = 0
        
        # Recording session
        self.session_transcript = ""
        self.session_audio = []
        
        # Initialize WebRTC VAD
        if WEBRTC_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)
                self.use_webrtc_vad = True
                print("✅ WebRTC VAD initialized")
            except:
                self.use_webrtc_vad = False
        else:
            self.use_webrtc_vad = False
            print("⚠️ WebRTC VAD not available, using RMS-based detection")
    
    def calculate_rms(self, audio_chunk):
        if len(audio_chunk) == 0:
            return 0.0
        audio_chunk = audio_chunk - np.mean(audio_chunk)
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def is_speech_detected(self, audio_chunk):
        rms = self.calculate_rms(audio_chunk)
        rms_speech = rms > self.speech_threshold
        
        webrtc_speech = False
        if self.use_webrtc_vad:
            try:
                audio_16bit = (audio_chunk * 32767).astype(np.int16)
                frame_size = len(audio_16bit)
                if frame_size not in [160, 320, 480, 960]:
                    target_size = 320
                    if frame_size < target_size:
                        audio_16bit = np.pad(audio_16bit, (0, target_size - frame_size), mode='constant')
                    else:
                        audio_16bit = audio_16bit[:target_size]
                webrtc_speech = self.vad.is_speech(audio_16bit.tobytes(), self.samplerate)
            except:
                pass
        
        is_speech = webrtc_speech or rms_speech
        self.speech_history.append(is_speech)
        
        recent_speech_ratio = sum(self.speech_history) / len(self.speech_history)
        return recent_speech_ratio > self.speech_confidence_threshold, rms
    
    def get_recording_audio(self, start_time, end_time):
        recording_chunks = []
        for i, timestamp in enumerate(self.buffer_timestamps):
            if start_time <= timestamp <= end_time:
                if i < len(self.audio_buffer):
                    recording_chunks.append(self.audio_buffer[i])
        
        return np.concatenate(recording_chunks) if recording_chunks else np.array([])
    
    def process_audio_chunk(self, audio_chunk):
        current_time = time.time()
        
        # Add to session audio
        if self.is_recording:
            self.session_audio.extend(audio_chunk.flatten())
        
        # Add to buffer
        self.audio_buffer.append(audio_chunk.copy())
        self.buffer_timestamps.append(current_time)
        
        # Detect speech
        is_speech, rms = self.is_speech_detected(audio_chunk)
        
        # State machine
        if self.speech_state == "idle":
            if is_speech:
                self.speech_state = "speech_detected"
                self.speech_start_time = current_time
                self.last_speech_time = current_time
                self.current_recording_start = max(0, current_time - 0.5)
                
        elif self.speech_state == "speech_detected":
            if is_speech:
                self.last_speech_time = current_time
                speech_duration = current_time - self.speech_start_time
                if speech_duration >= self.min_speech_duration:
                    self.speech_state = "recording"
            else:
                silence_duration = current_time - self.last_speech_time
                if silence_duration > self.max_silence_duration:
                    self.speech_state = "idle"
                    
        elif self.speech_state == "recording":
            if is_speech:
                self.last_speech_time = current_time
            else:
                silence_duration = current_time - self.last_speech_time
                if silence_duration > self.max_silence_duration:
                    self.speech_state = "post_speech"
                    
        elif self.speech_state == "post_speech":
            self.process_recording()
            self.speech_state = "idle"
    
    def process_recording(self):
        recording_start = self.current_recording_start
        recording_end = self.last_speech_time + 0.3
        
        audio_data = self.get_recording_audio(recording_start, recording_end)
        
        if len(audio_data) == 0:
            return
        
        duration = len(audio_data) / self.samplerate
        if duration < self.min_speech_duration:
            return
        
        # Transcribe
        transcription = self.transcribe_audio(audio_data)
        if transcription:
            self.session_transcript += transcription + " "
            print(f"📝 Transcribed: {transcription}")
    
    def transcribe_audio(self, audio_data):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                sf.write(temp_file.name, audio_data, self.samplerate)
                temp_path = temp_file.name
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Transcribe
            output = self.asr_model.transcribe([temp_path])
            transcription_text = self.extract_clean_text(output)
            
            # Cleanup
            os.remove(temp_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return transcription_text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def extract_clean_text(self, transcription_output):
        if not transcription_output or len(transcription_output) == 0:
            return ""
        
        result = transcription_output[0]
        
        # Method 1: Direct text
        if hasattr(result, 'text') and result.text and result.text.strip():
            return result.text.strip()
        
        # Method 2: String representation
        result_str = str(result)
        if 'text=' in result_str:
            try:
                text_match = re.search(r"text='([^']*)'", result_str)
                if text_match and text_match.group(1).strip():
                    return text_match.group(1).strip()
            except:
                pass
        
        # Method 3: Segments
        if hasattr(result, 'timestamp') and result.timestamp and result.timestamp.get('segment'):
            segments = result.timestamp['segment']
            segment_texts = []
            
            for segment in segments:
                if isinstance(segment, dict) and 'segment' in segment:
                    segment_text = segment['segment'].strip()
                    if segment_text:
                        segment_texts.append(segment_text)
            
            if segment_texts:
                return ' '.join(segment_texts)
        
        return ""
    
    def start_recording(self):
        """Start recording session"""
        self.is_recording = True
        self.session_transcript = ""
        self.session_audio = []
        self.speech_state = "idle"
        print("🎤 Recording started")
    
    def stop_recording(self):
        """Stop recording session"""
        self.is_recording = False
        print("⏹️ Recording stopped")
        
        # Save session audio
        if self.session_audio:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{session_id}.wav"
            sf.write(filename, np.array(self.session_audio), self.samplerate)
            print(f"💾 Audio saved: {filename}")
        
        return self.session_transcript
    
    def get_current_transcript(self):
        """Get current session transcript"""
        return self.session_transcript


class DetailedNotesExtractor:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print("🔄 Loading Gemma model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()
        
        # Settings
        self.max_context = 7000
        self.chunk_size = 60
        self.overlap_lines = 20
        
        print(f"✅ Gemma model loaded!")
    
    def get_detailed_notes_prompt(self):
        return """You are a medical scribe creating COMPREHENSIVE detailed notes. Extract EVERY SINGLE detail from this medical conversation. Do NOT summarize or skip anything.

FORMAT (include everything under each section):

PATIENT PRESENTATION & CONCERNS:
• [Every symptom, complaint, and desire mentioned]
• [Exact patient descriptions and words used]

DOCTOR'S COMPLETE EXPLANATIONS:
• [All medical concepts explained in detail]
• [Personal anecdotes and experiences shared]
• [Analogies and examples used]

SPECIFIC PRODUCT RECOMMENDATIONS:
• [Every brand name, product name, and formulation]
• [Exact percentages and concentrations mentioned]
• [Specific application instructions for each product]

DETAILED TREATMENT PROTOCOLS:
• [Complete morning routine with all steps]
• [Complete evening routine with all steps]
• [Exact application amounts and techniques]

SAFETY WARNINGS & CONTRAINDICATIONS:
• [All pregnancy warnings]
• [Irritation precautions]
• [What NOT to do]

PATIENT EDUCATION PROVIDED:
• [All skin science explained]
• [How ingredients work]
• [Environmental factors discussed]

Extract EVERYTHING. Include ALL brand names, percentages, and specific instructions."""
    
    def get_rolling_notes_prompt(self):
        return """Merge existing comprehensive notes with new content. PRESERVE ALL details, specifications, and information.

MERGING RULES:
• PRESERVE every brand name, percentage, and product detail
• KEEP all application instructions and techniques
• MAINTAIN all safety warnings and contraindications
• ADD new information to existing categories
• NEVER remove or compress existing details

The final notes must be MORE detailed than either input."""
    
    def get_final_structure_prompt(self):
        return """Organize these comprehensive detailed notes into medical note format while preserving EVERY SINGLE DETAIL.

Use this structure but EXPAND each section with ALL available details:

PATIENT VISIT NOTES

CHIEF COMPLAINT & SYMPTOMS:
• [All patient concerns with exact descriptions]

HISTORY & BACKGROUND:
• [All relevant background information]
• [Doctor's personal experiences shared]

PHYSICAL EXAMINATION:
• [All examination details OR state "No physical examination performed"]

ASSESSMENT & EXPLANATIONS:
• [COMPLETE medical explanations provided]
• [All definitions and medical concepts taught]
• [Include specific quotes for important corrections]

TREATMENT PLAN:
• [DETAILED morning routine with ALL products and instructions]
• [DETAILED evening routine with ALL products and instructions]
• [ALL product names, brands, percentages, concentrations]

PATIENT EDUCATION:
• [ALL educational content provided]
• [Every concept explained]

LIFESTYLE MODIFICATIONS:
• [ALL lifestyle recommendations]

FOLLOW-UP CARE:
• [Any follow-up mentioned OR state "None specified"]

IMPORTANT NOTES:
• [ALL safety warnings and contraindications]
• [Important application tips and techniques]

Include EVERYTHING."""
    
    def clean_transcript(self, text):
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if (line and 
                not line.startswith('[') and 
                not line.startswith('=') and
                'Transcription completed' not in line and
                'END OF TRANSCRIPTION' not in line and
                'Conversation Transcript' not in line):
                lines.append(line)
        return lines
    
    def count_tokens(self, text):
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split()) * 1.3
    
    def extract_detailed_notes(self, prompt, content, max_tokens=1200):
        full_prompt = f"{prompt}\n\nMEDICAL CONVERSATION:\n{content}"
        
        # Check token count
        if self.count_tokens(full_prompt) > self.max_context:
            words = content.split()
            target_length = self.max_context - self.count_tokens(prompt) - 200
            while self.count_tokens(' '.join(words)) > target_length and len(words) > 100:
                words = words[:-3]
            content = ' '.join(words)
            full_prompt = f"{prompt}\n\nMEDICAL CONVERSATION:\n{content}"
        
        messages = [{"role": "user", "content": full_prompt}]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self.max_context
            ).to(self.model.device)
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.05,
                    length_penalty=0.8
                )
            
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = generation[0][input_len:]
            notes = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return notes.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"[Error extracting notes: {str(e)[:100]}]"
    
    def rolling_approach(self, lines):
        print("🔄 Using rolling detailed notes approach...")
        
        current_notes = ""
        total_chunks = math.ceil(len(lines) / self.chunk_size)
        
        for i in range(0, len(lines), self.chunk_size):
            chunk_num = (i // self.chunk_size) + 1
            
            start_idx = max(0, i - self.overlap_lines)
            end_idx = i + self.chunk_size
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = '\n'.join(chunk_lines)
            
            print(f"🔄 Processing chunk {chunk_num}/{total_chunks}...")
            
            if not current_notes:
                prompt = self.get_detailed_notes_prompt()
                new_notes = self.extract_detailed_notes(prompt, chunk_content, max_tokens=1500)
            else:
                prompt = self.get_rolling_notes_prompt()
                content = f"EXISTING DETAILED NOTES:\n{current_notes}\n\nNEW CONTENT:\n{chunk_content}"
                new_notes = self.extract_detailed_notes(prompt, content, max_tokens=1800)
            
            current_notes = new_notes
        
        # Final structure
        final_prompt = self.get_final_structure_prompt()
        final_content = f"COMPREHENSIVE DETAILED NOTES:\n{current_notes}"
        final_notes = self.extract_detailed_notes(final_prompt, final_content, max_tokens=2000)
        
        return final_notes
    
    def hierarchical_approach(self, lines):
        print("🔄 Using hierarchical detailed notes approach...")
        
        def recursive_combine_notes(content_list, level=1):
            if len(content_list) == 1:
                return content_list[0]
            
            print(f"📊 Level {level}: Processing {len(content_list)} sections...")
            
            combined_notes = []
            items_per_batch = 3
            
            for i in range(0, len(content_list), items_per_batch):
                batch = content_list[i:i + items_per_batch]
                
                if level == 1:
                    prompt = self.get_detailed_notes_prompt()
                    content = f"MEDICAL CONVERSATION:\n\n" + "\n\n".join(batch)
                else:
                    prompt = "Combine these detailed notes into ONE comprehensive set that preserves every detail. Merge similar sections without losing any information."
                    content = f"DETAILED NOTES SECTIONS:\n\n" + "\n\n--- SECTION BREAK ---\n\n".join(batch)
                
                notes = self.extract_detailed_notes(prompt, content, max_tokens=1500)
                combined_notes.append(notes)
            
            if len(combined_notes) > 1:
                return recursive_combine_notes(combined_notes, level + 1)
            else:
                return combined_notes[0]
        
        # Create chunks
        chunks = []
        for i in range(0, len(lines), self.chunk_size):
            start_idx = max(0, i - self.overlap_lines)
            end_idx = i + self.chunk_size
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = '\n'.join(chunk_lines)
            chunks.append(chunk_content)
        
        # Recursive combine
        comprehensive_notes = recursive_combine_notes(chunks)
        
        # Final structure
        final_prompt = self.get_final_structure_prompt()
        final_content = f"COMPREHENSIVE DETAILED NOTES:\n{comprehensive_notes}"
        final_notes = self.extract_detailed_notes(final_prompt, final_content, max_tokens=2000)
        
        return final_notes
    
    def process_transcript(self, transcript_text):
        lines = self.clean_transcript(transcript_text)
        
        if not lines:
            return "❌ No valid content found!"
        
        print(f"📄 Processing {len(lines)} lines of content")
        
        # Use both methods
        rolling_notes = self.rolling_approach(lines)
        hierarchical_notes = self.hierarchical_approach(lines)
        
        # Combine both results
        final_prompt = """Combine these two detailed medical note sets into ONE final comprehensive set. Preserve ALL details from both sources.

Use this structure:
PATIENT VISIT NOTES

CHIEF COMPLAINT & SYMPTOMS:
HISTORY & BACKGROUND:
PHYSICAL EXAMINATION:
ASSESSMENT & EXPLANATIONS:
TREATMENT PLAN:
PATIENT EDUCATION:
LIFESTYLE MODIFICATIONS:
FOLLOW-UP CARE:
IMPORTANT NOTES:

Include EVERYTHING from both note sets."""
        
        content = f"ROLLING METHOD NOTES:\n{rolling_notes}\n\nHIERARCHICAL METHOD NOTES:\n{hierarchical_notes}"
        
        final_notes = self.extract_detailed_notes(final_prompt, content, max_tokens=2500)
        
        return final_notes


# Global instances
recorder = None
extractor = None
audio_thread = None
audio_queue = queue.Queue()

# Request/Response models
class TranscriptRequest(BaseModel):
    transcript: str

class NotesResponse(BaseModel):
    notes: str
    timestamp: str

class RecordingStatus(BaseModel):
    is_recording: bool
    transcript: str
    message: str

# FastAPI app
app = FastAPI(title="Medical Voice Recording & Analysis")

# Initialize models
@app.on_event("startup")
async def startup_event():
    global recorder, extractor
    print("🚀 Initializing models...")
    recorder = VoiceRecorder()
    extractor = DetailedNotesExtractor()
    print("✅ All models loaded successfully!")

def audio_callback(indata, frames, time, status):
    """Audio callback for continuous recording"""
    if status:
        print(f"⚠️ Audio status: {status}")
    
    try:
        audio_queue.put(indata.copy(), block=False)
    except queue.Full:
        pass  # Drop frame if queue is full

def audio_processing_thread():
    """Background thread to process audio"""
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
            if recorder and recorder.is_recording:
                recorder.process_audio_chunk(audio_chunk.flatten())
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Audio processing error: {e}")

# Start audio processing thread
audio_thread = threading.Thread(target=audio_processing_thread, daemon=True)
audio_thread.start()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Voice Recording & Analysis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                font-size: 2.5rem;
                color: white;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                color: rgba(255,255,255,0.9);
                font-size: 1.1rem;
            }
            
            .card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 30px;
            }
            
            .recording-section {
                text-align: center;
            }
            
            .record-button {
                width: 120px;
                height: 120px;
                border-radius: 50%;
                border: none;
                font-size: 2rem;
                color: white;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 20px auto;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .record-button.inactive {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }
            
            .record-button.active {
                background: linear-gradient(135deg, #27ae60, #229954);
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            .status {
                font-size: 1.2rem;
                font-weight: 600;
                margin: 20px 0;
            }
            
            .transcript {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                min-height: 150px;
                max-height: 300px;
                overflow-y: auto;
                font-family: monospace;
                border: 2px dashed #dee2e6;
                white-space: pre-wrap;
            }
            
            .btn {
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 10px;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none !important;
            }
            
            .notes-output {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                white-space: pre-wrap;
                font-family: monospace;
                font-size: 0.9rem;
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
            }
            
            .loading {
                text-align: center;
                color: #666;
                font-style: italic;
            }
            
            .error {
                color: #e74c3c;
                background: #fadbd8;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            
            .success {
                color: #27ae60;
                background: #d5f4e6;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ Medical Voice Recording & Analysis</h1>
                <p>Record medical conversations and generate comprehensive detailed notes</p>
            </div>
            
            <div class="card">
                <div class="recording-section">
                    <h2>Voice Recording</h2>
                    <button class="record-button inactive" id="recordBtn">
                        <span id="recordIcon">🎤</span>
                    </button>
                    <div class="status" id="status">Click to start recording</div>
                    
                    <div class="transcript" id="transcript">
                        <div class="loading">Transcript will appear here...</div>
                    </div>
                    
                    <button class="btn btn-primary" id="generateBtn" disabled onclick="generateNotes()">
                        📝 Generate Detailed Medical Notes
                    </button>
                </div>
            </div>
            
            <div class="card" id="notesCard" style="display: none;">
                <h2>📋 Detailed Medical Notes</h2>
                <div class="notes-output" id="notesOutput"></div>
                <button class="btn btn-primary" onclick="downloadNotes()">💾 Download Notes</button>
            </div>
        </div>

        <script>
            let isRecording = false;
            let currentTranscript = '';
            let transcriptUpdateInterval = null;

            const recordBtn = document.getElementById('recordBtn');
            const recordIcon = document.getElementById('recordIcon');
            const status = document.getElementById('status');
            const transcript = document.getElementById('transcript');
            const generateBtn = document.getElementById('generateBtn');
            const notesCard = document.getElementById('notesCard');
            const notesOutput = document.getElementById('notesOutput');

            recordBtn.addEventListener('click', toggleRecording);

            async function toggleRecording() {
                if (!isRecording) {
                    await startRecording();
                } else {
                    await stopRecording();
                }
            }

            async function startRecording() {
                try {
                    const response = await fetch('/start-recording', {
                        method: 'POST'
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to start recording');
                    }
                    
                    const data = await response.json();
                    
                    isRecording = true;
                    recordBtn.className = 'record-button active';
                    recordIcon.textContent = '⏹️';
                    status.textContent = 'Recording... (Click to stop)';
                    transcript.innerHTML = '<div class="loading">Listening for speech...</div>';
                    
                    // Start polling for transcript updates
                    transcriptUpdateInterval = setInterval(updateTranscript, 1000);
                    
                    showSuccess('Recording started successfully!');
                    
                } catch (error) {
                    showError('Failed to start recording: ' + error.message);
                }
            }

            async function stopRecording() {
                try {
                    const response = await fetch('/stop-recording', {
                        method: 'POST'
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to stop recording');
                    }
                    
                    const data = await response.json();
                    
                    isRecording = false;
                    recordBtn.className = 'record-button inactive';
                    recordIcon.textContent = '🎤';
                    status.textContent = 'Recording stopped';
                    
                    // Stop polling
                    if (transcriptUpdateInterval) {
                        clearInterval(transcriptUpdateInterval);
                        transcriptUpdateInterval = null;
                    }
                    
                    // Update final transcript
                    currentTranscript = data.transcript;
                    transcript.textContent = currentTranscript || 'No speech detected in recording.';
                    
                    if (currentTranscript.trim()) {
                        generateBtn.disabled = false;
                        showSuccess('Recording completed! You can now generate detailed notes.');
                    } else {
                        showError('No speech was detected in the recording. Please try again.');
                    }
                    
                } catch (error) {
                    showError('Failed to stop recording: ' + error.message);
                }
            }

            async function updateTranscript() {
                if (!isRecording) return;
                
                try {
                    const response = await fetch('/get-transcript');
                    
                    if (response.ok) {
                        const data = await response.json();
                        if (data.transcript && data.transcript !== currentTranscript) {
                            currentTranscript = data.transcript;
                            transcript.textContent = currentTranscript;
                            
                            if (currentTranscript.trim()) {
                                generateBtn.disabled = false;
                            }
                        }
                    }
                } catch (error) {
                    console.error('Failed to update transcript:', error);
                }
            }

            async function generateNotes() {
                if (!currentTranscript.trim()) {
                    showError('No transcript available to analyze');
                    return;
                }

                generateBtn.disabled = true;
                generateBtn.textContent = '⏳ Generating Notes...';
                notesOutput.innerHTML = '<div class="loading">Processing transcript and generating detailed medical notes...<br><br>This may take a few minutes depending on the transcript length.</div>';
                notesCard.style.display = 'block';

                try {
                    const response = await fetch('/generate-notes', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            transcript: currentTranscript
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to generate notes');
                    }

                    const data = await response.json();
                    notesOutput.textContent = data.notes;
                    showSuccess('Detailed medical notes generated successfully!');

                } catch (error) {
                    showError('Failed to generate notes: ' + error.message);
                } finally {
                    generateBtn.disabled = false;
                    generateBtn.textContent = '📝 Generate Detailed Medical Notes';
                }
            }

            function downloadNotes() {
                const notes = notesOutput.textContent;
                if (!notes.trim()) return;

                const blob = new Blob([notes], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `medical_notes_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                showSuccess('Medical notes downloaded successfully!');
            }

            function showError(message) {
                removeMessages();
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error';
                errorDiv.textContent = '❌ ' + message;
                document.querySelector('.container').appendChild(errorDiv);
                setTimeout(() => errorDiv.remove(), 5000);
            }

            function showSuccess(message) {
                removeMessages();
                const successDiv = document.createElement('div');
                successDiv.className = 'success';
                successDiv.textContent = '✅ ' + message;
                document.querySelector('.container').appendChild(successDiv);
                setTimeout(() => successDiv.remove(), 5000);
            }

            function removeMessages() {
                document.querySelectorAll('.error, .success').forEach(el => el.remove());
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/start-recording")
async def start_recording(background_tasks: BackgroundTasks):
    try:
        if not recorder:
            raise HTTPException(status_code=500, detail="Recorder not initialized")
        
        # Start recording
        recorder.start_recording()
        
        # Start audio stream in background
        background_tasks.add_task(start_audio_stream)
        
        return RecordingStatus(
            is_recording=True,
            transcript="",
            message="Recording started successfully"
        )
        
    except Exception as e:
        print(f"❌ Error starting recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")

@app.post("/stop-recording")
async def stop_recording():
    try:
        if not recorder:
            raise HTTPException(status_code=500, detail="Recorder not initialized")
        
        print("🛑 Stop recording request received")
        
        # Stop recording and get transcript
        transcript = recorder.stop_recording()
        
        # Stop audio stream
        stop_audio_stream()
        
        # Debug output
        print(f"🔍 Debug - Returned transcript: '{transcript}'")
        print(f"🔍 Debug - Transcript length: {len(transcript)} chars")
        
        # Additional fallback check
        if not transcript.strip():
            print("⚠️ Warning - Empty transcript, trying alternative approach...")
            
            # Check if any transcriptions happened
            if recorder.total_recordings > 0:
                # Try to read from the transcript file directly
                try:
                    with open(recorder.transcript_file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        print(f"🔍 Debug - File content: '{file_content}'")
                        
                        # Extract meaningful content
                        lines = file_content.split('\n')
                        content_lines = []
                        for line in lines:
                            line = line.strip()
                            if (line and 
                                not line.startswith('=') and
                                not line.startswith('Conversation Transcript') and
                                not line.startswith('END OF TRANSCRIPTION')):
                                content_lines.append(line)
                        
                        if content_lines:
                            transcript = ' '.join(content_lines)
                            print(f"🔄 Recovered transcript from file: '{transcript}'")
                except Exception as e:
                    print(f"❌ Error reading transcript file: {e}")
        
        return RecordingStatus(
            is_recording=False,
            transcript=transcript,
            message="Recording stopped successfully" if transcript.strip() else "Recording stopped - no speech detected"
        )
        
    except Exception as e:
        print(f"❌ Error stopping recording: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")

@app.get("/get-transcript")
async def get_current_transcript():
    try:
        if not recorder:
            raise HTTPException(status_code=500, detail="Recorder not initialized")
        
        transcript = recorder.get_current_transcript()
        
        # Debug output
        print(f"🔍 Debug - Current transcript: '{transcript}' (length: {len(transcript)})")
        
        return {
            "transcript": transcript,
            "is_recording": recorder.is_recording,
            "total_recordings": recorder.total_recordings,
            "speech_state": recorder.speech_state,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting transcript: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get transcript: {str(e)}") 

@app.post("/generate-notes", response_model=NotesResponse)
async def generate_notes(request: TranscriptRequest):
    try:
        if not request.transcript.strip():
            raise HTTPException(status_code=400, detail="Empty transcript provided")
        
        print(f"📝 Generating notes for transcript: {len(request.transcript)} characters")
        
        # Process transcript with both methods
        notes = extractor.process_transcript(request.transcript)
        
        return NotesResponse(
            notes=notes,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Error generating notes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate notes: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "recorder_loaded": recorder is not None,
        "extractor_loaded": extractor is not None,
        "recording": recorder.is_recording if recorder else False,
        "timestamp": datetime.now().isoformat()
    }

# Audio stream management
audio_stream = None

def start_audio_stream():
    """Start the audio input stream"""
    global audio_stream
    
    try:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        
        audio_stream = sd.InputStream(
            samplerate=recorder.samplerate,
            channels=recorder.channels,
            callback=audio_callback,
            blocksize=recorder.chunk_size,
            dtype=np.float32
        )
        
        audio_stream.start()
        print("🎤 Audio stream started")
        
    except Exception as e:
        print(f"❌ Error starting audio stream: {e}")

def stop_audio_stream():
    """Stop the audio input stream"""
    global audio_stream
    
    try:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
        print("⏹️ Audio stream stopped")
        
    except Exception as e:
        print(f"❌ Error stopping audio stream: {e}")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    stop_audio_stream()
    print("🛑 Server shutdown complete")

if __name__ == "__main__":
    print("🚀 Starting Medical Voice Recording & Analysis Server...")
    print("📝 Features:")
    print("  - Real-time voice recording with speech detection")
    print("  - Automatic transcription using NeMo ASR")
    print("  - Detailed medical notes generation using Gemma")
    print("  - Both rolling and hierarchical analysis methods")
    print("  - Simple HTTP API (no WebSocket required)")
    print("\n🌐 Server will be available at: http://localhost:8000")
    print("⚡ Press Ctrl+C to stop the server")
    print("\n📋 Installation requirements:")
    print("  pip install fastapi uvicorn torch transformers sounddevice soundfile")
    print("  pip install numpy nemo_toolkit[asr] webrtcvad")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )