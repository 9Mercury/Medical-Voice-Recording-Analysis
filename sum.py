import nemo.collections.asr as nemo_asr
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
from datetime import datetime
import queue
import threading
import asyncio
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
import psutil
from collections import deque
import re

# WebRTC VAD is optional - will fallback to RMS if not available
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtcvad = None

class ContinuousVoiceActivityRecorder:
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2"):
        # Load the ASR model
        print("🔄 Loading ASR model...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        print("✅ ASR model loaded!")
        
        # Optimize model for better performance
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            self.device = "cpu"
            print("💻 Using CPU")
        
        # Audio config - smaller chunks for better responsiveness
        self.samplerate = 16000
        self.channels = 1
        self.chunk_duration = 0.1  # Much smaller chunks (100ms)
        self.chunk_size = int(self.chunk_duration * self.samplerate)
        
        # Initialize WebRTC VAD for better voice detection
        if WEBRTC_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3 (2 is balanced)
                self.use_webrtc_vad = True
                print("✅ WebRTC VAD initialized")
            except Exception as e:
                print(f"⚠️ WebRTC VAD initialization failed: {e}")
                self.use_webrtc_vad = False
        else:
            print("⚠️ WebRTC VAD not available, using RMS-based detection")
            self.use_webrtc_vad = False
        
        # Continuous recording config
        self.silence_threshold = 0.008  # Will be auto-calibrated
        self.speech_threshold = 0.015   # Higher threshold for speech detection
        self.min_speech_duration = 0.5  # Minimum speech duration to process
        self.max_silence_duration = 2.5  # Maximum silence before processing
        self.max_recording_duration = 45.0  # Maximum single recording
        self.pre_speech_buffer_duration = 0.5  # Buffer before speech starts
        self.post_speech_buffer_duration = 0.3  # Buffer after speech ends
        
        # Continuous audio buffer - rolling window
        self.buffer_duration = 60.0  # Keep 60 seconds of audio
        self.audio_buffer = deque(maxlen=int(self.buffer_duration / self.chunk_duration))
        self.buffer_timestamps = deque(maxlen=int(self.buffer_duration / self.chunk_duration))
        
        # State tracking
        self.speech_state = "idle"  # idle, speech_detected, recording, post_speech
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.current_recording_start = 0
        self.silence_duration = 0
        
        # Smoothing for speech detection
        self.speech_history = deque(maxlen=10)  # Last 10 chunks (1 second)
        self.speech_confidence_threshold = 0.3  # 30% of recent chunks must be speech
        
        # Memory management
        self.cleanup_interval = 20.0
        self.last_cleanup = time.time()
        self.memory_report_interval = 30.0
        self.last_memory_report = time.time()
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="transcribe")
        self.transcription_futures = []
        
        # Output files
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript_file = f"conversation_transcript_{self.session_id}.txt"
        self.audio_file = f"conversation_audio_{self.session_id}.wav"
        
        # Initialize transcript file
        self.init_transcript_file()
        
        # Session stats
        self.total_recordings = 0
        self.total_duration = 0.0
        self.false_starts = 0
        self.skipped_recordings = 0
        
        # Audio queue
        self.audio_queue = queue.Queue(maxsize=50)
        
        # Calibration flag
        self.is_calibrated = False
    
    def init_transcript_file(self):
        """Initialize transcript file with minimal header"""
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            # Just write a simple header
            f.write(f"Conversation Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def calculate_rms(self, audio_chunk):
        """Calculate RMS with noise reduction"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Remove DC offset
        audio_chunk = audio_chunk - np.mean(audio_chunk)
        
        # Apply simple bandpass filter for speech frequencies (300-3400 Hz)
        # This is a simple approximation - for better results, use scipy.signal
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        return rms
    
    def webrtc_vad_check(self, audio_chunk):
        """Use WebRTC VAD if available"""
        if not self.use_webrtc_vad:
            return False
        
        try:
            # WebRTC VAD expects 16-bit PCM
            audio_16bit = (audio_chunk * 32767).astype(np.int16)
            
            # WebRTC VAD works with specific frame sizes
            frame_size = len(audio_16bit)
            if frame_size in [160, 320, 480, 960]:  # Valid frame sizes for 16kHz
                return self.vad.is_speech(audio_16bit.tobytes(), self.samplerate)
            else:
                # Pad or truncate to nearest valid size
                target_size = 320  # 20ms at 16kHz
                if frame_size < target_size:
                    padded = np.pad(audio_16bit, (0, target_size - frame_size), mode='constant')
                else:
                    padded = audio_16bit[:target_size]
                return self.vad.is_speech(padded.tobytes(), self.samplerate)
        except Exception as e:
            # Fallback to RMS if WebRTC VAD fails
            return False
    
    def is_speech_detected(self, audio_chunk):
        """Enhanced speech detection combining multiple methods"""
        rms = self.calculate_rms(audio_chunk)
        
        # Primary detection methods
        rms_speech = rms > self.speech_threshold
        webrtc_speech = self.webrtc_vad_check(audio_chunk) if self.use_webrtc_vad else False
        
        # Combine methods
        if self.use_webrtc_vad:
            # Use WebRTC as primary, RMS as secondary
            is_speech = webrtc_speech or (rms_speech and rms > self.speech_threshold * 1.5)
        else:
            # Use enhanced RMS detection
            is_speech = rms_speech
            
            # Additional spectral analysis for better accuracy
            if is_speech:
                try:
                    fft = np.fft.fft(audio_chunk)
                    freqs = np.fft.fftfreq(len(audio_chunk), 1/self.samplerate)
                    
                    # Speech frequency range (300-3400 Hz)
                    speech_mask = (freqs >= 300) & (freqs <= 3400)
                    if np.any(speech_mask):
                        speech_energy = np.sum(np.abs(fft[speech_mask]))
                        total_energy = np.sum(np.abs(fft))
                        speech_ratio = speech_energy / (total_energy + 1e-10)
                        
                        # Require at least 15% of energy in speech range
                        is_speech = is_speech and speech_ratio > 0.15
                except:
                    pass  # Fall back to RMS only if spectral analysis fails
        
        # Update speech history for smoothing
        self.speech_history.append(is_speech)
        
        # Smooth decision based on recent history
        recent_speech_ratio = sum(self.speech_history) / len(self.speech_history)
        smoothed_speech = recent_speech_ratio > self.speech_confidence_threshold
        
        return smoothed_speech, rms
    
    def get_recording_audio(self, start_time, end_time):
        """Extract audio for a specific time range from buffer"""
        recording_chunks = []
        current_time = time.time()
        
        for i, timestamp in enumerate(self.buffer_timestamps):
            # Check if this chunk falls within our recording window
            if start_time <= timestamp <= end_time:
                if i < len(self.audio_buffer):
                    recording_chunks.append(self.audio_buffer[i])
        
        if recording_chunks:
            return np.concatenate(recording_chunks)
        else:
            return np.array([])
    
    def process_audio_chunk(self, audio_chunk):
        """Process audio chunk with continuous state management"""
        current_time = time.time()
        
        # Add to circular buffer
        self.audio_buffer.append(audio_chunk.copy())
        self.buffer_timestamps.append(current_time)
        
        # Detect speech
        is_speech, rms = self.is_speech_detected(audio_chunk)
        
        # State machine for continuous recording
        if self.speech_state == "idle":
            if is_speech:
                print("🎤 Speech detected - Starting recording...")
                self.speech_state = "speech_detected"
                self.speech_start_time = current_time
                self.last_speech_time = current_time
                self.current_recording_start = max(0, current_time - self.pre_speech_buffer_duration)
                self.silence_duration = 0
                
        elif self.speech_state == "speech_detected":
            if is_speech:
                self.last_speech_time = current_time
                self.silence_duration = 0
                
                # Transition to recording after minimum speech duration
                speech_duration = current_time - self.speech_start_time
                if speech_duration >= self.min_speech_duration:
                    self.speech_state = "recording"
                    print(f"📝 Recording confirmed ({speech_duration:.1f}s of speech)")
            else:
                self.silence_duration = current_time - self.last_speech_time
                
                # If silence too long, return to idle
                if self.silence_duration > self.max_silence_duration:
                    print("⏩ False start - returning to idle")
                    self.speech_state = "idle"
                    self.false_starts += 1
                    
        elif self.speech_state == "recording":
            if is_speech:
                self.last_speech_time = current_time
                self.silence_duration = 0
            else:
                self.silence_duration = current_time - self.last_speech_time
                
                # Check if we should stop recording
                if self.silence_duration > self.max_silence_duration:
                    self.speech_state = "post_speech"
                    print(f"🔇 Silence detected ({self.silence_duration:.1f}s) - Processing recording...")
                    
            # Check maximum recording duration
            recording_duration = current_time - self.current_recording_start
            if recording_duration > self.max_recording_duration:
                print(f"⏰ Max recording duration reached ({recording_duration:.1f}s) - Processing...")
                self.speech_state = "post_speech"
                
        elif self.speech_state == "post_speech":
            # Process the recording
            self.process_recording()
            self.speech_state = "idle"
            
        # Periodic maintenance
        self.cleanup_memory()
        self.report_memory_usage()
    
    def process_recording(self):
        """Process completed recording"""
        current_time = time.time()
        
        # Calculate recording bounds
        recording_start = self.current_recording_start
        recording_end = self.last_speech_time + self.post_speech_buffer_duration
        
        # Extract audio from buffer
        audio_data = self.get_recording_audio(recording_start, recording_end)
        
        if len(audio_data) == 0:
            print("❌ No audio data to process")
            return
        
        duration = len(audio_data) / self.samplerate
        
        if duration < self.min_speech_duration:
            print(f"⏩ Recording too short ({duration:.1f}s) - Skipping...")
            return
        
        self.total_recordings += 1
        
        # Submit for async transcription
        print(f"📤 Queuing recording #{self.total_recordings} ({duration:.1f}s) for transcription...")
        self.transcribe_audio_async(audio_data.copy(), self.total_recordings)
    
    def extract_clean_text(self, transcription_output):
        """Extract only the clean text from transcription output"""
        clean_text = ""
        
        if not transcription_output or len(transcription_output) == 0:
            return clean_text
        
        result = transcription_output[0]
        
        # Method 1: Direct text attribute
        if hasattr(result, 'text') and result.text and result.text.strip():
            clean_text = result.text.strip()
            return clean_text
            
        # Method 2: Extract from string representation
        result_str = str(result)
        if 'text=' in result_str:
            try:
                text_match = re.search(r"text='([^']*)'", result_str)
                if text_match and text_match.group(1).strip():
                    clean_text = text_match.group(1).strip()
                    return clean_text
            except:
                pass
        
        # Method 3: Extract from segments (combine all segment text)
        if hasattr(result, 'timestamp') and result.timestamp and result.timestamp.get('segment'):
            segments = result.timestamp['segment']
            segment_texts = []
            
            for segment in segments:
                if isinstance(segment, dict) and 'segment' in segment:
                    segment_text = segment['segment'].strip()
                    if segment_text:
                        segment_texts.append(segment_text)
            
            if segment_texts:
                clean_text = ' '.join(segment_texts)
                return clean_text
        
        return clean_text
    
    def transcribe_audio_async(self, audio_data, recording_number):
        """Async transcription function - saves only clean text"""
        def _transcribe():
            start_time = time.time()
            duration = len(audio_data) / self.samplerate
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                sf.write(temp_file.name, audio_data, self.samplerate)
                temp_path = temp_file.name
            
            try:
                print(f"🔄 Transcribing recording #{recording_number} ({duration:.1f}s)...")
                
                # Clear GPU cache before transcription
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Transcribe
                output = self.asr_model.transcribe([temp_path])
                
                transcription_time = time.time() - start_time
                
                # Extract clean text only
                transcription_text = self.extract_clean_text(output)
                
                # Only proceed if we have valid transcription
                if transcription_text:
                    # Write ONLY the clean text to transcript file
                    with open(self.transcript_file, 'a', encoding='utf-8') as f:
                        f.write(transcription_text + " ")  # Add space between segments
                        f.flush()
                    
                    # Print the transcription (without timestamps)
                    print(f"📝 {transcription_text}")
                    
                    # Save audio
                    self.append_to_audio_file(audio_data, recording_number)
                    
                    print(f"✅ Recording #{recording_number} processed in {transcription_time:.1f}s")
                    self.total_duration += duration
                    
                else:
                    # No valid transcription - likely silence or noise
                    print(f"🔇 Recording #{recording_number} contains no speech - skipping")
                    self.skipped_recordings += 1
                    
            except Exception as e:
                print(f"❌ Transcription error for recording #{recording_number}: {e}")
                
                # Save audio even on error (might be useful for debugging)
                self.append_to_audio_file(audio_data, recording_number)
            
            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Force cleanup after transcription
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Submit to thread pool
        future = self.executor.submit(_transcribe)
        self.transcription_futures.append(future)
        return future
    
    def append_to_audio_file(self, audio_data, recording_number):
        """Append audio data to master file"""
        try:
            if os.path.exists(self.audio_file):
                existing_audio, _ = sf.read(self.audio_file)
                # Add small silence between recordings
                silence = np.zeros(int(0.5 * self.samplerate))
                combined_audio = np.concatenate([existing_audio, silence, audio_data])
            else:
                combined_audio = audio_data
            
            sf.write(self.audio_file, combined_audio, self.samplerate)
            
        except Exception as e:
            print(f"❌ Error saving audio for recording #{recording_number}: {e}")
            backup_name = f"backup_audio_{recording_number}_{datetime.now().strftime('%H%M%S')}.wav"
            try:
                sf.write(backup_name, audio_data, self.samplerate)
                print(f"🎵 Saved as backup: {backup_name}")
            except Exception as e2:
                print(f"❌ Backup save failed: {e2}")
    
    def cleanup_memory(self):
        """Efficient memory cleanup"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean up completed transcription futures
        completed_futures = [f for f in self.transcription_futures if f.done()]
        for future in completed_futures:
            try:
                future.result()
            except Exception as e:
                print(f"❌ Transcription future error: {e}")
            finally:
                self.transcription_futures.remove(future)
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        collected = gc.collect()
        
        self.last_cleanup = current_time
    
    def report_memory_usage(self):
        """Report memory usage periodically"""
        current_time = time.time()
        if current_time - self.last_memory_report < self.memory_report_interval:
            return
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            print(f"\n📊 Memory: {memory_mb:.1f}MB | State: {self.speech_state} | Recordings: {self.total_recordings} | Skipped: {self.skipped_recordings} | False starts: {self.false_starts}", end="")
            
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_free = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024 / 1024
                print(f" | GPU: {gpu_allocated:.1f}MB")
            else:
                print()
            
            print(f"⏱️ Total transcribed: {self.total_duration:.1f}s | Buffer: {len(self.audio_buffer)}/{self.audio_buffer.maxlen} chunks\n")
            
        except Exception as e:
            print(f"❌ Memory report error: {e}")
        
        self.last_memory_report = current_time
    
    def calibrate_thresholds(self):
        """Calibrate speech detection thresholds"""
        print("🔧 Calibrating speech detection thresholds...")
        print("Please remain quiet for 3 seconds...")
        
        # Record background noise
        noise_samples = []
        for _ in range(30):  # 3 seconds of 100ms chunks
            try:
                audio_chunk = self.audio_queue.get(timeout=0.2)
                noise_samples.append(self.calculate_rms(audio_chunk.flatten()))
            except queue.Empty:
                continue
        
        if noise_samples:
            avg_noise = np.mean(noise_samples)
            max_noise = np.max(noise_samples)
            std_noise = np.std(noise_samples)
            
            # Set thresholds based on noise floor
            self.silence_threshold = avg_noise + 2 * std_noise
            self.speech_threshold = max(avg_noise + 4 * std_noise, max_noise * 2, 0.01)
            
            # Cap thresholds to reasonable values
            self.silence_threshold = min(self.silence_threshold, 0.02)
            self.speech_threshold = min(self.speech_threshold, 0.08)
            
            print(f"📊 Noise floor: {avg_noise:.4f} ± {std_noise:.4f}")
            print(f"🎯 Silence threshold: {self.silence_threshold:.4f}")
            print(f"🎯 Speech threshold: {self.speech_threshold:.4f}")
            
            self.is_calibrated = True
        else:
            print("❌ Calibration failed - using default thresholds")
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback function"""
        if status:
            print(f"⚠️ Audio status: {status}")
        
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            print("⚠️ Audio queue full - dropping frame")
    
    def start_listening(self):
        """Start continuous listening and processing"""
        print(f"🎙️ Continuous Voice Activity Detection started!")
        print(f"📄 Transcript: {self.transcript_file}")
        print(f"🎵 Audio: {self.audio_file}")
        print(f"🔊 Using {'WebRTC VAD + RMS' if self.use_webrtc_vad else 'RMS-based'} detection")
        print("🎤 Listening continuously - speak naturally!")
        print("⌨️  Press Ctrl+C to stop\n")
        
        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                # Auto-calibrate thresholds
                if not self.is_calibrated:
                    self.calibrate_thresholds()
                
                print("🎯 Ready for speech detection!\n")
                
                while True:
                    try:
                        # Get audio chunk
                        audio_chunk = self.audio_queue.get(timeout=0.5)
                        self.process_audio_chunk(audio_chunk.flatten())
                        
                    except queue.Empty:
                        continue
                        
        except KeyboardInterrupt:
            print("\n🛑 Stopping recording...")
            
            # Process any ongoing recording
            if self.speech_state in ["recording", "speech_detected"]:
                print("🔄 Processing final recording...")
                self.process_recording()
            
            # Wait for all transcriptions to complete
            print("⏳ Waiting for transcriptions to complete...")
            for future in self.transcription_futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    print(f"❌ Final transcription error: {e}")
            
            # Final cleanup
            self.executor.shutdown(wait=True)
            
            # Add final session summary to transcript
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\nEND OF TRANSCRIPTION")
            
            print(f"\n✅ Session complete!")
            print(f"📊 Total recordings: {self.total_recordings}")
            print(f"⏱️ Total duration: {self.total_duration:.1f}s")
            print(f"🔇 Skipped (no speech): {self.skipped_recordings}")
            print(f"🚫 False starts: {self.false_starts}")
            print(f"📄 Clean transcript saved: {self.transcript_file}")
            if os.path.exists(self.audio_file):
                print(f"🎵 Audio saved: {self.audio_file}")

def main():
    print("🎤 Initializing Continuous Voice Activity Recorder...")
    
    # Check for optional WebRTC VAD
    if not WEBRTC_AVAILABLE:
        print("💡 For better voice detection, install webrtcvad:")
        print("   pip install webrtcvad")
        print("   Using RMS-based detection for now...")
    else:
        print("✅ WebRTC VAD available for enhanced detection")
    
    # Create recorder
    recorder = ContinuousVoiceActivityRecorder()
    
    # Start listening
    recorder.start_listening()

if __name__ == "__main__":
    main()