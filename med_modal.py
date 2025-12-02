from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import os
from typing import List, Dict
from datetime import datetime

class DetailedMedicalNotesExtractor:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print("🔄 Loading model for detailed notes extraction...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()
        
        # Context settings - more generous for capturing details
        self.max_context = 7000      # Safe context limit
        self.chunk_size = 10         # Even smaller chunks for maximum detail capture
        self.overlap_lines = 60      # More overlap to ensure no details lost
        
        print(f"✅ Model loaded on: {self.model.device}")
        print(f"📊 Using {self.chunk_size} lines per chunk with {self.overlap_lines} line overlap")
    
    def get_detailed_notes_prompt(self):
        """Prompt for extracting ALL details without missing anything"""
        return """You are a medical scribe creating COMPREHENSIVE detailed notes. Your job is to extract EVERY SINGLE detail from this medical conversation. Do NOT summarize, compress, or skip anything.

CRITICAL INSTRUCTIONS:
- Extract EVERY symptom, complaint, and concern mentioned
- Include ALL medical explanations word-for-word when important
- List EVERY product name, brand, percentage, and specific instruction
- Capture ALL dosages, application methods, timing, and frequencies
- Include ALL doctor's personal anecdotes and examples
- Note EVERY warning, contraindication, and safety instruction
- Document ALL patient education points and explanations
- Include specific quotes when they clarify important points

FORMAT (include everything under each section):

PATIENT PRESENTATION & CONCERNS:
• [Every symptom, complaint, and desire mentioned]
• [Exact patient descriptions and words used]
• [Specific concerns about appearance, texture, etc.]

DOCTOR'S COMPLETE EXPLANATIONS:
• [All medical concepts explained in detail]
• [Definitions of medical terms provided]
• [Mechanisms of action described]
• [Personal anecdotes and experiences shared]
• [Analogies and examples used]

SPECIFIC PRODUCT RECOMMENDATIONS:
• [Every brand name, product name, and formulation]
• [Exact percentages and concentrations mentioned]
• [Specific application instructions for each product]
• [Layering order and timing details]

DETAILED TREATMENT PROTOCOLS:
• [Complete morning routine with all steps]
• [Complete evening routine with all steps]
• [Exact application amounts (pea-sized, etc.)]
• [Application techniques (dry skin vs damp skin)]

SAFETY WARNINGS & CONTRAINDICATIONS:
• [All pregnancy warnings]
• [Irritation precautions]
• [What NOT to do]
• [Potential side effects mentioned]

PATIENT EDUCATION PROVIDED:
• [All skin science explained]
• [Hydration vs dryness concepts]
• [How ingredients work]
• [Environmental factors discussed]

SPECIFIC INSTRUCTIONS & TIPS:
• [Application timing and frequency]
• [Product layering rules]
• [Seasonal considerations]
• [Geographic/climate factors]

Extract EVERYTHING. If the doctor mentions a specific brand, include it. If they give an exact percentage, include it. If they explain how something works, include the full explanation. Be extremely thorough."""
    
    def get_rolling_notes_prompt(self):
        """Prompt for combining detailed notes without losing ANY information"""
        return """You are a medical scribe. I will give you:
1. EXISTING COMPREHENSIVE NOTES from earlier conversation
2. NEW DETAILED CONTENT from current section

Your task: Merge these into COMPLETE COMPREHENSIVE NOTES. CRITICAL: Do NOT lose ANY details, specifications, or information.

MERGING RULES:
• PRESERVE every brand name, percentage, and product detail
• KEEP all application instructions and techniques
• MAINTAIN all safety warnings and contraindications
• INCLUDE all doctor's explanations and patient education
• ADD new information to existing categories
• NEVER remove or compress existing details
• MERGE similar items but keep all specifics
• PRESERVE exact quotes and specific instructions

Example of CORRECT merging:
OLD: "Apply retinoid on dry skin"
NEW: "Use Cerave Resurfacing Retinol Serum, 0.3% concentration, pea-sized amount"
MERGED: "Apply retinoid on dry skin - specific products: Cerave Resurfacing Retinol Serum (0.3% concentration), Neutrogena Retinol Oil (0.3%), use pea-sized amount"

The final notes must be MORE detailed than either input, not less. Preserve EVERYTHING."""
    
    def get_hierarchical_notes_prompt(self):
        """Prompt for combining multiple detailed note sections"""
        return """You are a medical scribe. I will give you multiple sets of detailed medical notes from different parts of a conversation.

Your task: Combine ALL these detailed notes into ONE comprehensive set that preserves every important detail.

Requirements:
• Merge similar sections without losing any information
• Keep all symptoms, treatments, instructions
• Preserve all medical explanations and terminology
• Maintain all follow-up instructions
• Include all patient concerns and doctor responses
• Organize logically but keep everything

DO NOT SUMMARIZE - PRESERVE ALL DETAILS. The goal is comprehensive medical documentation."""
    
    def get_final_structure_prompt(self):
        """Final structuring prompt that preserves ALL details"""
        return """You are a medical scribe. Take these comprehensive detailed notes and organize them into the requested medical note format while preserving EVERY SINGLE DETAIL.

CRITICAL: Do NOT summarize, compress, or remove ANY information. Include ALL:
- Brand names and product specifications
- Exact percentages and concentrations
- Complete application instructions
- All safety warnings and contraindications
- Every explanation and education point
- All personal anecdotes and examples
- Specific quotes when they add clarity

Use this structure but EXPAND each section with ALL available details:

PATIENT VISIT NOTES

CHIEF COMPLAINT & SYMPTOMS:
• [All patient concerns with exact descriptions]
• [Specific desires and goals mentioned]
• [Any concerns about appearance, texture, etc.]

HISTORY & BACKGROUND:
• [All relevant background information]
• [Doctor's personal experiences shared]
• [Environmental factors discussed]
• [Previous treatment history if mentioned]

PHYSICAL EXAMINATION:
• [All examination details OR state "No physical examination performed"]
• [Any observations made during consultation]

ASSESSMENT & EXPLANATIONS:
• [COMPLETE medical explanations provided]
• [All definitions and medical concepts taught]
• [Mechanisms of action explained]
• [Skin science education provided]
• [Include specific quotes for important corrections]

TREATMENT PLAN:
• [DETAILED morning routine with ALL products and instructions]
• [DETAILED evening routine with ALL products and instructions]
• [ALL product names, brands, percentages, concentrations]
• [EXACT application amounts and techniques]
• [COMPLETE layering instructions]

PATIENT EDUCATION:
• [ALL educational content provided]
• [Every concept explained]
• [All analogies and examples used]
• [Hydration vs dryness explanations]
• [Skin type dynamics discussed]

LIFESTYLE MODIFICATIONS:
• [ALL lifestyle recommendations]
• [Hydration advice]
• [Environmental considerations]
• [Seasonal adjustments mentioned]

FOLLOW-UP CARE:
• [Any follow-up mentioned OR state "None specified"]

PATIENT QUESTIONS & RESPONSES:
• [All questions and responses OR state if none]

IMPORTANT NOTES:
• [ALL safety warnings and contraindications]
• [ALL precautions and potential side effects]
• [Important application tips and techniques]
• [Any special considerations]

This must be the MOST detailed and comprehensive medical note possible. Include everything."""
    
    def clean_and_split_transcript(self, file_path: str) -> List[str]:
        """Clean transcript and split into lines"""
        print(f"📖 Reading and cleaning transcript from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []
        
        # Clean the content
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if (line and 
                not line.startswith('[') and 
                not line.startswith('=') and
                'Transcription completed' not in line and
                'END OF TRANSCRIPTION' not in line and
                'Conversation Transcript' not in line):
                lines.append(line)
        
        print(f"📄 Processed {len(lines)} lines of content")
        return lines
    
    def count_tokens(self, text: str) -> int:
        """Count tokens safely"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split()) * 1.3  # Rough estimate
    
    def extract_detailed_notes(self, prompt: str, content: str, max_tokens: int = 1200) -> str:
        """Extract detailed notes with error handling and higher token limits"""
        
        full_prompt = f"{prompt}\n\nMEDICAL CONVERSATION:\n{content}"
        
        # Check token count - preserve more content for detailed extraction
        if self.count_tokens(full_prompt) > self.max_context:
            # Truncate content if needed, but preserve much more than summary mode
            words = content.split()
            target_length = self.max_context - self.count_tokens(prompt) - 200  # More buffer
            while self.count_tokens(' '.join(words)) > target_length and len(words) > 100:
                words = words[:-3]  # Remove fewer words to preserve maximum detail
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
                    max_new_tokens=max_tokens,  # Higher token limit for detailed notes
                    do_sample=False,
                    repetition_penalty=1.05,    # Very low penalty to allow detailed repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.05,           # Very low temperature for consistency
                    length_penalty=0.8          # Encourage longer, more detailed output
                )
            
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = generation[0][input_len:]
            notes = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return notes.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"[Error extracting notes for this section: {str(e)[:100]}]"
    
    def rolling_detailed_notes_approach(self, lines: List[str]) -> str:
        """
        Rolling approach but for detailed notes instead of summaries
        """
        print("\n🔄 Using ROLLING DETAILED NOTES approach...")
        print(f"📊 Processing {len(lines)} lines in chunks of {self.chunk_size}")
        
        current_notes = ""
        total_chunks = math.ceil(len(lines) / self.chunk_size)
        
        for i in range(0, len(lines), self.chunk_size):
            chunk_num = (i // self.chunk_size) + 1
            
            # Add overlap from previous chunk
            start_idx = max(0, i - self.overlap_lines)
            end_idx = i + self.chunk_size
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = '\n'.join(chunk_lines)
            
            print(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_lines)} lines)...")
            
            if not current_notes:
                # First chunk - create initial detailed notes
                prompt = self.get_detailed_notes_prompt()
                new_notes = self.extract_detailed_notes(prompt, chunk_content, max_tokens=1500)
            else:
                # Subsequent chunks - combine with existing notes
                prompt = self.get_rolling_notes_prompt()
                content = f"EXISTING DETAILED NOTES:\n{current_notes}\n\nNEW CONTENT:\n{chunk_content}"
                new_notes = self.extract_detailed_notes(prompt, content, max_tokens=1800)
            
            current_notes = new_notes
            print(f"✅ Chunk {chunk_num} processed. Notes length: {len(current_notes)} chars")
        
        print("🎯 Rolling detailed notes complete! Creating final structured notes...")
        
        # Create final structured notes
        final_prompt = self.get_final_structure_prompt()
        final_content = f"COMPREHENSIVE DETAILED NOTES:\n{current_notes}"
        final_notes = self.extract_detailed_notes(final_prompt, final_content, max_tokens=2000)
        
        return final_notes
    
    def hierarchical_detailed_notes_approach(self, lines: List[str]) -> str:
        """
        Hierarchical approach for detailed notes
        """
        print("\n🔄 Using HIERARCHICAL DETAILED NOTES approach...")
        print(f"📊 Processing {len(lines)} lines in chunks of {self.chunk_size}")
        
        def recursive_combine_notes(content_list: List[str], level: int = 1) -> str:
            if len(content_list) == 1:
                return content_list[0]
            
            print(f"📊 Level {level}: Processing {len(content_list)} sections...")
            
            combined_notes = []
            items_per_batch = 3  # Combine fewer items to preserve detail
            
            for i in range(0, len(content_list), items_per_batch):
                batch = content_list[i:i + items_per_batch]
                batch_num = (i // items_per_batch) + 1
                total_batches = math.ceil(len(content_list) / items_per_batch)
                
                print(f"  🔄 Level {level}, Batch {batch_num}/{total_batches}...")
                
                if level == 1:
                    # First level - extract detailed notes from original content
                    prompt = self.get_detailed_notes_prompt()
                    content = f"MEDICAL CONVERSATION:\n\n" + "\n\n".join(batch)
                else:
                    # Higher levels - combine detailed notes
                    prompt = self.get_hierarchical_notes_prompt()
                    content = f"DETAILED NOTES SECTIONS:\n\n" + "\n\n--- SECTION BREAK ---\n\n".join(batch)
                
                notes = self.extract_detailed_notes(prompt, content, max_tokens=1500)
                combined_notes.append(notes)
            
            print(f"✅ Level {level} complete: {len(content_list)} -> {len(combined_notes)} note sections")
            
            # If we still have multiple note sections, recurse
            if len(combined_notes) > 1:
                return recursive_combine_notes(combined_notes, level + 1)
            else:
                return combined_notes[0]
        
        # Create initial chunks with overlap
        chunks = []
        total_chunks = math.ceil(len(lines) / self.chunk_size)
        
        for i in range(0, len(lines), self.chunk_size):
            start_idx = max(0, i - self.overlap_lines)
            end_idx = i + self.chunk_size
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = '\n'.join(chunk_lines)
            chunks.append(chunk_content)
        
        print(f"📑 Created {len(chunks)} initial chunks with overlap")
        
        # Recursively combine detailed notes
        comprehensive_notes = recursive_combine_notes(chunks)
        
        print("🎯 Hierarchical detailed notes complete! Creating final structured notes...")
        
        # Create final structured notes
        final_prompt = self.get_final_structure_prompt()
        final_content = f"COMPREHENSIVE DETAILED NOTES:\n{comprehensive_notes}"
        final_notes = self.extract_detailed_notes(final_prompt, final_content, max_tokens=1500)
        
        return final_notes
    
    def process_transcript(self, file_path: str, method: str = "both") -> Dict[str, str]:
        """Process transcript using specified method(s)"""
        
        # Read and clean transcript
        lines = self.clean_and_split_transcript(file_path)
        
        if not lines:
            print("❌ No valid content found!")
            return {}
        
        results = {}
        
        if method in ["rolling", "both"]:
            print("\n" + "="*60)
            print("🔄 STARTING ROLLING DETAILED NOTES METHOD")
            print("="*60)
            results["rolling"] = self.rolling_detailed_notes_approach(lines)
        
        if method in ["hierarchical", "both"]:
            print("\n" + "="*60)
            print("🔄 STARTING HIERARCHICAL DETAILED NOTES METHOD") 
            print("="*60)
            results["hierarchical"] = self.hierarchical_detailed_notes_approach(lines)
        
        return results
    
    def save_results(self, results: Dict[str, str], base_filename: str):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for method, notes in results.items():
            filename = f"{base_filename}_detailed_notes_{method}_{timestamp}.txt"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"DETAILED MEDICAL NOTES - {method.upper()} METHOD\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(notes)
                    f.write(f"\n\n" + "="*60)
                    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    f.write(f"\nMethod: {method}")
                    f.write(f"\nApproach: Detailed Notes Extraction")
                    f.write(f"\nModel: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
                
                print(f"💾 {method.capitalize()} detailed notes saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error saving {method} notes: {e}")

def main():
    """Main function with method selection"""
    
    print("📝 Detailed Medical Notes Extractor")
    print("="*50)
    print("This tool creates comprehensive detailed notes instead of summaries")
    print("to capture ALL important medical information without compression.")
    
    # Configuration
    transcript_file = "conversation_transcript_20250606_191235.txt"  # Change this
    
    print("\nAvailable methods:")
    print("1. Rolling Detailed Notes (accumulative detailed notes)")
    print("2. Hierarchical Detailed Notes (combine detailed sections)")
    print("3. Both methods for comparison")
    
    method_choice = input("\nChoose method (1/2/3): ").strip()
    
    method_map = {
        "1": "rolling",
        "2": "hierarchical", 
        "3": "both"
    }
    
    method = method_map.get(method_choice, "both")
    
    # Initialize notes extractor
    extractor = DetailedMedicalNotesExtractor()
    
    # Process transcript
    print(f"\n🚀 Extracting detailed notes with {method} method(s)...")
    results = extractor.process_transcript(transcript_file, method)
    
    # Display results
    for method_name, notes in results.items():
        print(f"\n" + "="*80)
        print(f"📝 DETAILED MEDICAL NOTES - {method_name.upper()} METHOD")
        print("="*80)
        print(notes)
        print("="*80)
    
    # Save results
    if results:
        extractor.save_results(results, "medical")
        print(f"\n✅ Processing complete! Generated {len(results)} detailed note set(s).")
        print("📝 These notes preserve all important details instead of summarizing.")
    else:
        print("\n❌ No notes generated!")

if __name__ == "__main__":
    main()