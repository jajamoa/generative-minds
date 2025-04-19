"""
QA Extractor - Extracts explicit and implicit question-answer pairs from interview transcripts using LLM.
"""

import os
import json
from pathlib import Path
import re
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class QAExtractor:
    """
    A class to extract question-answer pairs from interview transcripts using LLM.
    This includes both explicit and implicit QA pairs.
    """
    
    def __init__(self, raw_data_dir, processed_data_dir):
        """
        Initialize QA Extractor with directories for raw and processed data.
        
        Args:
            raw_data_dir (str): Path to directory containing raw transcript files
            processed_data_dir (str): Path to directory where processed QA pairs will be saved
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create processed directory if it doesn't exist
        if not self.processed_data_dir.exists():
            self.processed_data_dir.mkdir(parents=True)
    
    def _clean_transcript(self, text):
        """
        Clean the transcript text by removing extra spaces and formatting.
        
        Args:
            text (str): Raw transcript text
            
        Returns:
            str: Cleaned transcript text
        """
        # Remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any timestamps or speaker labels if present
        text = re.sub(r'\[\d+:\d+:\d+\]', '', text)
        
        return text.strip()
    
    def _extract_qa_pairs_with_llm(self, transcript_text, interview_name):
        """
        Extract question-answer pairs from transcript using LLM.
        This includes both explicit questions and implicit questions derived from narrative content.
        
        Args:
            transcript_text (str): Cleaned transcript text
            interview_name (str): Name of the interviewee
            
        Returns:
            list: List of dictionaries containing question-answer pairs
        """
        # If transcript is too long, split it into chunks
        max_chunk_length = 10000  # Ensure we stay within token limits
        if len(transcript_text) > max_chunk_length:
            print(f"Transcript for {interview_name} is too long ({len(transcript_text)} chars). Splitting into chunks.")
            chunks = [transcript_text[i:i+max_chunk_length] for i in range(0, len(transcript_text), max_chunk_length)]
            all_qa_pairs = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                chunk_pairs = self._process_transcript_chunk(chunk, interview_name, i+1)
                all_qa_pairs.extend(chunk_pairs)
            
            return all_qa_pairs
        else:
            return self._process_transcript_chunk(transcript_text, interview_name)
    
    def _process_transcript_chunk(self, transcript_chunk, interview_name, chunk_num=None):
        """
        Process a single chunk of transcript text to extract QA pairs.
        
        Args:
            transcript_chunk (str): Chunk of transcript text
            interview_name (str): Name of the interviewee
            chunk_num (int, optional): Chunk number for logging purposes
            
        Returns:
            list: List of dictionaries containing question-answer pairs with source text
        """
        chunk_info = f" (chunk {chunk_num})" if chunk_num else ""
        
        # Prepare system message
        system_message = """
        You are an AI specialized in extracting both explicit and implicit question-answer pairs from interview transcripts.
        
        IMPORTANT: In unstructured oral interviews, many important statements are responses to IMPLICIT questions.
        Your task is to identify these statements and reconstruct what question they are answering.
        
        For each statement or segment that contains meaningful information:
        1. Infer the implicit question that would naturally prompt such a response
        2. Structure the existing content as the answer
        3. Preserve the original meaning and context
        
        Organize QA pairs into these categories when appropriate:
        - Housing Experience & Access
        - Race, Gender, and Identity  
        - Community & Social Change
        - Civic & Policy Views
        
        Each QA pair should include:
        - A clear, well-formulated question (even if implicit in the original text)
        - The corresponding answer from the transcript
        - The source text this was derived from (exact text from transcript)
        
        Format your response as a valid JSON object with a key 'qa_pairs' that contains an array of objects.
        Each object should have 'question', 'answer', 'category', and 'source_text' fields.
        
        Example format:
        {
          "qa_pairs": [
            {
              "question": "Why did your family move to the Sunset in 1966?",
              "answer": "My grandmother gave my parents $7,000 for the down payment on a $32,000 house on 15th and Taraval because they were expecting a fourth child.",
              "category": "Housing Experience & Access",
              "source_text": "in 1966 our family were my parents were expecting the third uh the fourth child and so my grandmother gave them seven thousand down for the down payment on a 32 000 home on 15th and Taraval"
            }
          ]
        }
        """
        
        # Prepare prompt for LLM
        prompt = f"""
        Below is a transcript from an interview with {interview_name}{chunk_info}. 
        
        Please extract both explicit and implicit question-answer pairs from this transcript. For each important statement:
        1. Create a precise question that the statement is answering (even if the question wasn't explicitly asked)
        2. Structure the statement as an answer
        3. Include the exact source text from the transcript
        4. Assign it to one of the following categories:
           - Housing Experience & Access
           - Race, Gender, and Identity
           - Community & Social Change
           - Civic & Policy Views
        
        IMPORTANT INSTRUCTIONS:
        - Identify meaningful insights even when they're buried in narrative
        - Create natural-sounding questions that would prompt the given answers
        - Keep answers concise but complete, capturing the key information
        - Include the exact source text so we can trace back to the original transcript
        - If multiple consecutive sentences form one coherent answer, include them together
        
        Transcript:
        {transcript_chunk}
        """
        
        try:
            # Call OpenAI API
            print(f"Calling OpenAI API for {interview_name}{chunk_info}...")
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            print(f"Received response, length: {len(content)} chars")
            
            # Save the raw response for debugging
            debug_dir = self.processed_data_dir / interview_name / "debug"
            debug_dir.mkdir(exist_ok=True, parents=True)
            debug_file = debug_dir / f"response_chunk_{chunk_num}.json" if chunk_num else debug_dir / "response.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            try:
                # Parse JSON response
                result = json.loads(content)
                
                # Check if result has 'qa_pairs' key
                if 'qa_pairs' in result:
                    qa_pairs = result['qa_pairs']
                    print(f"Successfully extracted {len(qa_pairs)} QA pairs from {interview_name}{chunk_info}")
                    return qa_pairs
                else:
                    # Try to find any key that contains an array of objects with question/answer fields
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Check if the first item has question and answer fields
                            if isinstance(value[0], dict) and 'question' in value[0] and 'answer' in value[0]:
                                print(f"Found QA pairs under key '{key}' instead of 'qa_pairs'")
                                return value
                    
                    # If we have a list directly, check if it contains question/answer objects
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'question' in result[0] and 'answer' in result[0]:
                            return result
                    
                    # If all else fails, create a manual structure from the result
                    print(f"Could not find QA pairs in standard format. Result keys: {list(result.keys())}")
                    print(f"Response content preview: {content[:200]}...")
                    
                    # Last resort: return empty list
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for {interview_name}{chunk_info}: {str(e)}")
                print(f"Response content preview: {content[:200]}...")
                
                # Try to salvage by fixing common JSON issues
                try:
                    # Try to find JSON between curly braces
                    json_pattern = r'({[\s\S]*})'
                    match = re.search(json_pattern, content)
                    if match:
                        json_str = match.group(1)
                        result = json.loads(json_str)
                        if 'qa_pairs' in result:
                            print("Successfully extracted JSON after fixing format")
                            return result['qa_pairs']
                except Exception:
                    pass
                
                # Still failed, return empty
                return []
                
        except Exception as e:
            print(f"Error calling OpenAI API for {interview_name}{chunk_info}: {str(e)}")
            return []
    
    def process_transcript(self, file_name):
        """
        Process a single transcript file to extract QA pairs.
        
        Args:
            file_name (str): Name of the transcript file to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Get interview name from file name
            interview_name = file_name.replace('_en_subs.txt', '')
            
            # Create output directory for this interview
            output_dir = self.processed_data_dir / interview_name
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            
            # Read transcript file
            file_path = self.raw_data_dir / file_name
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            # Clean transcript
            cleaned_text = self._clean_transcript(transcript_text)
            
            # Extract QA pairs
            qa_pairs = self._extract_qa_pairs_with_llm(cleaned_text, interview_name)
            
            # Save QA pairs to JSON file
            output_file = output_dir / 'qa_pairs.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                'interview_name': interview_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_qa_pairs': len(qa_pairs),
                'file_processed': file_name,
                'categories': list(set([qa.get('category', 'Uncategorized') for qa in qa_pairs]))
            }
            
            metadata_file = output_dir / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Create category-based files
            self._save_categorized_qa_pairs(qa_pairs, output_dir)
            
            print(f"Successfully processed {interview_name}, extracted {len(qa_pairs)} QA pairs.")
            return True
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            return False
    
    def _save_categorized_qa_pairs(self, qa_pairs, output_dir):
        """
        Save QA pairs organized by category.
        
        Args:
            qa_pairs (list): List of QA pair dictionaries
            output_dir (Path): Directory to save categorized files
        """
        # Group QA pairs by category
        categories = {}
        for qa in qa_pairs:
            category = qa.get('category', 'Uncategorized')
            if category not in categories:
                categories[category] = []
            categories[category].append(qa)
        
        # Save each category to a separate file
        for category, pairs in categories.items():
            if not pairs:
                continue
                
            # Create a valid filename from the category
            filename = re.sub(r'[^\w\s-]', '', category).strip().replace(' ', '_').lower()
            category_file = output_dir / f"{filename}_qa_pairs.json"
            
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(pairs)} QA pairs for category: {category}")
    
    def process_all_transcripts(self):
        """
        Process all transcript files in the raw data directory.
        
        Returns:
            dict: Statistics about the processing
        """
        # Get all transcript files
        transcript_files = [f.name for f in self.raw_data_dir.glob('*_en_subs.txt')]
        
        if not transcript_files:
            print("No transcript files found in the raw data directory.")
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        # Process each file
        successful = 0
        failed = 0
        
        for file_name in transcript_files:
            print(f"Processing {file_name}...")
            result = self.process_transcript(file_name)
            
            if result:
                successful += 1
            else:
                failed += 1
        
        # Return statistics
        stats = {
            'total': len(transcript_files),
            'successful': successful,
            'failed': failed
        }
        
        print(f"Processing complete. Total: {stats['total']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
        return stats


if __name__ == "__main__":
    # Define paths
    raw_data_dir = "data/housing_choice_community_interviews/raw"
    processed_data_dir = "data/housing_choice_community_interviews/processed"
    
    # Create and run extractor
    extractor = QAExtractor(raw_data_dir, processed_data_dir)
    extractor.process_all_transcripts() 