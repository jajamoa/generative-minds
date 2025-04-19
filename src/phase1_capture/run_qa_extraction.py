"""
Run script for QA Extraction process.
This script allows for easy execution of the QA extraction process from command line.
"""

import argparse
import os
from pathlib import Path
from qa_extractor import QAExtractor

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Extract QA pairs from interview transcripts using LLM")
    
    # Add arguments
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        default="data/housing_choice_community_interviews/raw",
        help="Directory containing raw transcript files"
    )
    
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default="data/housing_choice_community_interviews/processed",
        help="Directory to save processed QA pairs"
    )
    
    parser.add_argument(
        "--single-file", 
        type=str, 
        default=None,
        help="Process only a specific file (optional)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the QA extraction process.
    """
    # Parse arguments
    args = parse_args()
    
    # Create extractor
    extractor = QAExtractor(args.raw_dir, args.processed_dir)
    
    # Run extraction
    if args.single_file:
        if not os.path.exists(os.path.join(args.raw_dir, args.single_file)):
            print(f"Error: File {args.single_file} not found in {args.raw_dir}")
            return
        
        print(f"Processing single file: {args.single_file}")
        extractor.process_transcript(args.single_file)
    else:
        print("Processing all transcript files...")
        extractor.process_all_transcripts()
    
    print("QA extraction process completed.")

if __name__ == "__main__":
    main() 