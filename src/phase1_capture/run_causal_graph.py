"""
Run script for Causal Graph building process.
This script builds causal graphs from extracted QA pairs.
"""

import argparse
import os
from pathlib import Path
from causal_graph_builder import CausalGraphBuilder

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Build causal graphs from QA pairs")
    
    # Add arguments
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default="data/housing_choice_community_interviews/processed",
        help="Directory containing processed QA pairs"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save causal graphs (defaults to processed-dir)"
    )
    
    parser.add_argument(
        "--single-interview", 
        type=str, 
        default=None,
        help="Process only a specific interview (optional)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the causal graph building process.
    """
    # Parse arguments
    args = parse_args()
    
    # Create graph builder
    builder = CausalGraphBuilder(args.processed_dir, args.output_dir)
    
    # Run graph building
    if args.single_interview:
        print(f"Building causal graph for single interview: {args.single_interview}")
        builder.process_interview(args.single_interview)
    else:
        print("Building causal graphs for all interviews...")
        builder.process_all_interviews()
    
    print("Causal graph building process completed.")

if __name__ == "__main__":
    main() 