#!/usr/bin/env python3
"""
Belief Graph Evolution Analysis

This script visualizes the step-by-step construction of personal belief graphs from interview data.
It supports two methods of analysis:
1. Single QA processing - analyzing each QA pair individually
2. Batch QA processing - analyzing all QA pairs at once
"""

import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import dashscope


def find_project_root() -> Path:
    """Find the project root directory by looking for .env file"""
    current = Path(__file__).resolve()
    while current.parent != current:  # Stop at root directory
        if (current / '.env').exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent  # Fallback to parent of script directory


def setup_output_dir(script_dir: Path, interview_id: str) -> Path:
    """Setup output directory in the script directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / 'output' / f"{interview_id}_{timestamp}"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'method1').mkdir(exist_ok=True)
    (output_dir / 'method2').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    return output_dir


def save_graph_data(graph: Dict, output_path: Path, name: str):
    """Save graph data to JSON file"""
    with open(output_path / f"{name}.json", 'w') as f:
        json.dump(graph, f, indent=2)


def save_visualization(mermaid_graph: str, output_path: Path, name: str):
    """Save Mermaid graph visualization"""
    with open(output_path / f"{name}.mmd", 'w') as f:
        f.write(mermaid_graph)


[Previous BeliefGraphExtractor class and other functions remain unchanged...]


def analyze_interview(interview_data: Dict, output_dir: Path) -> Tuple[Dict, Dict]:
    """Analyze an interview using both methods and save results"""
    # Initialize extractor
    extractor = BeliefGraphExtractor()
    
    # Extract QA pairs
    qa_pairs = extract_qa_pairs(interview_data)
    
    interview_id = interview_data.get('id', 'Unknown ID')
    print(f"Analyzing interview {interview_id}")
    print(f"Number of QA pairs: {len(qa_pairs)}")
    print(f"Saving results to: {output_dir}")
    
    # Method 1: Single QA approach
    print("\nMethod 1: Processing each QA pair individually...")
    individual_graphs = []
    method1_dir = output_dir / 'method1'
    
    for i, qa_pair in enumerate(qa_pairs):
        print(f"Processing QA pair {i+1}/{len(qa_pairs)}...")
        graph = extractor.extract_single_qa(qa_pair)
        if graph:
            individual_graphs.append(graph)
            save_graph_data(graph, method1_dir, f"qa_pair_{i+1}")
            
    merged_graph = merge_belief_graphs(individual_graphs)
    save_graph_data(merged_graph, method1_dir, "merged")
    print(f"\nMerged graph has {len(merged_graph['nodes'])} nodes and {len(merged_graph['edges'])} edges")
    
    # Method 2: Batch QA approach
    print("\nMethod 2: Processing all QA pairs at once...")
    method2_dir = output_dir / 'method2'
    batch_graph = extractor.extract_batch_qa(qa_pairs)
    if batch_graph:
        save_graph_data(batch_graph, method2_dir, "batch")
        print(f"Batch graph has {len(batch_graph['nodes'])} nodes and {len(batch_graph['edges'])} edges")
    
    return merged_graph, batch_graph


def main():
    """Main execution function"""
    # Set data path
    project_root = find_project_root()
    data_path = project_root / 'prototypes' / 'data' / 'zoning' / 'qas.json'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
        
    # Load interview data
    interviews = load_interview_data(data_path)
    print(f"Loaded {len(interviews)} interviews")
    
    if not interviews:
        print("No interviews found!")
        return
        
    # Select first interview for demonstration
    interview = interviews[0]
    interview_id = interview.get('id', 'unknown')
    
    # Setup output directory
    script_dir = Path(__file__).parent
    output_dir = setup_output_dir(script_dir, interview_id)
    
    # Analyze interview
    merged_graph, batch_graph = analyze_interview(interview, output_dir)
    
    # Generate and save visualizations
    vis_dir = output_dir / 'visualizations'
    
    if merged_graph:
        print("\nGenerating visualizations for Method 1 (Single QA)...")
        mermaid_graphs = visualize_graph_evolution(merged_graph, steps=5)
        for i, graph in enumerate(mermaid_graphs, 1):
            save_visualization(graph, vis_dir, f"method1_step_{i}")
            print(f"\nStep {i}/5 saved")
            
    if batch_graph:
        print("\nGenerating visualizations for Method 2 (Batch QA)...")
        mermaid_graphs = visualize_graph_evolution(batch_graph, steps=5)
        for i, graph in enumerate(mermaid_graphs, 1):
            save_visualization(graph, vis_dir, f"method2_step_{i}")
            print(f"\nStep {i}/5 saved")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()