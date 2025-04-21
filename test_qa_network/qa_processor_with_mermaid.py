import json
import os
from causal_graph_schema import CausalGraph

def load_qa_data(file_path):
    """Load QA pair data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def ensure_mermaid_dir():
    """Ensure mermaid output directory exists"""
    mermaid_dir = "mermaid_snapshots"
    if not os.path.exists(mermaid_dir):
        os.makedirs(mermaid_dir)
    return mermaid_dir

def save_mermaid_snapshot(graph, step_count):
    """Save a mermaid snapshot with incremental filename"""
    mermaid_dir = ensure_mermaid_dir()
    filename = f"graph_step_{step_count:03d}.mmd"
    filepath = os.path.join(mermaid_dir, filename)
    graph.export_mermaid(filepath)
    print(f"Saved mermaid snapshot: {filepath}")

def process_qa_to_causal_graph(input_file, graph):
    """Process QA pairs and update the causal graph"""
    print(f"Processing {input_file}...")
    
    # Load QA data
    qa_pairs = load_qa_data(input_file)
    
    # Process each QA pair
    for i, qa_pair in enumerate(qa_pairs):
        print(f"Processing QA pair {i+1}/{len(qa_pairs)}")
        graph.update_from_qa(qa_pair)
        
        # Save mermaid snapshot every 3 steps
        if (i + 1) % 3 == 0:
            save_mermaid_snapshot(graph, i + 1)
        
        # input()
    
    # Save final snapshot if not already saved
    if len(qa_pairs) % 3 != 0:
        save_mermaid_snapshot(graph, len(qa_pairs))
    
    print(f"Processed {len(qa_pairs)} QA pairs")
    return graph

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        from causal_graph_schema import CausalGraph
        
        # Process from file
        input_file = sys.argv[1]
        
        # Create graph structure
        graph = CausalGraph()
        
        # Process QA pairs
        process_qa_to_causal_graph(input_file, graph)
        
        # Save final results
        graph.save_to_files("causal_graph")
    else:
        print("Please provide a QA pair file path")
        print("Usage: python qa_processor_with_mermaid.py qa_pairs.json") 