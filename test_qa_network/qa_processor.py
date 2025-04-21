import json
from causal_graph_schema import CausalGraph

def load_qa_data(file_path):
    """Load QA pair data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def process_qa_to_causal_graph(input_file, graph):
    """Process QA pairs and update the causal graph"""
    print(f"Processing {input_file}...")
    
    # Load QA data
    qa_pairs = load_qa_data(input_file)
    
    # Process each QA pair
    for i, qa_pair in enumerate(qa_pairs):
        print(f"Processing QA pair {i+1}/{len(qa_pairs)}")
        graph.update_from_qa(qa_pair)
        input();
    
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
        
        # Save results
        graph.save_to_files("causal_graph")
    else:
        print("Please provide a QA pair file path")
        print("Usage: python qa_processor.py qa_pairs.json") 