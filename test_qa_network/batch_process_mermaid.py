"""
Simple utility to convert JSON graph data to Mermaid format.
Uses CausalGraph for consistent graph handling.

Usage:
    # Process a single JSON file
    python batch_process_mermaid.py path/to/graph.json
    
    # Process all JSON files in a directory
    python batch_process_mermaid.py path/to/directory
"""

import json
import os
from pathlib import Path
from causal_graph_schema import CausalGraph

def load_json_file(file_path):
    """Load JSON file with UTF-8 encoding"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(directory):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def convert_json_to_mermaid(json_file_path, output_dir=None):
    """Convert a single JSON file to Mermaid format using CausalGraph"""
    # Determine output location
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    ensure_dir(output_dir)
    file_name = os.path.basename(json_file_path)
    base_name = os.path.splitext(file_name)[0]
    
    print(f"Processing: {file_name}")
    
    # Load JSON data
    data = load_json_file(json_file_path)
    
    # Create CausalGraph instance
    graph = CausalGraph()
    
    # Add nodes to graph
    for node_id, node_data in data.get('nodes', {}).items():
        graph.add_node(
            label=node_data['label'],
            confidence=node_data.get('confidence', 1.0),
            node_id=node_id
        )
    
    # Add edges to graph
    for edge_id, edge_data in data.get('edges', {}).items():
        try:
            graph.add_edge(
                source=edge_data['source'],
                target=edge_data['target'],
                confidence=edge_data.get('aggregate_confidence', 1.0),
                modifier=edge_data.get('modifier', 0),
                edge_id=edge_id
            )
        except ValueError:
            continue
    
    # Export to Mermaid format
    mermaid_file = os.path.join(output_dir, f"{base_name}.mmd")
    graph.export_mermaid(mermaid_file)
    print(f"Created Mermaid file: {mermaid_file}")
    
    return mermaid_file

def process_directory(directory, output_dir=None):
    """Process all JSON files in a directory"""
    if output_dir is None:
        output_dir = directory
    
    ensure_dir(output_dir)
    
    # Find all JSON files
    json_files = list(Path(directory).glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    for json_file in json_files:
        convert_json_to_mermaid(str(json_file), output_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Please provide a JSON file or directory path")
        print(f"Usage: python {sys.argv[0]} path/to/file.json")
        print(f"   or: python {sys.argv[0]} path/to/directory")
        sys.exit(1)
    
    path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(path):
        convert_json_to_mermaid(path, output)
    elif os.path.isdir(path):
        process_directory(path, output)
    else:
        print(f"Error: Path '{path}' does not exist") 