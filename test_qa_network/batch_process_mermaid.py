import json
import os
from pathlib import Path

def ensure_output_dir(base_dir):
    """Ensure output directory exists"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_json_file(file_path):
    """Load JSON file with UTF-8 encoding"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_mermaid(data, output_file):
    """Convert JSON data to Mermaid format"""
    mermaid_lines = ['graph TD']
    
    # Add nodes
    for node_id, node_data in data['nodes'].items():
        label = node_data['label'].replace('_', ' ')
        confidence = node_data.get('confidence', 1.0)
        # Add confidence to label
        label = f"{label}\n(conf: {confidence:.2f})"
        mermaid_lines.append(f'    {node_id}["{label}"]')
    
    # Add edges
    for edge_id, edge_data in data['edges'].items():
        source = edge_data['source']
        target = edge_data['target']
        modifier = edge_data.get('modifier', 0)
        confidence = edge_data.get('aggregate_confidence', 1.0)
        
        # Choose arrow style and color based on modifier value
        if modifier > 0:
            arrow_style = '-->|+|'
            style = f'style {source}{target} stroke:green,stroke-width:2px'
        elif modifier < 0:
            arrow_style = '-->|-|'
            style = f'style {source}{target} stroke:red,stroke-width:2px'
        else:
            arrow_style = '-->'
            style = f'style {source}{target} stroke:grey,stroke-width:1px'
            
        # Add edge and style
        edge_line = f'    {source} {arrow_style} {target}'
        if confidence < 1.0:
            edge_line += f'|{confidence:.2f}|'
        mermaid_lines.append(edge_line)
        mermaid_lines.append(f'    {style}')
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mermaid_lines))

def process_json_files(input_dir):
    """Batch process JSON files to Mermaid format"""
    # Get all JSON files
    json_files = list(Path(input_dir).glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"Processing file: {json_file.name}")
        
        # Load JSON data
        data = load_json_file(json_file)
        
        # Create output filenames in the same directory
        output_file = json_file.with_suffix('.mmd')
        metadata_file = json_file.with_suffix('.metadata')
        
        # Convert to Mermaid format
        convert_to_mermaid(data, output_file)
        
        # Save metadata information
        with open(metadata_file, 'w', encoding='utf-8') as f:
            metadata = data.get('metadata', {})
            f.write("Graph Metadata Information:\n")
            f.write(f"Perspective: {metadata.get('perspective', 'unknown')}\n")
            f.write(f"Number of nodes: {metadata.get('num_nodes', 0)}\n")
            f.write(f"Number of edges: {metadata.get('num_edges', 0)}\n")
            f.write(f"Number of QA pairs: {metadata.get('num_qa_pairs', 0)}\n")
            f.write("\nFocus Areas:\n")
            for area in metadata.get('focus_areas', []):
                f.write(f"- {area}\n")
        
        print(f"Generated files:\n- {output_file}\n- {metadata_file}")

def main():
    # Set input directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "data", "samples")
    
    # Process files
    process_json_files(input_dir)
    print("Batch processing completed!")

if __name__ == "__main__":
    main() 