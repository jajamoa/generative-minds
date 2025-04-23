import json
import networkx as nx
import time

def serialize_graph(nodes, edges, qa_history=None, include_qa=True):
    """
    Serialize graph data to dictionary format
    Args:
        nodes: Dictionary of nodes
        edges: Dictionary of edges
        qa_history: Optional QA history dictionary
        include_qa: Whether to include QA history in output
    Returns:
        Dictionary containing serialized graph data
    """
    result = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    if include_qa and qa_history is not None:
        result["qa_history"] = qa_history
        result["metadata"]["total_qa_pairs"] = len(qa_history)
    
    return result

def save_graph_to_json(nodes, edges, qa_history=None, base_filename="graph_export"):
    """
    Save causal graph data to JSON files
    Args:
        nodes: Dictionary of nodes
        edges: Dictionary of edges
        qa_history: Optional QA history dictionary
        base_filename: Base name for output files
    Returns:
        Dictionary containing paths to saved files
    """
    # Get serialized data
    data = serialize_graph(nodes, edges, qa_history)
    
    # Save to separate files for better version control
    file_paths = {}
    
    # Save nodes
    nodes_file = f"{base_filename}_nodes.json"
    with open(nodes_file, 'w') as f:
        json.dump(data["nodes"], f, indent=2)
    file_paths["nodes_file"] = nodes_file
    
    # Save edges
    edges_file = f"{base_filename}_edges.json"
    with open(edges_file, 'w') as f:
        json.dump(data["edges"], f, indent=2)
    file_paths["edges_file"] = edges_file
    
    # Save QA history if present
    if "qa_history" in data:
        qa_file = f"{base_filename}_qa_history.json"
        with open(qa_file, 'w') as f:
            json.dump(data["qa_history"], f, indent=2)
        file_paths["qa_file"] = qa_file
    
    # Save metadata
    metadata_file = f"{base_filename}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(data["metadata"], f, indent=2)
    file_paths["metadata_file"] = metadata_file
    
    return file_paths

def export_to_mermaid(G, output_file="graph_export.mmd"):
    """
    Export NetworkX graph to Mermaid flowchart
    """
    # Get node order (topological sort for directed graphs)
    try:
        node_order = list(nx.topological_sort(G)) if nx.is_directed(G) else list(G.nodes())
    except nx.NetworkXUnfeasible:
        node_order = list(G.nodes())
    
    # Build Mermaid diagram
    mermaid_lines = ["```mermaid", "flowchart TD"]
    
    # Add nodes
    for node_id in node_order:
        label = G.nodes[node_id].get("label", str(node_id))
        mermaid_lines.append(f"    {node_id}[{label}]")
    
    # Add edges with styles
    edge_count = 0
    link_styles = []
    
    for source, target, data in G.edges(data=True):
        if source in node_order and target in node_order:
            positive = data.get("positive", True)
            confidence = data.get("confidence", 0.5)
            
            edge_style = "-->" if positive else "--x"
            mermaid_lines.append(f"    {source} {edge_style} {target}")
            
            style = "stroke:#00AA00" if positive else "stroke:#FF0000,stroke-dasharray:3"
            thickness = max(1, int(confidence * 3))
            link_styles.append(f"    linkStyle {edge_count} {style},stroke-width:{thickness}px")
            edge_count += 1
    
    mermaid_lines.extend(link_styles)
    mermaid_lines.append("```")
    
    # Write to file
    mermaid_code = "\n".join(mermaid_lines)
    with open(output_file, 'w') as f:
        f.write(mermaid_code)
    
    return mermaid_code 