#!/usr/bin/env python3
"""
Extract semantic motifs from causal graphs

This script combines topology-based motif detection with semantic filtering
to identify functionally equivalent reasoning patterns across different causal graphs.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
from semantic_similarity import SemanticSimilarityEngine, apply_semantic_grouping, analyze_motifs

# Minimum motif size to extract (number of nodes)
MIN_MOTIF_SIZE = 3
# Maximum motif size to extract
MAX_MOTIF_SIZE = 5
# Minimum semantic similarity to group motifs (0-1)
MIN_SEMANTIC_SIMILARITY = 0.4  # Lowered from 0.6 to allow more grouping

# Define basic motif types
MOTIF_TYPES = {
    "M1": "Chain",          # A → B → C
    "M2.1": "Basic Fork",   # A → B, A → C (1-to-2)
    "M2.2": "Extended Fork", # A → B, A → C, A → D (1-to-3)
    "M2.3": "Large Fork",   # A → B, A → C, A → D, A → E, ... (1-to-4+)
    "M3.1": "Basic Collider", # A → C, B → C (2-to-1)
    "M3.2": "Extended Collider", # A → D, B → D, C → D (3-to-1)
    "M3.3": "Large Collider"  # A → E, B → E, C → E, D → E, ... (4+-to-1)
}

def load_graph_from_json(file_path):
    """
    Load a causal graph from JSON file and convert to NetworkX
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        NetworkX DiGraph object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes with labels
    for node_id, node_data in data["nodes"].items():
        # Normalize the label by lowercasing and removing special characters
        raw_label = node_data["label"]
        normalized_label = raw_label.lower().replace('_', ' ').strip()
        G.add_node(node_id, label=normalized_label, original_label=raw_label)
    
    # Add edges with metadata
    for edge_id, edge_data in data["edges"].items():
        G.add_edge(
            edge_data["source"], 
            edge_data["target"],
            id=edge_id,
            modifier=edge_data.get("modifier", 0),
            confidence=edge_data.get("aggregate_confidence", 0)
        )
    
    # Add metadata
    G.graph["metadata"] = data.get("metadata", {})
    
    return G

def create_motif_template(motif_type, size=3):
    """
    Create a template graph for a specific motif type
    
    Args:
        motif_type: Type of motif (e.g., 'M1', 'M2.1')
        size: Size of the motif (nodes)
        
    Returns:
        NetworkX DiGraph representing the motif template
    """
    G = nx.DiGraph()
    
    if motif_type == "M1":  # Chain
        # Create a simple path of 'size' nodes
        for i in range(size-1):
            G.add_edge(f"n{i}", f"n{i+1}")
    
    elif motif_type == "M2.1":  # Basic Fork (1-to-2)
        G.add_edge("n0", "n1")
        G.add_edge("n0", "n2")
        # Add additional nodes in chain if size > 3
        for i in range(3, size):
            # Connect to the second branch
            G.add_edge(f"n2", f"n{i}")
    
    elif motif_type == "M2.2":  # Extended Fork (1-to-3)
        G.add_edge("n0", "n1")
        G.add_edge("n0", "n2")
        G.add_edge("n0", "n3")
        # Add additional nodes in chain if size > 4
        for i in range(4, size):
            # Connect to the third branch
            G.add_edge(f"n3", f"n{i}")
    
    elif motif_type == "M2.3":  # Large Fork (1-to-4+)
        # Create a star with a center node connecting to all others
        for i in range(1, size):
            G.add_edge("n0", f"n{i}")
    
    elif motif_type == "M3.1":  # Basic Collider (2-to-1)
        G.add_edge("n0", "n2")
        G.add_edge("n1", "n2")
        # Add additional nodes in chain if size > 3
        for i in range(3, size):
            # Connect from the first source
            G.add_edge(f"n{i-1}", f"n{i}")
    
    elif motif_type == "M3.2":  # Extended Collider (3-to-1)
        G.add_edge("n0", "n3")
        G.add_edge("n1", "n3")
        G.add_edge("n2", "n3")
        # Add additional nodes in chain if size > 4
        for i in range(4, size):
            # Connect from the sink
            G.add_edge(f"n3", f"n{i}")
    
    elif motif_type == "M3.3":  # Large Collider (4+-to-1)
        # Create a reversed star with all nodes connecting to one
        sink_node = f"n{size-1}"
        for i in range(size-1):
            G.add_edge(f"n{i}", sink_node)
    
    # Add dummy labels to nodes
    for node in G.nodes():
        G.nodes[node]['label'] = f"node_{node}"
    
    return G

def extract_motifs_with_nx(G, motif_types=None, min_size=3, max_size=5):
    """
    Extract motifs from a graph using NetworkX subgraph isomorphism
    
    Args:
        G: NetworkX DiGraph to analyze
        motif_types: List of motif types to search for (default: all)
        min_size: Minimum motif size
        max_size: Maximum motif size
        
    Returns:
        Dictionary of found motifs grouped by type and size
    """
    if motif_types is None:
        motif_types = list(MOTIF_TYPES.keys())
    
    motifs = defaultdict(list)
    
    # Step 1: Track central nodes of each motif type
    # We only track center nodes, not all nodes in motifs
    fork_centers = set()     # Track centers of fork patterns (M2.x)
    collider_centers = set() # Track centers of collider patterns (M3.x)
    
    # Process motif types in order of complexity (larger motifs first)
    ordered_motif_types = [
        # Larger fork patterns first
        "M2.3", "M2.2", "M2.1",
        # Larger collider patterns first
        "M3.3", "M3.2", "M3.1",
        # Chain patterns last
        "M1"
    ]
    
    # Filter out motif types not requested
    ordered_motif_types = [mt for mt in ordered_motif_types if mt in motif_types]
    
    # For each motif type and size (in descending order of size)
    for motif_type in ordered_motif_types:
        print(f"Searching for {motif_type} motifs...")
        
        # Process sizes from large to small
        # For M1 (chain), only process size 3
        if motif_type == "M1":
            sizes = [3]  # Only size 3 for chains
        else:
            sizes = range(max_size, min_size - 1, -1)
            
        for size in sizes:
            # Create template for this motif type and size
            template = create_motif_template(motif_type, size)
            
            # Use VF2 algorithm to find all subgraph matches
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(G, template)
            
            # Find all subgraph isomorphisms (limited to prevent excessive matches)
            max_matches = 20  # Limit to prevent too many matches
            match_count = 0
            
            # Collect subgraph instances
            subgraphs = []
            
            for mapping in matcher.subgraph_isomorphisms_iter():
                # Mapping is from G to template, we need the reverse
                # Create a reverse mapping from template to G
                reverse_mapping = {v: k for k, v in mapping.items()}
                
                # Extract the subgraph nodes
                nodes = [reverse_mapping[n] for n in template.nodes()]
                
                # Handle different motif types differently
                if motif_type.startswith("M2"):  # Fork patterns
                    # For fork patterns, the first node is the central node
                    central_node = nodes[0]
                    
                    # Skip if this center node is already part of a larger fork
                    if central_node in fork_centers:
                        continue
                    
                    # Extract the subgraph
                    subgraph = G.subgraph(nodes).copy()
                    subgraphs.append(subgraph)
                    
                    # Mark central node as processed for future fork patterns
                    fork_centers.add(central_node)
                
                elif motif_type.startswith("M3"):  # Collider patterns
                    # For collider patterns, find the node with highest in-degree
                    in_degrees = {n: G.in_degree(n) for n in nodes}
                    central_node = max(in_degrees, key=in_degrees.get)
                    
                    # Skip if this center node is already part of a larger collider
                    if central_node in collider_centers:
                        continue
                    
                    # Extract the subgraph
                    subgraph = G.subgraph(nodes).copy()
                    subgraphs.append(subgraph)
                    
                    # Mark central node as processed for future collider patterns
                    collider_centers.add(central_node)
                
                else:  # Chain patterns (M1) and others
                    # For chains, we only accept size 3
                    if len(nodes) == 3:
                        subgraph = G.subgraph(nodes).copy()
                        subgraphs.append(subgraph)
                
                match_count += 1
                if match_count >= max_matches:
                    print(f"  Reached limit of {max_matches} matches for {motif_type}_size_{size}")
                    break
            
            # Add the found subgraphs to the result dictionary
            if subgraphs:
                key = f"{motif_type}_size_{size}"
                motifs[key] = subgraphs
                print(f"  Found {len(subgraphs)} instances of {key}")
    
    # Remove empty entries
    motifs = {k: v for k, v in motifs.items() if v}
    
    return motifs

def visualize_motif_group(motifs, group_key, output_dir, max_examples=3):
    """
    Visualize a group of semantically similar motifs
    
    Args:
        motifs: List of motif subgraphs
        group_key: Key for the motif group
        output_dir: Directory to save visualization
        max_examples: Maximum number of example motifs to show
    """
    if not motifs:
        return
    
    # Limit the number of examples
    examples = motifs[:min(max_examples, len(motifs))]
    
    # Create a figure with subplots for each example
    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5))
    if len(examples) == 1:
        axes = [axes]
    
    # Get motif type from the key
    motif_type = group_key.split('_')[0] if '_' in group_key else "Unknown"
    motif_name = MOTIF_TYPES.get(motif_type, "Unknown Type")
    
    # Plot each example
    for i, subgraph in enumerate(examples):
        ax = axes[i]
        
        # Create node labels
        node_labels = {node: subgraph.nodes[node].get('label', str(node)) for node in subgraph.nodes()}
        
        # Create edge labels
        edge_labels = {(u, v): f"{subgraph.edges[u, v].get('modifier', 0):.1f}" 
                      for u, v in subgraph.edges()}
        
        # Simple coloring scheme based on node position in the graph
        in_degree = dict(subgraph.in_degree())
        out_degree = dict(subgraph.out_degree())
        
        node_colors = []
        for node in subgraph.nodes():
            # Source nodes (no incoming edges)
            if in_degree[node] == 0 and out_degree[node] > 0:
                node_colors.append('skyblue')
            # Sink nodes (no outgoing edges)
            elif in_degree[node] > 0 and out_degree[node] == 0:
                node_colors.append('salmon')
            # Intermediate nodes
            else:
                node_colors.append('lightgreen')
        
        # Draw the graph
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors, node_size=1500, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, width=2, edge_color='gray', arrowsize=20)
        nx.draw_networkx_labels(subgraph, pos, labels=node_labels, ax=ax, font_size=10)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax, font_size=9)
        
        # Add a title
        if i == 0:
            ax.set_title(f"{motif_name} ({motif_type}): Group {group_key}\nSample {i+1}/{len(motifs)} examples")
        else:
            ax.set_title(f"Sample {i+1}/{len(motifs)} examples")
        
        ax.axis('off')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/motif_group_{group_key}.png")
    plt.close()

def create_motif_summary(semantic_groups, output_file):
    """
    Create a summary of discovered motifs
    
    Args:
        semantic_groups: Dictionary of semantically grouped motifs
        output_file: File to save the summary
    """
    summary_data = []
    
    for group_key, motifs in semantic_groups.items():
        if not motifs:
            continue
            
        # Extract basic motif properties
        sample_motif = motifs[0]
        
        # Extract motif type from key
        parts = group_key.split('_')
        if len(parts) > 0 and parts[0] in MOTIF_TYPES:
            motif_type = parts[0]
            motif_name = MOTIF_TYPES[motif_type]
        else:
            motif_type = "Unknown"
            motif_name = "Unknown Type"
        
        summary_data.append({
            "motif_id": group_key,
            "motif_type": motif_type,
            "description": motif_name,
            "size": len(sample_motif.nodes()),
            "edges": len(sample_motif.edges()),
            "instances": len(motifs)
        })
    
    # Create a DataFrame
    df = pd.DataFrame(summary_data)
    df = df.sort_values(by=["motif_type", "instances"], ascending=[True, False])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    return df

def apply_custom_semantic_grouping(topological_motifs, min_similarity=0.4):
    """
    Apply a custom semantic grouping with additional preprocessing
    
    Args:
        topological_motifs: Dictionary of topologically grouped motifs
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary of semantic groups
    """
    similarity_engine = SemanticSimilarityEngine(use_wordnet=True)
    semantic_groups = {}
    
    # Group by motif type first
    motif_type_groups = defaultdict(list)
    for key, motifs in topological_motifs.items():
        # Extract motif type from the key (e.g., M1, M2.1)
        motif_type = key.split('_')[0]
        motif_type_groups[motif_type].extend([(key, motif) for motif in motifs])
    
    # For each motif type, perform semantic subgrouping within isomorphism classes
    for motif_type, motif_items in motif_type_groups.items():
        # Group by size class
        size_groups = defaultdict(list)
        for key, motif in motif_items:
            size_part = '_'.join(key.split('_')[1:])  # Everything after the motif type
            size_groups[size_part].append((key, motif))
        
        # For each size class, apply semantic grouping
        for size_class, items in size_groups.items():
            if len(items) <= 1:
                # If only one item, no need to group
                orig_key = items[0][0]
                semantic_groups[orig_key] = [items[0][1]]
                continue
            
            # Extract just the motifs for semantic comparison
            motifs = [item[1] for item in items]
            orig_keys = [item[0] for item in items]
            
            # Apply semantic grouping
            processed = set()
            current_group_id = 1
            
            for i, motif1 in enumerate(motifs):
                if i in processed:
                    continue
                    
                # Create a new semantic group
                group_key = f"{orig_keys[i]}_{current_group_id}"
                semantic_groups[group_key] = [motif1]
                processed.add(i)
                
                # Find semantically similar motifs
                for j, motif2 in enumerate(motifs):
                    if j in processed or i == j:
                        continue
                    
                    # Check for isomorphism
                    matcher = nx.algorithms.isomorphism.DiGraphMatcher(motif1, motif2)
                    if matcher.is_isomorphic():
                        mapping = next(matcher.isomorphisms_iter())
                        
                        # Check semantic similarity
                        similarity = compute_enhanced_similarity(similarity_engine, motif1, motif2, mapping)
                        if similarity >= min_similarity:
                            semantic_groups[group_key].append(motif2)
                            processed.add(j)
                
                current_group_id += 1
    
    return semantic_groups

def compute_enhanced_similarity(engine, motif1, motif2, node_mapping):
    """
    Compute enhanced semantic similarity between motifs with more preprocessing
    
    Args:
        engine: SemanticSimilarityEngine instance
        motif1: First motif
        motif2: Second motif
        node_mapping: Node mapping from motif1 to motif2
        
    Returns:
        Similarity score (0-1)
    """
    # Check that the mapping is valid
    if not node_mapping or len(node_mapping) == 0:
        return 0.0
    
    # Calculate node-by-node similarity with position weighting
    similarities = []
    position_weights = []
    
    # Get node positions
    in_degree1 = dict(motif1.in_degree())
    out_degree1 = dict(motif1.out_degree())
    in_degree2 = dict(motif2.in_degree())
    out_degree2 = dict(motif2.out_degree())
    
    for node1, node2 in node_mapping.items():
        label1 = motif1.nodes[node1].get('label', str(node1))
        label2 = motif2.nodes[node2].get('label', str(node2))
        
        # Calculate basic similarity
        sim = engine.node_similarity(label1, label2)
        
        # Determine node positions
        pos1 = get_node_position(in_degree1[node1], out_degree1[node1])
        pos2 = get_node_position(in_degree2[node2], out_degree2[node2])
        
        # Add position similarity bonus (0.1) if positions match
        if pos1 == pos2:
            sim = min(1.0, sim + 0.1)
        
        # Weight by position importance
        weight = get_position_weight(pos1)
        similarities.append(sim)
        position_weights.append(weight)
    
    if not similarities:
        return 0.0
    
    # Weighted average
    return sum(s * w for s, w in zip(similarities, position_weights)) / sum(position_weights)

def get_node_position(in_deg, out_deg):
    """Get node position type based on in/out degree"""
    if in_deg == 0 and out_deg > 0:
        return "source"
    elif in_deg > 0 and out_deg == 0:
        return "sink"
    else:
        return "intermediate"

def get_position_weight(position):
    """Get importance weight based on position"""
    if position == "source":
        return 1.5  # Sources are important
    elif position == "sink":
        return 1.5  # Sinks are important
    else:
        return 1.0  # Normal weight for intermediate nodes

def process_sample_graphs(samples_dir, output_dir):
    """
    Process all sample graphs to extract and analyze semantic motifs
    
    Args:
        samples_dir: Directory containing sample graph JSON files
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all graphs
    graphs = {}
    for filename in os.listdir(samples_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(samples_dir, filename)
            sample_id = filename.replace('.json', '')
            graphs[sample_id] = load_graph_from_json(file_path)
            print(f"Loaded graph {sample_id}: {len(graphs[sample_id].nodes())} nodes, {len(graphs[sample_id].edges())} edges")
    
    # Extract topological motifs from each graph
    all_topological_motifs = {}
    
    for sample_id, G in graphs.items():
        print(f"\nProcessing graph {sample_id}...")
        # Use NetworkX to find motifs
        topo_motifs = extract_motifs_with_nx(G, min_size=MIN_MOTIF_SIZE, max_size=MAX_MOTIF_SIZE)
        
        for key, motifs in topo_motifs.items():
            all_topological_motifs[f"{sample_id}_{key}"] = motifs
    
    # Apply semantic filtering to group similar motifs
    print("\nApplying semantic filtering...")
    semantic_groups = apply_custom_semantic_grouping(all_topological_motifs, min_similarity=MIN_SEMANTIC_SIMILARITY)
    
    # Create summary
    print("Creating summary...")
    summary_file = os.path.join(output_dir, "motif_summary.csv")
    summary = create_motif_summary(semantic_groups, summary_file)
    
    # Visualize representative motifs
    print("Visualizing motifs...")
    for group_key, motifs in semantic_groups.items():
        visualize_motif_group(motifs, group_key, output_dir)
    
    # Save semantic groups
    with open(os.path.join(output_dir, "semantic_motifs.json"), 'w') as f:
        # Convert NetworkX graphs to serializable format
        serializable_groups = {}
        for group_key, motifs in semantic_groups.items():
            serializable_groups[group_key] = [
                {
                    "nodes": list(m.nodes()),
                    "edges": list(m.edges()),
                    "node_labels": {str(n): m.nodes[n].get('label', str(n)) for n in m.nodes()}
                }
                for m in motifs
            ]
        
        json.dump({
            "semantic_groups": serializable_groups
        }, f, indent=2)
    
    # Summarize results by motif type
    motif_type_counts = defaultdict(int)
    motif_instance_counts = defaultdict(int)
    
    for group_key, motifs in semantic_groups.items():
        motif_type = group_key.split('_')[0]
        if motif_type in MOTIF_TYPES:
            motif_type_counts[motif_type] += 1
            motif_instance_counts[motif_type] += len(motifs)
    
    print(f"\nFound {len(semantic_groups)} semantic motif groups by type:")
    for motif_type, count in sorted(motif_type_counts.items()):
        motif_name = MOTIF_TYPES.get(motif_type, "Unknown")
        instances = motif_instance_counts[motif_type]
        print(f"  {motif_type} ({motif_name}): {count} groups with {instances} total instances")
    
    print(f"\nTop 10 groups by instance count:")
    for i, (group_key, motifs) in enumerate(sorted(semantic_groups.items(), key=lambda x: len(x[1]), reverse=True)):
        if i < 10:  # Show top 10
            motif_type = group_key.split('_')[0]
            motif_name = MOTIF_TYPES.get(motif_type, "Unknown")
            print(f"  {group_key} ({motif_name}): {len(motifs)} instances")
    
    print(f"\nResults saved to {output_dir}")
    return semantic_groups

def main():
    # Sample data directory
    samples_dir = 'data/samples'
    
    # Output directory
    output_dir = 'src/phase2_motif/output/semantic_motifs'
    
    # Process sample graphs
    semantic_groups = process_sample_graphs(samples_dir, output_dir)
    
    # Print summary
    print("\nMotif analysis complete!")
    print(f"Found {sum(len(motifs) for motifs in semantic_groups.values())} total motif instances")
    print(f"Grouped into {len(semantic_groups)} semantic motif patterns")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 