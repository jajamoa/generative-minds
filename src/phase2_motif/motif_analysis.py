#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Motif Library Clustering/Construction
Implementation using NetworkX isomorphism functions to identify motif patterns
"""

import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

# Define basic motifs as NetworkX graph templates
def create_motif_templates():
    """Create NetworkX graph templates for each motif pattern"""
    templates = {}
    
    # M1: Chain (A → B → C)
    chain = nx.DiGraph()
    chain.add_edges_from([('A', 'B'), ('B', 'C')])
    templates["M1"] = chain
    
    # M2: Fork variations (one-to-many)
    # M2.1: Basic fork (1-to-2)
    fork_basic = nx.DiGraph()
    fork_basic.add_edges_from([('A', 'B'), ('A', 'C')])
    templates["M2.1"] = fork_basic
    
    # M2.2: Extended fork (1-to-3)
    fork_extended = nx.DiGraph()
    fork_extended.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D')])
    templates["M2.2"] = fork_extended
    
    # M2.3: Large fork (1-to-many)
    fork_large = nx.DiGraph()
    fork_large.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E')])
    templates["M2.3"] = fork_large
    
    # M3: Collider variations (many-to-one)
    # M3.1: Basic collider (2-to-1)
    collider_basic = nx.DiGraph()
    collider_basic.add_edges_from([('A', 'C'), ('B', 'C')])
    templates["M3.1"] = collider_basic
    
    # M3.2: Extended collider (3-to-1)
    collider_extended = nx.DiGraph()
    collider_extended.add_edges_from([('A', 'D'), ('B', 'D'), ('C', 'D')])
    templates["M3.2"] = collider_extended
    
    # M3.3: Large collider (many-to-1)
    collider_large = nx.DiGraph()
    collider_large.add_edges_from([('A', 'E'), ('B', 'E'), ('C', 'E'), ('D', 'E')])
    templates["M3.3"] = collider_large
    
    return templates

# Define motif descriptions
MOTIFS = {
    "M1": "chain",               # Chain: A → B → C
    "M2.1": "basic_fork",        # Fork (1-to-2): A → B, A → C
    "M2.2": "extended_fork",     # Fork (1-to-3): A → B, A → C, A → D
    "M2.3": "large_fork",        # Fork (1-to-4+): A → B, A → C, A → D, A → E, ...
    "M3.1": "basic_collider",    # Collider (2-to-1): A → C, B → C
    "M3.2": "extended_collider", # Collider (3-to-1): A → D, B → D, C → D
    "M3.3": "large_collider",    # Collider (4+-to-1): A → E, B → E, C → E, D → E, ...
}

def load_graph(file_path):
    """Load cognitive causal graph from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_to_networkx(graph):
    """Convert JSON graph data to NetworkX DiGraph"""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in graph["nodes"].items():
        G.add_node(node_id, label=node_data["label"])
    
    # Add edges
    for edge_id, edge_data in graph["edges"].items():
        G.add_edge(edge_data["source"], edge_data["target"], 
                  modifier=edge_data.get("modifier", 0),
                  confidence=edge_data.get("aggregate_confidence", 0))
    
    return G

def identify_motifs_isomorphism(graph):
    """
    Identify motifs in graph using subgraph isomorphism
    
    This approach finds all instances of each motif pattern in the graph
    using NetworkX's subgraph isomorphism function.
    """
    # Convert JSON graph to NetworkX DiGraph
    G = convert_to_networkx(graph)
    
    # Create motif templates
    templates = create_motif_templates()
    
    # Initialize motif counts
    motif_counts = {motif_id: 0 for motif_id in MOTIFS.keys()}
    
    # Identify motifs using subgraph isomorphism
    for motif_id, template in templates.items():
        # Use VF2 algorithm to find all isomorphic subgraphs
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(G, template)
        
        # Count subgraph matches without enumerating all of them (more efficient)
        # We use a generator expression and sum to count matches
        matches = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
        
        # Handle special cases for fork and collider patterns
        # For these patterns, we need to adjust counts due to symmetric matches
        if motif_id.startswith("M2") or motif_id.startswith("M3"):
            # For forks and colliders, divide by factorial of identical nodes
            if motif_id == "M2.1" or motif_id == "M3.1":  # 2 identical nodes
                matches = matches // 2
            elif motif_id == "M2.2" or motif_id == "M3.2":  # 3 identical nodes
                matches = matches // 6  # 3! = 6
            elif motif_id == "M2.3" or motif_id == "M3.3":  # 4 identical nodes
                matches = matches // 24  # 4! = 24
            
        motif_counts[motif_id] = matches
    
    return motif_counts

def extract_demographic_info(graph):
    """Extract demographic label and stance on upzoning"""
    metadata = graph.get("metadata", {})
    
    # Extract demographic label
    demographic_label = metadata.get("perspective", "unknown")
    
    # Extract stance on upzoning
    stance = "neutral"  # Default
    
    # Find stance from final node
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # Find upzoning_stance node
    stance_node_id = None
    for node_id, node_data in nodes.items():
        if "upzoning_stance" in node_data["label"]:
            stance_node_id = node_id
            break
    
    if stance_node_id:
        # Get incoming edge
        incoming_edge_id = nodes[stance_node_id]["incoming_edges"][0] if nodes[stance_node_id]["incoming_edges"] else None
        
        if incoming_edge_id:
            # Use modifier value to determine stance
            modifier = edges[incoming_edge_id]["modifier"]
            if modifier > 0.3:
                stance = "support"
            elif modifier < -0.3:
                stance = "oppose"
            else:
                stance = "neutral"
    
    return demographic_label, stance

def construct_feature_vector(graph):
    """
    Construct feature-motif frequency vector
    
    For each graph, we identify motifs, normalize their counts,
    and combine with demographic information.
    """
    # Extract demographic info and stance
    demographic_label, stance = extract_demographic_info(graph)
    
    # Identify motifs using isomorphism
    motif_counts = identify_motifs_isomorphism(graph)
    
    # Calculate motif ratios (normalized)
    total_motifs = sum(motif_counts.values()) or 1  # Avoid division by zero
    motif_ratios = {motif_id: count / total_motifs for motif_id, count in motif_counts.items()}
    
    # Construct feature vector
    feature_vector = {
        "demographic_label": demographic_label,
        "stance": stance,
        "motif_counts": motif_counts,
        "motif_ratios": motif_ratios
    }
    
    return feature_vector

def analyze_samples(samples_dir):
    """Analyze all sample data files in directory"""
    results = []
    
    # Process all JSON sample files
    for filename in os.listdir(samples_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(samples_dir, filename)
            graph = load_graph(file_path)
            feature_vector = construct_feature_vector(graph)
            
            # Add sample ID
            sample_id = filename.replace('.json', '')
            feature_vector['sample_id'] = sample_id
            
            results.append(feature_vector)
    
    return results

def visualize_motif_distribution(results):
    """Visualize motif distribution across samples"""
    # Prepare data
    samples = [r['sample_id'] for r in results]
    motifs = list(MOTIFS.keys())
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot distribution for each sample
    bar_width = 0.9 / len(samples)
    for i, result in enumerate(results):
        ratios = [result['motif_ratios'][motif] for motif in motifs]
        x = np.arange(len(motifs))
        plt.bar(x + i * bar_width, ratios, bar_width, label=result['sample_id'])
    
    plt.xlabel('Motif Types')
    plt.ylabel('Frequency Ratio')
    plt.title('Motif Distribution Across Samples')
    plt.xticks(np.arange(len(motifs)) + bar_width * (len(samples) - 1) / 2, motifs, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_distribution.png')
    plt.close()

def visualize_motif_counts(results):
    """Visualize raw motif counts across samples"""
    # Prepare data
    samples = [r['sample_id'] for r in results]
    motifs = list(MOTIFS.keys())
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot counts for each sample
    bar_width = 0.9 / len(samples)
    for i, result in enumerate(results):
        counts = [result['motif_counts'][motif] for motif in motifs]
        x = np.arange(len(motifs))
        plt.bar(x + i * bar_width, counts, bar_width, label=result['sample_id'])
    
    plt.xlabel('Motif Types')
    plt.ylabel('Count')
    plt.title('Motif Counts Across Samples')
    plt.xticks(np.arange(len(motifs)) + bar_width * (len(samples) - 1) / 2, motifs, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_counts.png')
    plt.close()

def cluster_by_stance(results):
    """Group samples by stance on upzoning"""
    clusters = defaultdict(list)
    
    for result in results:
        stance = result['stance']
        clusters[stance].append(result)
    
    return clusters

def main():
    # Sample data directory
    samples_dir = 'data/samples'
    
    # Create output directory if it doesn't exist
    os.makedirs('src/phase2_motif/output', exist_ok=True)
    
    # Analyze all samples
    results = analyze_samples(samples_dir)
    
    # Visualize motif distribution
    visualize_motif_distribution(results)
    visualize_motif_counts(results)
    
    # Group by stance
    stance_clusters = cluster_by_stance(results)
    
    # Output results
    print(f"Total samples: {len(results)}")
    for stance, samples in stance_clusters.items():
        print(f"Stance '{stance}': {len(samples)} samples")
        for sample in samples:
            print(f"  - {sample['sample_id']} ({sample['demographic_label']})")
    
    # Save results
    with open('src/phase2_motif/output/motif_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 