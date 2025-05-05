#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Motif Analysis - Integration of MotifLibrary with original implementation

This module extends the original motif_analysis.py with the new MotifLibrary
implementation while maintaining backward compatibility.
"""

import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import the MotifLibrary class
try:
    from motif_library import MotifLibrary, load_graph_from_json
    MOTIF_LIBRARY_AVAILABLE = True
except ImportError:
    print("Warning: MotifLibrary not available, using original implementation only.")
    MOTIF_LIBRARY_AVAILABLE = False

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
        incoming_edge_id = nodes[stance_node_id].get("incoming_edges", [])
        incoming_edge_id = incoming_edge_id[0] if incoming_edge_id else None
        
        if incoming_edge_id and incoming_edge_id in edges:
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

def analyze_samples(samples_dir, use_library=False):
    """
    Analyze all sample data files in directory
    
    Args:
        samples_dir: Directory with sample files
        use_library: Whether to use MotifLibrary (if available)
        
    Returns:
        List of results with feature vectors
    """
    results = []
    
    # Check if we should use MotifLibrary
    if use_library and MOTIF_LIBRARY_AVAILABLE:
        print("Using MotifLibrary for analysis...")
        library = MotifLibrary()
        
        # Process all JSON sample files
        for filename in os.listdir(samples_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(samples_dir, filename)
                try:
                    # Load graph
                    G = load_graph_from_json(file_path)
                    
                    # Extract demographic info and stance
                    metadata = G.graph.get("metadata", {})
                    demographic_label = metadata.get("perspective", "unknown")
                    stance = metadata.get("stance", "neutral")
                    
                    # Calculate motif vector
                    motif_vector = library.calculate_motif_vector(G)
                    
                    # Convert to original format
                    feature_vector = {
                        "demographic_label": demographic_label,
                        "stance": stance,
                        "motif_ratios": motif_vector,
                        "motif_counts": {k: int(v * 100) for k, v in motif_vector.items()}
                    }
                    
                    # Add sample ID
                    sample_id = filename.replace('.json', '')
                    feature_vector['sample_id'] = sample_id
                    
                    results.append(feature_vector)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Update library stats
        library_summary = library.get_motif_summary()
        print(f"MotifLibrary extracted {library_summary['total_motifs']} motifs in {library_summary['semantic_groups']} semantic groups.")
        
    else:
        # Use original implementation
        print("Using original implementation for analysis...")
        
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
        ratios = [result['motif_ratios'].get(motif, 0) for motif in motifs]
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
    
    # Collect all motifs from all samples
    all_motifs = set()
    for result in results:
        all_motifs.update(result['motif_counts'].keys())
    
    motifs = sorted(list(all_motifs))
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot counts for each sample
    bar_width = 0.9 / len(samples)
    for i, result in enumerate(results):
        counts = [result['motif_counts'].get(motif, 0) for motif in motifs]
        x = np.arange(len(motifs))
        plt.bar(x + i * bar_width, counts, bar_width, label=result['sample_id'])
    
    plt.xlabel('Motif Types')
    plt.ylabel('Count')
    plt.title('Motif Counts Across Samples')
    plt.xticks(np.arange(len(motifs)) + bar_width * (len(samples) - 1) / 2, 
              [m[:10] + '...' for m in motifs], rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_counts.png')
    plt.close()

def plot_motif_heatmap(results):
    """Generate motif frequency heatmap visualization"""
    # Prepare data
    samples = [r['sample_id'] for r in results]
    
    # Collect all motifs from all samples
    all_motifs = set()
    for result in results:
        all_motifs.update(result['motif_ratios'].keys())
    
    motifs = sorted(list(all_motifs))
    
    # Build frequency matrix
    frequency_matrix = np.zeros((len(samples), len(motifs)))
    for i, result in enumerate(results):
        for j, motif in enumerate(motifs):
            frequency_matrix[i, j] = result['motif_ratios'].get(motif, 0)
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(frequency_matrix, annot=True, fmt='.2f', 
                xticklabels=[m[:10] + '...' for m in motifs], 
                yticklabels=samples, 
                cmap='YlGnBu', cbar_kws={'label': 'Motif Frequency'})
    plt.title('Motif Frequency Across Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_heatmap.png')
    plt.close()

def cluster_by_stance(results):
    """Group samples by stance on upzoning"""
    clusters = defaultdict(list)
    
    for result in results:
        stance = result['stance']
        clusters[stance].append(result)
    
    return clusters

def kmeans_clustering(results):
    """Perform K-means clustering on samples based on motif patterns"""
    # Prepare data
    X = []
    sample_ids = []
    demo_labels = []
    stances = []
    
    for result in results:
        # Collect all motifs from all samples
        all_motifs = set()
        for r in results:
            all_motifs.update(r['motif_ratios'].keys())
        
        motifs = sorted(list(all_motifs))
        
        # Extract features from motif ratios
        features = [result['motif_ratios'].get(motif, 0) for motif in motifs]
        X.append(features)
        sample_ids.append(result['sample_id'])
        demo_labels.append(result['demographic_label'])
        stances.append(result['stance'])
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    # Apply K-means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Create cluster information
    cluster_data = []
    for i in range(len(sample_ids)):
        cluster_data.append({
            'sample_id': sample_ids[i],
            'demographic': demo_labels[i],
            'stance': stances[i],
            'cluster': f'Cluster {clusters[i]+1}'
        })
    
    # Save cluster results
    with open('src/phase2_motif/output/kmeans_results.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    
    # Count samples per cluster
    cluster_counts = {}
    for c in range(3):
        cluster_counts[f'Cluster {c+1}'] = list(clusters).count(c)
    
    # Plot cluster counts
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_counts.keys(), cluster_counts.values())
    plt.title('Samples per Cluster')
    plt.ylabel('Count')
    plt.savefig('src/phase2_motif/output/kmeans_clusters.png')
    plt.close()
    
    # Create a visualization of clusters with stance information
    plt.figure(figsize=(10, 8))
    colors = {'support': 'green', 'oppose': 'red', 'neutral': 'blue'}
    markers = {0: 'o', 1: 's', 2: '^'}
    
    for i in range(len(sample_ids)):
        plt.scatter(X[i, 0], X[i, 1], 
                   c=colors.get(stances[i], 'gray'),
                   marker=markers.get(clusters[i], 'x'),
                   s=100, alpha=0.7,
                   label=f"Cluster {clusters[i]+1}" if i == 0 else "")
        plt.text(X[i, 0] + 0.05, X[i, 1] + 0.05, sample_ids[i], fontsize=9)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    stance_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=s, markersize=10) 
                    for s, c in colors.items()]
    cluster_legend = [Line2D([0], [0], marker=m, color='gray', label=f'Cluster {i+1}', markersize=10) 
                     for i, m in markers.items()]
    
    plt.legend(handles=stance_legend + cluster_legend, loc='best')
    plt.title('Sample Clustering by Motif Patterns')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/cluster_visualization.png')
    plt.close()
    
    print("Clustering Results:")
    for cluster_id in range(3):
        print(f"Cluster {cluster_id+1}: {list(clusters).count(cluster_id)} samples")
        
    return cluster_data

def extended_analysis(samples_dir, output_dir='src/phase2_motif/output', use_library=True):
    """
    Run extended analysis using the new MotifLibrary if available
    
    Args:
        samples_dir: Directory with sample files
        output_dir: Output directory
        use_library: Whether to use MotifLibrary (if available)
        
    Returns:
        Results of the analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we can use MotifLibrary
    if use_library and MOTIF_LIBRARY_AVAILABLE:
        # Use MotifLibrary implementation for more advanced features
        print("Running extended analysis with MotifLibrary...")
        
        # Initialize MotifLibrary
        library = MotifLibrary()
        
        # Process sample graphs
        graphs = {}
        for filename in os.listdir(samples_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(samples_dir, filename)
                sample_id = filename.replace('.json', '')
                try:
                    graphs[sample_id] = load_graph_from_json(file_path)
                    print(f"Loaded graph {sample_id}: {len(graphs[sample_id].nodes())} nodes, {len(graphs[sample_id].edges())} edges")
                except Exception as e:
                    print(f"Error loading graph {filename}: {e}")
        
        # Extract topological motifs and apply semantic filtering
        for sample_id, G in graphs.items():
            print(f"Processing graph {sample_id}...")
            library.extract_topological_motifs(G)
        
        semantic_groups = library.apply_semantic_filtering()
        
        # Create semantic motifs directory
        semantic_dir = os.path.join(output_dir, "semantic_motifs")
        os.makedirs(semantic_dir, exist_ok=True)
        
        # Save library
        library.save_library(os.path.join(semantic_dir, "motif_library.json"))
        
        # Visualize motifs
        library.visualize_all_groups(output_dir=semantic_dir)
        
        # Create motif vectors for each sample
        results = []
        for sample_id, G in graphs.items():
            # Extract demographic info and stance
            demographic_label = "unknown"
            stance = "neutral"
            
            # Try to extract from metadata
            metadata = G.graph.get("metadata", {})
            demographic_label = metadata.get("perspective", demographic_label)
            stance = metadata.get("stance", stance)
            
            # Calculate motif vector
            motif_vector = library.calculate_motif_vector(G)
            
            # Create feature vector
            feature_vector = {
                "sample_id": sample_id,
                "demographic_label": demographic_label,
                "stance": stance,
                "motif_ratios": motif_vector,
                "motif_counts": {k: int(v * 100) for k, v in motif_vector.items()}
            }
            
            results.append(feature_vector)
        
        # Run original visualizations
        visualize_motif_distribution(results)
        visualize_motif_counts(results)
        plot_motif_heatmap(results)
        
        # Run clustering analysis
        kmeans_clustering(results)
        
        # Return results
        return {
            "results": results,
            "library": library,
            "semantic_groups": semantic_groups,
            "summary": library.get_motif_summary()
        }
    
    else:
        # Use original implementation
        print("Running standard analysis with original implementation...")
        results = analyze_samples(samples_dir, use_library=False)
        
        # Run standard visualizations
        visualize_motif_distribution(results)
        visualize_motif_counts(results)
        plot_motif_heatmap(results)
        
        # Run clustering analysis
        kmeans_clustering(results)
        
        # Return results
        return {"results": results}

def perform_data_augmentation(library, num_samples=5, output_dir='src/phase2_motif/output/augmented'):
    """
    Perform data augmentation on the motif library
    
    Args:
        library: MotifLibrary instance
        num_samples: Number of samples to generate per group
        output_dir: Output directory
        
    Returns:
        Dictionary with augmentation results
    """
    if not MOTIF_LIBRARY_AVAILABLE:
        print("Error: MotifLibrary not available for data augmentation.")
        return {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing data augmentation...")
    
    # Nearest neighbor augmentation
    print("  Applying nearest neighbor augmentation...")
    nn_groups = library.augment_by_nearest_neighbor(num_samples=num_samples)
    print(f"    Generated {sum(len(motifs) for motifs in nn_groups.values())} new motifs using nearest neighbor")
    
    # Bootstrapping augmentation
    print("  Applying bootstrapping augmentation...")
    bootstrap_groups = library.augment_by_bootstrapping(num_samples=num_samples)
    print(f"    Generated {sum(len(motifs) for motifs in bootstrap_groups.values())} new motifs using bootstrapping")
    
    # Save augmented library
    library.save_library(os.path.join(output_dir, "augmented_library.json"))
    
    # Visualize augmented motifs
    print("  Visualizing augmented motifs...")
    augmented_dir = os.path.join(output_dir, "motif_groups")
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Visualize nearest neighbor groups
    nn_files = []
    for group_key, motifs in nn_groups.items():
        file_path = library.visualize_motif_group(group_key, output_dir=augmented_dir)
        nn_files.append(file_path)
    
    # Visualize bootstrapping groups
    bootstrap_files = []
    for group_key, motifs in bootstrap_groups.items():
        file_path = library.visualize_motif_group(group_key, output_dir=augmented_dir)
        bootstrap_files.append(file_path)
    
    # Generate summary
    summary = {
        "nearest_neighbor": {
            "groups": len(nn_groups),
            "motifs": sum(len(motifs) for motifs in nn_groups.values()),
            "visualizations": len(nn_files)
        },
        "bootstrapping": {
            "groups": len(bootstrap_groups),
            "motifs": sum(len(motifs) for motifs in bootstrap_groups.values()),
            "visualizations": len(bootstrap_files)
        },
        "total_augmented_motifs": library.stats["total_augmented_motifs"]
    }
    
    # Save summary
    with open(os.path.join(output_dir, "augmentation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Data augmentation complete:")
    print(f"  Nearest neighbor: {summary['nearest_neighbor']['groups']} groups, {summary['nearest_neighbor']['motifs']} motifs")
    print(f"  Bootstrapping: {summary['bootstrapping']['groups']} groups, {summary['bootstrapping']['motifs']} motifs")
    print(f"  Total augmented motifs: {summary['total_augmented_motifs']}")
    print(f"  Results saved to {output_dir}")
    
    return summary

def main():
    # Sample data directory
    samples_dir = 'data/samples'
    
    # Create output directory if it doesn't exist
    os.makedirs('src/phase2_motif/output', exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extended motif analysis with library and data augmentation")
    
    parser.add_argument("--samples-dir", default=samples_dir, help="Directory with sample files")
    parser.add_argument("--output-dir", default="src/phase2_motif/output", help="Output directory")
    parser.add_argument("--use-library", action="store_true", help="Use MotifLibrary if available")
    parser.add_argument("--augment", action="store_true", help="Perform data augmentation")
    parser.add_argument("--augment-samples", type=int, default=5, help="Number of augmented samples per group")
    
    args = parser.parse_args()
    
    # Run analysis
    result = extended_analysis(args.samples_dir, args.output_dir, args.use_library)
    
    # Optionally perform data augmentation
    if args.augment and MOTIF_LIBRARY_AVAILABLE and args.use_library:
        augment_dir = os.path.join(args.output_dir, "augmented")
        library = result.get("library")
        if library:
            perform_data_augmentation(library, args.augment_samples, augment_dir)
    
    # Output results
    print(f"Total samples: {len(result['results'])}")
    
    # Group by stance
    stance_clusters = cluster_by_stance(result['results'])
    for stance, samples in stance_clusters.items():
        print(f"Stance '{stance}': {len(samples)} samples")
        for sample in samples:
            print(f"  - {sample['sample_id']} ({sample['demographic_label']})")
    
    # Save results
    with open('src/phase2_motif/output/motif_analysis_results.json', 'w', encoding='utf-8') as f:
        # Clean up result to make it JSON serializable
        clean_result = {
            "results": result['results'],
            "summary": result.get('summary', {})
        }
        json.dump(clean_result, f, ensure_ascii=False, indent=2)
    
    print("Extended motif analysis complete!")
    print("All results saved to src/phase2_motif/output/")

if __name__ == "__main__":
    main()