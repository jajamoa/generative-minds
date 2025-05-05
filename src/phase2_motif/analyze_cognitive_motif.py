#!/usr/bin/env python3
"""
Analyze Cognitive Motifs - Driver script for motif library extraction and analysis

This script integrates the motif_library.py module with the existing codebase to:
1. Extract motifs from cognitive causal graphs
2. Apply semantic filtering
3. Perform data augmentation
4. Generate visualizations and analysis
"""

import os
import sys
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Import the MotifLibrary class
from motif_library import MotifLibrary, process_sample_graphs

def setup_argparse():
    """Set up command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Analyze motifs in cognitive causal graphs using a two-step extraction process"
    )
    
    # Input/output options
    parser.add_argument(
        "--samples-dir", 
        default="data/samples",
        help="Directory containing sample graph JSON files"
    )
    parser.add_argument(
        "--output-dir", 
        default="src/phase2_motif/output/semantic_motifs",
        help="Directory to save output files"
    )
    
    # Motif extraction parameters
    parser.add_argument(
        "--min-size", 
        type=int, 
        default=3,
        help="Minimum motif size (nodes)"
    )
    parser.add_argument(
        "--max-size", 
        type=int, 
        default=5,
        help="Maximum motif size (nodes)"
    )
    parser.add_argument(
        "--similarity", 
        type=float, 
        default=0.4,
        help="Minimum semantic similarity threshold (0-1)"
    )
    
    # Data augmentation options
    parser.add_argument(
        "--augment", 
        action="store_true",
        help="Perform data augmentation"
    )
    parser.add_argument(
        "--augment-samples", 
        type=int, 
        default=10,
        help="Number of augmented samples to generate per motif group"
    )
    parser.add_argument(
        "--augment-method", 
        choices=["nn", "bootstrap", "both"], 
        default="both",
        help="Augmentation method (nn=nearest neighbor, bootstrap, both)"
    )
    
    # Visualization and analysis options
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize motif groups"
    )
    parser.add_argument(
        "--max-groups", 
        type=int, 
        default=None,
        help="Maximum number of groups to visualize (None=all)"
    )
    parser.add_argument(
        "--max-examples", 
        type=int, 
        default=3,
        help="Maximum number of example motifs per group in visualizations"
    )
    
    # Library management
    parser.add_argument(
        "--save-library", 
        action="store_true",
        help="Save the motif library to a file"
    )
    parser.add_argument(
        "--load-library", 
        help="Load a previously saved motif library file"
    )
    
    # Additional analysis
    parser.add_argument(
        "--analyze-demographic", 
        action="store_true",
        help="Analyze motifs by demographic group"
    )
    parser.add_argument(
        "--analyze-stance", 
        action="store_true",
        help="Analyze motifs by stance on upzoning"
    )
    
    return parser

def load_demographic_info(file_path):
    """Load demographic information from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading demographic info: {e}")
        return {}

def analyze_by_demographic(library, demo_file, output_dir):
    """Analyze motif distribution by demographic group"""
    # Load demographic information
    demo_info = load_demographic_info(demo_file)
    if not demo_info:
        print("No demographic information available for analysis.")
        return
        
    # Create motif distributions by demographic
    demo_distributions = defaultdict(lambda: defaultdict(float))
    
    # Group by demographic
    for sample_id, info in demo_info.items():
        demographic = info.get("demographic_label", "unknown")
        
        # For each sample, get motif distribution
        sample_graph = library.process_graph(sample_id)
        if sample_graph:
            motif_vector = library.calculate_motif_vector(sample_graph)
            
            # Add to demographic distribution
            for motif_group, freq in motif_vector.items():
                demo_distributions[demographic][motif_group] += freq
    
    # Normalize distributions
    for demo, distribution in demo_distributions.items():
        total = sum(distribution.values()) or 1
        for motif_group in distribution:
            demo_distributions[demo][motif_group] /= total
    
    # Create heatmap of distributions
    demographics = list(demo_distributions.keys())
    all_motifs = set()
    for dist in demo_distributions.values():
        all_motifs.update(dist.keys())
    all_motifs = sorted(all_motifs)
    
    # Create matrix for heatmap
    matrix = np.zeros((len(demographics), len(all_motifs)))
    for i, demo in enumerate(demographics):
        for j, motif in enumerate(all_motifs):
            matrix[i, j] = demo_distributions[demo].get(motif, 0)
    
    # Plot heatmap
    plt.figure(figsize=(15, 10))
    plt.imshow(matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Frequency')
    plt.xticks(range(len(all_motifs)), [m[:10] + '...' for m in all_motifs], rotation=90)
    plt.yticks(range(len(demographics)), demographics)
    plt.title('Motif Distribution by Demographic Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'motif_by_demographic.png'))
    plt.close()
    
    # Save distribution data
    with open(os.path.join(output_dir, 'demographic_distributions.json'), 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json.dump({k: dict(v) for k, v in demo_distributions.items()}, f, indent=2)
    
    return demo_distributions

def analyze_by_stance(library, stance_file, output_dir):
    """Analyze motif distribution by stance on upzoning"""
    # Load stance information
    stance_info = load_demographic_info(stance_file)
    if not stance_info:
        print("No stance information available for analysis.")
        return
        
    # Create motif distributions by stance
    stance_distributions = defaultdict(lambda: defaultdict(float))
    
    # Group by stance
    for sample_id, info in stance_info.items():
        stance = info.get("stance", "neutral")
        
        # For each sample, get motif distribution
        sample_graph = library.process_graph(sample_id)
        if sample_graph:
            motif_vector = library.calculate_motif_vector(sample_graph)
            
            # Add to stance distribution
            for motif_group, freq in motif_vector.items():
                stance_distributions[stance][motif_group] += freq
    
    # Normalize distributions
    for stance, distribution in stance_distributions.items():
        total = sum(distribution.values()) or 1
        for motif_group in distribution:
            stance_distributions[stance][motif_group] /= total
    
    # Create bar chart of distributions for top motifs
    stances = list(stance_distributions.keys())
    
    # Find top motifs (present in most stances)
    motif_counts = defaultdict(int)
    for dist in stance_distributions.values():
        for motif in dist:
            motif_counts[motif] += 1
    
    top_motifs = sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_motif_keys = [m[0] for m in top_motifs]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(top_motif_keys))
    width = 0.8 / len(stances)
    
    for i, stance in enumerate(stances):
        values = [stance_distributions[stance].get(motif, 0) for motif in top_motif_keys]
        ax.bar(x + i * width - 0.4 + width/2, values, width, label=stance)
    
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Top Motif Distribution by Stance on Upzoning')
    ax.set_xticks(x)
    ax.set_xticklabels([m[:10] + '...' for m in top_motif_keys], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'motif_by_stance.png'))
    plt.close()
    
    # Save distribution data
    with open(os.path.join(output_dir, 'stance_distributions.json'), 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json.dump({k: dict(v) for k, v in stance_distributions.items()}, f, indent=2)
    
    return stance_distributions

def create_motif_summary_sheet(library, output_dir):
    """Create a summary sheet of discovered motifs"""
    summary_data = []
    
    # Add metadata for each group
    for group_key, metadata in library.motif_metadata.items():
        if group_key in library.semantic_motifs:
            motifs = library.semantic_motifs[group_key]
            
            motif_type = metadata.get("motif_type", group_key.split('_')[0])
            description = metadata.get("description", library.motif_types.get(motif_type, "Unknown Type"))
            
            summary_data.append({
                "motif_id": group_key,
                "motif_type": motif_type,
                "description": description,
                "size": metadata.get("size", 0),
                "edges": metadata.get("edges", 0),
                "instances": len(motifs),
                "augmented": metadata.get("augmented", False),
                "augmentation_method": metadata.get("augmentation_method", ""),
                "parent_group": metadata.get("parent_group", "")
            })
    
    # Create a DataFrame
    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.sort_values(by=["motif_type", "instances"], ascending=[True, False])
        
        # Save to CSV
        csv_file = os.path.join(output_dir, "motif_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"Motif summary saved to {csv_file}")
        
        return df
    else:
        print("No motif data available for summary.")
        return None

def main():
    """Main function to run motif analysis"""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize or load library
    library = None
    
    if args.load_library:
        print(f"Loading motif library from {args.load_library}...")
        library = MotifLibrary.load_library(args.load_library)
        if not library:
            print("Error loading library. Creating a new one instead.")
            library = MotifLibrary(
                min_motif_size=args.min_size,
                max_motif_size=args.max_size,
                min_semantic_similarity=args.similarity
            )
    else:
        print("Creating new motif library...")
        library = MotifLibrary(
            min_motif_size=args.min_size,
            max_motif_size=args.max_size,
            min_semantic_similarity=args.similarity
        )
        
        # Process sample graphs
        print(f"Processing graphs from {args.samples_dir}...")
        process_sample_graphs(
            args.samples_dir, 
            args.output_dir, 
            min_semantic_similarity=args.similarity
        )
    
    # Apply data augmentation if requested
    if args.augment and library:
        if args.augment_method in ["nn", "both"]:
            print("Applying nearest neighbor augmentation...")
            library.augment_by_nearest_neighbor(num_samples=args.augment_samples)
        
        if args.augment_method in ["bootstrap", "both"]:
            print("Applying bootstrapping augmentation...")
            library.augment_by_bootstrapping(num_samples=args.augment_samples)
    
    # Save library if requested
    if args.save_library and library:
        library_file = os.path.join(args.output_dir, "motif_library.json")
        print(f"Saving motif library to {library_file}...")
        library.save_library(library_file)
    
    # Create summary sheet
    if library:
        create_motif_summary_sheet(library, args.output_dir)
    
    # Visualize motif groups if requested
    if args.visualize and library:
        vis_dir = os.path.join(args.output_dir, "motif_groups")
        print(f"Visualizing motif groups in {vis_dir}...")
        library.visualize_all_groups(
            output_dir=vis_dir,
            max_groups=args.max_groups,
            max_examples=args.max_examples
        )
    
    # Analyze by demographic if requested
    if args.analyze_demographic and library:
        demo_file = os.path.join(args.samples_dir, "demographic_info.json")
        if os.path.exists(demo_file):
            print("Analyzing motifs by demographic...")
            analyze_by_demographic(library, demo_file, args.output_dir)
        else:
            print(f"Demographic information file not found: {demo_file}")
    
    # Analyze by stance if requested
    if args.analyze_stance and library:
        stance_file = os.path.join(args.samples_dir, "stance_info.json")
        if os.path.exists(stance_file):
            print("Analyzing motifs by stance...")
            analyze_by_stance(library, stance_file, args.output_dir)
        else:
            print(f"Stance information file not found: {stance_file}")
    
    # Print summary
    if library:
        summary = library.get_motif_summary()
        print("\nMotif Analysis Complete!")
        print(f"Total topological motifs: {summary['stats']['total_topological_motifs']}")
        print(f"Total semantic groups: {summary['semantic_groups']}")
        print(f"Total augmented groups: {summary['augmented_groups']}")
        print(f"Total motifs: {summary['total_motifs']}")
        
        print("\nResults saved to:")
        print(f"- Motif library: {os.path.join(args.output_dir, 'motif_library.json')}")
        print(f"- Motif summary: {os.path.join(args.output_dir, 'motif_summary.csv')}")
        if args.visualize:
            print(f"- Visualizations: {os.path.join(args.output_dir, 'motif_groups')}")

if __name__ == "__main__":
    main