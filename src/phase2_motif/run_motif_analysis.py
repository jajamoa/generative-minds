#!/usr/bin/env python3
"""
Run Motif Library Analysis - Example script demonstrating the integration with existing code

This script shows how to use the new MotifLibrary implementation with the existing codebase
and demonstrates the complete workflow from graph loading to motif extraction,
semantic filtering, data augmentation, and visualization.
"""

import os
import sys
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add the necessary paths to import from our project
sys.path.append("src/phase2_motif")
from motif_library import MotifLibrary, load_graph_from_json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run motif library analysis")
    
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
    
    parser.add_argument(
        "--min-similarity", 
        type=float, 
        default=0.4,
        help="Minimum semantic similarity threshold (0-1)"
    )
    
    parser.add_argument(
        "--augment", 
        action="store_true",
        help="Perform data augmentation"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize motif groups"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the example"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running motif library analysis on samples from {args.samples_dir}")
    print(f"Results will be saved to {args.output_dir}")
    print(f"Minimum semantic similarity: {args.min_similarity}")
    
    # Step 1: Initialize the motif library
    print("\nStep 1: Initializing motif library...")
    library = MotifLibrary(min_semantic_similarity=args.min_similarity)
    
    # Step 2: Load graph samples
    print("\nStep 2: Loading graph samples...")
    graphs = {}
    for filename in os.listdir(args.samples_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.samples_dir, filename)
            sample_id = filename.replace('.json', '')
            try:
                graphs[sample_id] = load_graph_from_json(file_path)
                print(f"  Loaded graph {sample_id}: {len(graphs[sample_id].nodes())} nodes, {len(graphs[sample_id].edges())} edges")
            except Exception as e:
                print(f"  Error loading graph {filename}: {e}")
    
    # Step 3: Extract topological motifs
    print("\nStep 3: Extracting topological motifs...")
    for sample_id, G in graphs.items():
        print(f"  Processing graph {sample_id}...")
        library.extract_topological_motifs(G)
    
    print(f"  Total topological motifs extracted: {library.stats['total_topological_motifs']}")
    
    # Step 4: Apply semantic filtering
    print("\nStep 4: Applying semantic filtering...")
    semantic_groups = library.apply_semantic_filtering()
    print(f"  Total semantic groups: {library.stats['total_semantic_groups']}")
    
    # Step 5: Optional data augmentation
    if args.augment:
        print("\nStep 5: Performing data augmentation...")
        
        # Nearest neighbor augmentation
        print("  Applying nearest neighbor augmentation...")
        nn_groups = library.augment_by_nearest_neighbor(num_samples=5)
        print(f"    Generated {sum(len(motifs) for motifs in nn_groups.values())} new motifs using nearest neighbor")
        
        # Bootstrapping augmentation
        print("  Applying bootstrapping augmentation...")
        bootstrap_groups = library.augment_by_bootstrapping(num_samples=5)
        print(f"    Generated {sum(len(motifs) for motifs in bootstrap_groups.values())} new motifs using bootstrapping")
        
        print(f"  Total motifs after augmentation: {sum(len(motifs) for motifs in library.semantic_motifs.values())}")
    
    # Step 6: Save results
    print("\nStep 6: Saving results...")
    
    # Save library to JSON
    library_file = os.path.join(args.output_dir, "motif_library.json")
    library.save_library(library_file)
    print(f"  Motif library saved to {library_file}")
    
    # Export summary to JSON
    summary_file = os.path.join(args.output_dir, "motif_summary.json")
    library.export_to_json(summary_file)
    print(f"  Motif summary exported to {summary_file}")
    
    # Step 7: Optional visualization
    if args.visualize:
        print("\nStep 7: Visualizing motif groups...")
        vis_dir = os.path.join(args.output_dir, "motif_groups")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize all groups
        vis_files = library.visualize_all_groups(output_dir=vis_dir, max_examples=2)
        print(f"  Created {len(vis_files)} visualizations in {vis_dir}")
    
    # Step 8: Generate summary statistics
    print("\nStep 8: Generating summary statistics...")
    summary = library.get_motif_summary()
    
    print("\nMotif Library Summary:")
    print(f"  Total topological motifs: {summary['stats']['total_topological_motifs']}")
    print(f"  Total semantic groups: {summary['semantic_groups']}")
    print(f"  Total augmented groups: {summary['augmented_groups']}")
    print(f"  Total motifs: {summary['total_motifs']}")
    
    # Print motif type counts
    print("\nCount by motif type:")
    for motif_type, info in summary['motif_types'].items():
        if info['count'] > 0:
            print(f"  {motif_type} ({info['description']}): {info['count']} instances")
    
    print("\nMotif library analysis complete!")
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()