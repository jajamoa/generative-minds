#!/usr/bin/env python3
"""
Visualize motif analysis results based on NetworkX isomorphism detection
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def load_results(file_path='src/phase2_motif/output/motif_analysis_results.json'):
    """Load analysis results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def plot_motif_heatmap(results):
    """Generate motif frequency heatmap visualization"""
    # Prepare data
    samples = [r['sample_id'] for r in results]
    motifs = list(results[0]['motif_ratios'].keys())
    
    # Build frequency matrix
    frequency_matrix = np.zeros((len(samples), len(motifs)))
    for i, result in enumerate(results):
        for j, motif in enumerate(motifs):
            frequency_matrix[i, j] = result['motif_ratios'][motif]
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(frequency_matrix, annot=True, fmt='.2f', 
                xticklabels=motifs, yticklabels=samples, 
                cmap='YlGnBu', cbar_kws={'label': 'Motif Frequency'})
    plt.title('Motif Frequency Across Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_heatmap.png')
    plt.close()

def plot_motif_groups(results):
    """Visualize motif groups (chain, fork variations, collider variations)"""
    # Define motif groups
    motif_groups = {
        'Chain': ['M1'],
        'Fork Variations': ['M2.1', 'M2.2', 'M2.3'],
        'Collider Variations': ['M3.1', 'M3.2', 'M3.3']
    }
    
    # Prepare data
    samples = [r['sample_id'] for r in results]
    
    # Create subplots for each motif group
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each group
    for i, (group_name, motifs) in enumerate(motif_groups.items()):
        # Calculate average for each sample
        group_data = []
        for result in results:
            # Get counts for this group
            counts = [result['motif_counts'][m] for m in motifs]
            group_data.append(sum(counts))
        
        # Plot bar chart
        axes[i].bar(samples, group_data)
        axes[i].set_title(f'{group_name} Patterns')
        axes[i].set_ylabel('Total Count')
        axes[i].set_xticklabels(samples, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_groups.png')
    plt.close()

def plot_counts_by_demographic(results):
    """Visualize motif counts grouped by demographic category"""
    # Group by demographic
    demographic_groups = {}
    for result in results:
        demo = result['demographic_label']
        if demo not in demographic_groups:
            demographic_groups[demo] = []
        demographic_groups[demo].append(result)
    
    # Prepare plot
    motifs = list(results[0]['motif_ratios'].keys())
    
    # Simplify by grouping motifs
    motif_groups = {
        'Chain (M1)': ['M1'],
        'Basic Fork (M2.1)': ['M2.1'],
        'Extended Fork (M2.2+)': ['M2.2', 'M2.3'],
        'Basic Collider (M3.1)': ['M3.1'],
        'Extended Collider (M3.2+)': ['M3.2', 'M3.3']
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up positions
    n_groups = len(demographic_groups)
    n_motif_groups = len(motif_groups)
    width = 0.8 / n_motif_groups
    
    # Plot each motif group
    for i, (group_name, group_motifs) in enumerate(motif_groups.items()):
        # Calculate average count per demographic group
        counts = []
        labels = []
        for demo, group in demographic_groups.items():
            # Sum up counts for all motifs in this group
            avg_count = 0
            for r in group:
                for motif in group_motifs:
                    avg_count += r['motif_counts'].get(motif, 0)
            avg_count /= len(group)  # Average per sample in demographic
            
            counts.append(avg_count)
            labels.append(demo)
        
        # Plot bar group
        x = np.arange(len(labels))
        ax.bar(x + i * width - 0.4 + width/2, counts, width, label=group_name)
    
    # Customize plot
    ax.set_ylabel('Average Motif Count')
    ax.set_title('Average Motif Counts by Demographic Group')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/motif_by_demographic.png')
    plt.close()

def kmeans_clustering(results):
    """Perform K-means clustering on samples based on motif patterns"""
    # Prepare data
    X = []
    sample_ids = []
    demo_labels = []
    stances = []
    
    for result in results:
        # Extract features from motif ratios
        features = list(result['motif_ratios'].values())
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

def plot_motif_complexity(results):
    """Visualize relationship between motif complexity and stance"""
    # Calculate complexity scores
    complexity_data = []
    
    for result in results:
        # Complexity weighted by motif types
        # Higher weights for more complex patterns (extended and large forks/colliders)
        complexity = (
            1 * result['motif_counts'].get('M1', 0) +
            1 * result['motif_counts'].get('M2.1', 0) +
            2 * result['motif_counts'].get('M2.2', 0) +
            3 * result['motif_counts'].get('M2.3', 0) +
            1 * result['motif_counts'].get('M3.1', 0) +
            2 * result['motif_counts'].get('M3.2', 0) +
            3 * result['motif_counts'].get('M3.3', 0)
        )
        
        complexity_data.append({
            'sample_id': result['sample_id'],
            'demographic': result['demographic_label'],
            'stance': result['stance'],
            'complexity': complexity
        })
    
    # Sort by complexity
    complexity_data.sort(key=lambda x: x['complexity'])
    
    # Prepare plot data
    sample_ids = [d['sample_id'] for d in complexity_data]
    complexity_scores = [d['complexity'] for d in complexity_data]
    colors = ['green' if d['stance'] == 'support' else 
              'red' if d['stance'] == 'oppose' else 'blue' 
              for d in complexity_data]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sample_ids, complexity_scores, color=colors)
    
    # Add labels
    plt.title('Reasoning Complexity by Sample')
    plt.xlabel('Sample ID')
    plt.ylabel('Complexity Score')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Support'),
        Patch(facecolor='red', label='Oppose'),
        Patch(facecolor='blue', label='Neutral')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('src/phase2_motif/output/reasoning_complexity.png')
    plt.close()

def main():
    # Ensure output directory exists
    os.makedirs('src/phase2_motif/output', exist_ok=True)
    
    try:
        results = load_results()
    except FileNotFoundError:
        print("Run motif_analysis.py first to generate results")
        return
    
    # Generate visualizations
    plot_motif_heatmap(results)
    plot_motif_groups(results)
    plot_counts_by_demographic(results)
    plot_motif_complexity(results)
    kmeans_clustering(results)
    
    print("Visualizations created:")
    print("- motif_heatmap.png: Motif frequency heatmap")
    print("- motif_groups.png: Chain, fork, and collider pattern groups")
    print("- motif_by_demographic.png: Motif patterns by demographic group")
    print("- reasoning_complexity.png: Reasoning complexity analysis")
    print("- kmeans_clusters.png: K-means clustering results")
    print("- cluster_visualization.png: 2D visualization of clusters")
    print("- kmeans_results.json: Detailed clustering information")

if __name__ == "__main__":
    main() 