# Cognitive Motif Analysis

Simple implementation of Phase 2: Motif Library Clustering/Construction based on cognitive causal graph samples.

## Overview

This project analyzes basic motifs in cognitive causal graphs representing how different demographics make decisions about urban zoning policy.

## Features

- Identifies 3 basic motifs in causal graphs:
  - M1: Chain (A → B → C)
  - M2: Fork/One-to-many (A → B, A → C)
  - M3: Collider/Many-to-one (A → B ← C)

- Extracts demographic labels and stance on upzoning
- Performs clustering based on motif distributions

## Project Structure

```
data/samples/         # Sample data
motif_analysis.py     # Main analysis script
visualize_motifs.py   # Visualization script
README.md             # Documentation
```

## Usage

1. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn seaborn
```

2. Run analysis:

```bash
python motif_analysis.py
```

3. Run visualization:

```bash
python visualize_motifs.py
```

## Output Files

- `motif_analysis_results.json`: Analysis results
- `motif_heatmap.png`: Motif frequency heatmap
- `kmeans_clusters.png`: Clustering results
- `kmeans_results.json`: Cluster details

## Analysis Method

1. **Motif Definition**: Define basic motifs (chain, fork, collider) representing common graph patterns

2. **Feature Construction**:
   - For each user j's causal graph Gj, construct feature vector Vj
   - Vj contains:
     - dj: demographic label
     - sj: stance on upzoning
     - X(j)i: ratio of motif Mi in Gj

3. **Clustering**: Apply K-means clustering (k=3) based on motif frequency vectors
