# Phase 2: Motif Analysis

Implementation of motif library clustering/construction based on cognitive causal graphs samples using NetworkX isomorphism detection and semantic similarity filtering.

## Overview

This module analyzes motif patterns in cognitive causal graphs representing how different demographics make decisions about urban zoning policy. First, we use graph isomorphism to identify structural patterns, then apply semantic similarity filtering to find functionally equivalent reasoning patterns across different causal graphs.

## Features

- Uses NetworkX's subgraph isomorphism functions to identify motif patterns
- Applies semantic similarity filtering to identify functionally equivalent reasoning patterns
- Identifies multiple variations of motifs in causal graphs:
  - M1: Chain (A → B → C) - Fixed at 3 nodes
  - M2: Fork variations (one-to-many)
    - M2.1: Basic fork (1-to-2)
    - M2.2: Extended fork (1-to-3)
    - M2.3: Large fork (1-to-4+)
  - M3: Collider variations (many-to-one)
    - M3.1: Basic collider (2-to-1)
    - M3.2: Extended collider (3-to-1)
    - M3.3: Large collider (4+-to-1)

- Extracts demographic labels and stance on upzoning
- Performs clustering based on motif distributions
- Analyzes reasoning complexity through motif patterns
- Visualizes semantic motif groups across different causal graphs

## Directory Structure

```
src/phase2_motif/
├── motif_analysis.py      # Basic motif analysis (topology-based)
├── semantic_similarity.py # Semantic similarity functions for node/motif comparison
├── extract_semantic_motifs.py # Extract and visualize semantic motifs
├── visualize_motifs.py    # Visualization of motif patterns
├── output/                # Generated files
└── README.md              # Documentation
```

## Two-Step Approach to Motif Identification

Our approach follows a two-step process:

1. **Topology-Based Candidate Grouping**: 
   - Identify subgraphs with similar topology using graph isomorphism
   - Group subgraphs that share the same structure (chain, fork, collider patterns)

2. **Semantic Filtering**:
   - Filter groups of isomorphic subgraphs by semantic similarity
   - Compare node labels using WordNet and/or word vectors
   - Group motifs that serve similar reasoning functions (e.g., "Transit → Affordability → Support" ≈ "Mobility → Value → Support")

## Usage

1. Install dependencies:

```bash
pip install -r src/phase2_motif/requirements.txt
```

2. Run basic topology-based motif analysis:

```bash
python src/phase2_motif/motif_analysis.py
```

3. Run semantic motif extraction:

```bash
python src/phase2_motif/extract_semantic_motifs.py
```

4. Run additional visualizations:

```bash
python src/phase2_motif/visualize_motifs.py
```

## Semantic Similarity Module

The `semantic_similarity.py` module provides a streamlined approach to comparing node and motif similarities. It follows these steps:

1. **Topological Classification**: First identifies motifs based on structure (M1: Chain, M2: Fork, M3: Collider)
2. **Semantic Subgrouping**: Within each topological class, groups motifs based on semantic similarity

### Using the Semantic Similarity Engine

```python
from semantic_similarity import SemanticSimilarityEngine, analyze_motifs

# Create a semantic similarity engine
engine = SemanticSimilarityEngine(use_wordnet=True)

# Calculate similarity between two node labels
similarity = engine.node_similarity("transit_access", "mobility_options")

# For complete motif analysis on a graph
graph = nx.DiGraph()  # Your causal graph
semantic_groups = analyze_motifs(graph, min_similarity=0.7)
```

### Word Vectors (Optional)

If you need more advanced semantic similarity, you can optionally use word vectors:

1. Download pre-trained word vectors (e.g., Google's Word2Vec or GloVe vectors)
2. Update the `WORD_VECTORS_PATH` constant in the module:

```python
# In semantic_similarity.py
WORD_VECTORS_PATH = "/path/to/your/wordvectors.bin"
```

3. Set `use_word_vectors=True` when initializing the engine:

```python
engine = SemanticSimilarityEngine(use_wordnet=True, use_word_vectors=True)
```

### Which Files to Keep

For minimal implementation, these files are essential:
- `semantic_similarity.py`: Core semantic comparison functionality
- `motif_analysis.py`: Basic topological motif detection
- `extract_semantic_motifs.py`: Script to run the full pipeline

## Optimized Motif Extraction

The latest implementation includes several optimizations to ensure more meaningful motif detection:

1. **Motif Templates**: Uses predefined motif templates for each pattern type (M1, M2.x, M3.x)
2. **Central Node Tracking**: 
   - For fork patterns (M2.x): Tracks only center nodes (source nodes)
   - For collider patterns (M3.x): Tracks only sink nodes (target nodes)
   - Prevents double-counting of nested motifs
3. **Size Constraints**:
   - M1 (Chain) patterns: Fixed at exactly 3 nodes
   - M2/M3 patterns: Variable size from 3-5 nodes, with larger motifs prioritized
4. **Priority Order**: Processes motifs in order of complexity (larger first) to avoid identifying sub-patterns of already discovered motifs

### Example: Fork Patterns

For fork patterns (M2.x), we first identify larger forks (M2.3), then extended forks (M2.2), and finally basic forks (M2.1). If a node is already the center of a larger fork, it won't be considered as the center of a smaller fork.

## Output Files

Generated in `src/phase2_motif/output/`:
- `motif_analysis_results.json`: Basic motif analysis results
- `motif_distribution.png`: Normalized motif frequency visualization
- `motif_counts.png`: Raw motif counts visualization
- `motif_groups.png`: Visualization of chain, fork, and collider pattern groups
- `motif_by_demographic.png`: Analysis of motif patterns by demographic group
- `reasoning_complexity.png`: Analysis of reasoning complexity across samples
- `kmeans_clusters.png`: Clustering results
- `cluster_visualization.png`: 2D visualization of sample clusters
- `kmeans_results.json`: Cluster details

Generated in `src/phase2_motif/output/semantic_motifs/`:
- `semantic_motifs.json`: Identified semantic motif groups and annotations
- `motif_summary.csv`: Summary statistics for all identified motifs
- `motif_group_*.png`: Visualizations of each semantic motif group

## Implementation Details

### Motif Detection using Graph Isomorphism

We use NetworkX's subgraph isomorphism algorithm to find all instances of predefined motif patterns:

1. **Template Creation**: Define each motif as a small directed graph template
2. **Graph Conversion**: Convert JSON graph data to NetworkX DiGraph objects
3. **Isomorphism Detection**: Use VF2 algorithm to find all subgraph matches for each motif pattern
4. **Deduplication**: Prevent overlapping patterns using central node tracking

### Semantic Similarity Measurement

We compute semantic similarity between nodes and motifs using:

1. **Label Preprocessing**: Process node labels to extract key terms
2. **WordNet Similarity**: Use linguistic semantic similarity based on lexical relationships
3. **Similarity Thresholding**: Apply minimum similarity thresholds to group related motifs
4. **Position-Based Weighting**: Give more weight to source and sink nodes in similarity calculations

### Motif Variations

We analyze multiple variations of each basic pattern type:

1. **Chain (M1)**: Simple three-node path A → B → C
   - Indicates sequential reasoning
   - Always exactly 3 nodes

2. **Fork Variations (M2.x)**: One-to-many patterns
   - Basic (M2.1): One node connecting to two others
   - Extended (M2.2): One node connecting to three others
   - Large (M2.3): One node connecting to four or more others
   - Indicates branching effects/implications

3. **Collider Variations (M3.x)**: Many-to-one patterns
   - Basic (M3.1): Two nodes connecting to one
   - Extended (M3.2): Three nodes connecting to one
   - Large (M3.3): Four or more nodes connecting to one
   - Indicates multiple causes for a single effect

The implementation is more robust than manual counting as it can handle complex structural patterns and semantic variations, while avoiding double-counting overlapping patterns.

## Analysis Method

1. **Motif Definition**: Define basic motifs as directed graph templates with variations

2. **Feature Construction**:
   - For each user j's causal graph Gj, construct feature vector Vj
   - Vj contains:
     - dj: demographic label
     - sj: stance on upzoning
     - X(j)i: ratio/count of motif Mi in Gj

3. **Clustering**: Apply K-means clustering based on motif frequency vectors

4. **Complexity Analysis**: Evaluate reasoning complexity through weighted motif patterns

5. **Semantic Grouping**: Group functionally equivalent motifs using semantic similarity 