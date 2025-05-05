# Motif Library for Cognitive Causal Graphs

Implementation for extracting, analyzing, and augmenting motifs in cognitive causal graphs.

## Overview

This project extends the original motif analysis code with a more sophisticated approach that:

1. Implements a two-step motif extraction process:
   - **Topology-Based Candidate Grouping**: Identifying structural patterns using graph isomorphism
   - **Semantic Filtering**: Grouping functionally equivalent patterns based on semantic similarity

2. Adds data augmentation capabilities for motif patterns:
   - **Nearest Neighbor Weighted**: Combining features of similar motifs
   - **Bootstrapping**: Resampling from the distribution of node/edge features

3. Provides visualization and analysis tools for the extracted motifs

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r src/phase2_motif/requirements.txt
```

## Usage

### Basic Usage

To run the motif analysis with the new implementation:

```bash
python src/phase2_motif/run_motif_analysis.py --samples-dir data/samples --output-dir output
```

For more options:

```bash
python run_motif_analysis.py --help
```

### Extended Analysis

For extended analysis with data augmentation:

```bash
python src/phase2_motif/extended_motif_analysis.py --samples-dir data/samples --output-dir output --use-library --augment
```

### MotifLibrary (plug-in version..)

```python
from motif_library import MotifLibrary, load_graph_from_json

# Initialize the library
library = MotifLibrary(min_semantic_similarity=0.4)

# Load a graph
G = load_graph_from_json('path/to/graph.json')

# Extract motifs
library.extract_topological_motifs(G)

# Apply semantic filtering
library.apply_semantic_filtering()

# Calculate motif vector for a graph
motif_vector = library.calculate_motif_vector(G)

# Augment with nearest neighbor approach
library.augment_by_nearest_neighbor(num_samples=5)

# Augment with bootstrapping
library.augment_by_bootstrapping(num_samples=5)

# Visualize motif groups
library.visualize_all_groups(output_dir='output')

# Save the library
library.save_library('motif_library.json')
```

## Motif Types

The library identifies and analyzes the following motif patterns:

1. **Chain Motifs (M1)**:
   - Three-node path A → B → C
   - Represents sequential reasoning

2. **Fork Motifs (M2)**:
   - M2.1: Basic fork (1-to-2): A → B, A → C
   - M2.2: Extended fork (1-to-3): A → B, A → C, A → D
   - M2.3: Large fork (1-to-4+): A → B, A → C, A → D, A → E, ...
   - Represents branching effects/implications

3. **Collider Motifs (M3)**:
   - M3.1: Basic collider (2-to-1): A → C, B → C
   - M3.2: Extended collider (3-to-1): A → D, B → D, C → D
   - M3.3: Large collider (4+-to-1): A → E, B → E, C → E, D → E, ...
   - Represents multiple causes for a single effect

## Data Augmentation

The library provides two methods for data augmentation:

1. **Nearest Neighbor Weighted**:
   - Generates new synthetic motifs by combining features of similar motifs
   - Maintains semantic coherence within motif groups
   - Best for more realistic variations based on existing motifs

2. **Bootstrapping**:
   - Generates new synthetic motifs by resampling from the distribution of node/edge features
   - Creates more diverse variations that may be less semantically coherent
   - Better for increasing pattern diversity

## Output Files

The analysis generates the following output files:

- `motif_library.json`: Serialized motif library with all extracted motifs
- `motif_summary.json`: Summary statistics for the motif library
- `motif_summary.csv`: Detailed information about each motif group
- `motif_distribution.png`: Normalized motif frequency visualization
- `motif_counts.png`: Raw motif counts visualization
- `motif_heatmap.png`: Heatmap of motif frequencies across samples
- `kmeans_clusters.png`: KMeans clustering results
- `cluster_visualization.png`: 2D visualization of sample clusters
- `motif_groups/`: Directory with visualizations of each motif group

For augmented motifs:
- `augmented/augmented_library.json`: Serialized library with augmented motifs
- `augmented/augmentation_summary.json`: Summary of data augmentation results
- `augmented/motif_groups/`: Directory with visualizations of augmented motifs