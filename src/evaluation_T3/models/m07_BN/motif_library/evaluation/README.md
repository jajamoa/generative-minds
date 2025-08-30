# Motif Library Evaluation

Simple evaluation metrics for motif library quality. Talk is cheap, show me the numbers.

## Coverage

Measures how well motifs capture semantic concepts from original graphs.

### Node Coverage
Evaluates concept overlap in node labels:
```
Original Graph:        Motif:
"housing cost"   →    "rent prices"
"community"      →    "neighborhood"
"safety"         →    "crime rate"
```
Uses BERT embeddings to capture semantic similarity between concepts. A high score means motifs use similar vocabulary as original graphs.

#### Calculation
1. Extract all unique node labels from original graphs and motifs
2. Convert labels to frequency vectors:
   ```
   Original: {"housing": 0.3, "community": 0.2, "safety": 0.5}
   Motif: {"rent": 0.4, "neighborhood": 0.3, "crime": 0.3}
   ```
3. Calculate cosine similarity between frequency vectors
4. Higher similarity = better coverage of original concepts

### Edge Coverage
Evaluates relationship patterns:
```
Original:                     Motif:
"housing cost" → "displacement"      "rent prices" → "eviction"
"crime rate" → "property value"      "safety" → "home prices"
```

#### Calculation
1. Extract edge patterns as source_label + "_" + target_label
2. Convert to frequency vectors like node coverage
3. Calculate cosine similarity
4. Lower score suggests missing important relationships

## Coherence (lower = better)

Measures label distinctness within motifs. For example:

Good (Low Coherence):
```
Motif nodes: ["housing cost", "job opportunities", "transportation"]
- Distinct concepts, clear roles
```

Bad (High Coherence):
```
Motif nodes: ["housing cost", "rent prices", "home prices"]
- Too similar, redundant concepts
```

#### Calculation
1. For each motif:
   ```python
   similarities = []
   for each pair of node labels (i,j) in motif:
       sim = bert_similarity(label_i, label_j)
       similarities.append(sim)
   motif_coherence = mean(similarities)
   ```
2. Average across all motifs:
   ```python
   overall_coherence = mean(motif_coherences)
   ```
3. Target: coherence < 0.4 for good concept diversity

## Diversity

### Semantic Diversity
Measures concept variety across motif groups:

High Diversity:
```
Group 1: housing cost → displacement
Group 2: crime rate → property value
Group 3: job market → rent prices
```

Low Diversity:
```
Group 1: housing cost → property value
Group 2: rent prices → home value
Group 3: apartment cost → house prices
```

#### Calculation
1. For each motif group:
   - Combine all node labels into group text
2. Calculate pairwise BERT similarities between groups
3. Semantic diversity = 1 - average similarity
4. Higher score = more diverse concepts across groups

### Structural Diversity
Evaluates variety in graph patterns using:
- Node count distribution
- Edge patterns
- Degree distributions

Example features:
```
Group 1: avg_nodes=3, avg_edges=2, degree_std=0.5
Group 2: avg_nodes=5, avg_edges=6, degree_std=1.2
```

#### Calculation
1. Extract structural features per group:
   ```python
   features = [
       mean(node_counts),
       mean(edge_counts),
       mean(degrees),
       std(node_counts),
       std(edge_counts),
       std(degrees)
   ]
   ```
2. Calculate average pairwise Euclidean distances
3. Normalize by maximum possible distance
4. Higher score = more structural variety

### Distribution Entropy
Measures balance between pattern types:

Good Distribution:
```
Chain patterns:    32% (320 motifs)
Fork patterns:     35% (350 motifs)
Collider patterns: 33% (330 motifs)
```

Poor Distribution:
```
Chain patterns:    80% (800 motifs)
Fork patterns:     15% (150 motifs)
Collider patterns:  5% (50 motifs)
```

#### Calculation
1. Calculate probability for each pattern type:
   ```python
   p = count_pattern_type / total_motifs
   ```
2. Calculate normalized entropy:
   ```python
   entropy = -sum(p * log2(p)) / log2(n_types)
   ```
3. Range: 0-1, higher = more balanced distribution

## Basic Pattern Types

### Chain (M1)
Sequential causation:
```
A → B → C
Example: housing cost → displacement → community change
```

### Fork (M2.x)
One cause, multiple effects:
```
     B
A → C
     D
Example: housing cost → {displacement, poverty, health}
```

### Collider (M3.x)
Multiple causes, one effect:
```
A
B → D
C
Example: {income, jobs, savings} → housing quality
```