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

### Edge Coverage
Evaluates relationship patterns:
```
Original:                     Motif:
"housing cost" → "displacement"      "rent prices" → "eviction"
"crime rate" → "property value"      "safety" → "home prices"
```
Captures how well motifs preserve important causal relationships. Lower score might indicate missing important relationship patterns.

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
Higher variance = more structural diversity

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