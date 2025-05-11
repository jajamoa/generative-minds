# **Causal Graph Data Schema**

This schema defines a minimal format for capturing causal relationships extracted from question-answer (QA) interactions. It includes:

- **Nodes**: Concepts (e.g., concerns, effects, motivations)  
- **Edges**: Causal links between nodes, with polarity and confidence  
- **QA History**: Source QA pairs that support node and edge generation  

Each element is keyed by a unique ID.

---

## Format

### 1. Nodes

```json
"node_id": {
  "label": "string",               // Concept name (e.g. "housing_supply")
  "confidence": float,             // 0.0–1.0, model's certainty about the node's existence
  "source_qa": ["qa_id"],          // Supporting QA IDs
  "incoming_edges": ["edge_id"],
  "outgoing_edges": ["edge_id"]
}
```

### 2. Edges

```json
"edge_id": {
  "source": "node_id",             // From-node
  "target": "node_id",             // To-node
  "aggregate_confidence": float,  // 0.0–1.0, strength of causal link based on all evidence
  "evidence": [
    { "qa_id": "qa_id", "confidence": float }  // Individual evidence confidence (0.0–1.0)
  ],
  "modifier": float                // Causal direction and strength: 
                                  // range [-1.0, 1.0]
                                  // positive: supports/causes
                                  // negative: opposes/prevents
}
```

### 3. QA History

```json
"qa_id": {
  "question": "string",
  "answer": "string",
  "extracted_pairs": [
    {
      "source": "node_id",
      "target": "node_id",
      "confidence": float        // 0.0–1.0, model’s confidence in extracting this relation
    }
  ]
}
```

---

## Notes

- **Confidence**: A probability-like score (0–1) indicating how certain the system is about the existence of a node or link.
- **Modifier**: Represents the causal *valence* and *strength*.
  - `+1.0`: Strong positive cause (e.g., “X leads to Y”)
  - `-1.0`: Strong negative effect (e.g., “X prevents Y”)
  - Values near `0`: Weak or ambiguous influence

---

## Example 1: Daily Life

```json
nodes = {
  "n1": { "label": "exercise", "confidence": 1.0, "source_qa": ["qa_001"], "incoming_edges": [], "outgoing_edges": ["e1", "e2"] },
  "n2": { "label": "stress", "confidence": 0.9, "source_qa": ["qa_002"], "incoming_edges": [], "outgoing_edges": ["e3"] },
  "n3": { "label": "better_sleep", "confidence": 0.95, "source_qa": ["qa_001", "qa_002"], "incoming_edges": ["e1", "e3"], "outgoing_edges": ["e4"] },
  "n4": { "label": "daily_energy", "confidence": 0.9, "source_qa": ["qa_001"], "incoming_edges": ["e2", "e4"], "outgoing_edges": [] }
}

edges = {
  "e1": {
    "source": "n1",
    "target": "n3",
    "aggregate_confidence": 0.95,
    "evidence": [{ "qa_id": "qa_001", "confidence": 0.95 }],
    "modifier": 1.0
  },
  "e2": {
    "source": "n1",
    "target": "n4",
    "aggregate_confidence": 0.85,
    "evidence": [{ "qa_id": "qa_001", "confidence": 0.85 }],
    "modifier": 1.0
  },
  "e3": {
    "source": "n2",
    "target": "n3",
    "aggregate_confidence": 0.8,
    "evidence": [{ "qa_id": "qa_002", "confidence": 0.8 }],
    "modifier": -0.8
  },
  "e4": {
    "source": "n3",
    "target": "n4",
    "aggregate_confidence": 0.9,
    "evidence": [{ "qa_id": "qa_002", "confidence": 0.9 }],
    "modifier": 1.0
  }
}

qa_history = {
  "qa_001": {
    "question": "What daily habits help you stay healthy?",
    "answer": "I exercise regularly. It helps me sleep better and gives me more energy during the day.",
    "extracted_pairs": [
      { "source": "n1", "target": "n3", "confidence": 0.95 },
      { "source": "n1", "target": "n4", "confidence": 0.85 }
    ]
  },
  "qa_002": {
    "question": "What affects your sleep quality?",
    "answer": "When I feel stressed, I can't sleep well. But if I sleep better, I feel more energetic.",
    "extracted_pairs": [
      { "source": "n2", "target": "n3", "confidence": 0.8 },
      { "source": "n3", "target": "n4", "confidence": 0.9 }
    ]
  }
}
```

---

## Example 2: Upzoning Policy

```json
nodes = {
  "n1": { "label": "housing_supply", "confidence": 1.0, "source_qa": ["qa_001"], "incoming_edges": [], "outgoing_edges": ["e1"] },
  "n2": { "label": "rent_increase", "confidence": 0.85, "source_qa": ["qa_002"], "incoming_edges": [], "outgoing_edges": ["e2"] },
  "n3": { "label": "policy_support", "confidence": 1.0, "source_qa": ["qa_001", "qa_002"], "incoming_edges": ["e1", "e2"], "outgoing_edges": [] }
}

edges = {
  "e1": { "source": "n1", "target": "n3", "aggregate_confidence": 1.0, "evidence": [{ "qa_id": "qa_001", "confidence": 1.0 }], "modifier": 1.0 },
  "e2": { "source": "n2", "target": "n3", "aggregate_confidence": 0.85, "evidence": [{ "qa_id": "qa_002", "confidence": 0.85 }], "modifier": -0.85 }
}

qa_history = {
  "qa_001": {
    "question": "Do you support or oppose upzoning?",
    "answer": "I support it because it increases housing supply.",
    "extracted_pairs": [{ "source": "n1", "target": "n3", "confidence": 1.0 }]
  },
  "qa_002": {
    "question": "What concerns do you have?",
    "answer": "I’m worried rents will go up.",
    "extracted_pairs": [{ "source": "n2", "target": "n3", "confidence": 0.85 }]
  }
}
```
