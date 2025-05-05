
# Cognitive SCM – Minimal Node-ID-Based Data Schema

This document defines a minimal and interpretable JSON schema for representing individual-level Structural Causal Models (SCMs), derived from human interviews. These SCMs formalize a person's reasoning trace by representing belief variables and their causal relationships. The schema is designed to support simulation, motif extraction, and structured analysis.

---

## Top-Level Structure

Each record corresponds to one individual.

```json
{
  "agent_id": "string",
  "demographics": { ... },
  "nodes": { ... },
  "edges": { ... },
  "qas": [ ... ]
}
```

---

## 1. `agent_id`

| Field     | Type   | Description                        |
|-----------|--------|------------------------------------|
| agent_id  | string | Unique identifier for the individual |

---

## 2. `demographics`

Basic demographic information about the individual.

| Field       | Type   | Description                            |
|-------------|--------|----------------------------------------|
| age         | number | Age of the individual                  |
| income      | string | Income bracket                         |
| education   | string | Highest level of education completed   |
| occupation  | string | Current or past occupation             |

Example:
```json
{
  "age": 34,
  "income": "$50,000–$99,999",
  "education": "college graduate",
  "occupation": "urban planner"
}
```

---

## 3. `nodes`

Nodes represent belief variables in the person's reasoning process. Each node has a unique `node_id` (e.g., `"n1"`, `"n2"`).

### Required Fields

| Field           | Type             | Description |
|----------------|------------------|-------------|
| label           | string           | Human-readable concept name |
| type            | string           | One of: `"binary"` or `"continuous"` |
| range           | array            | Required if type is `"continuous"` (e.g., `[0.0, 1.0]`) |
| values          | array            | Required if type is `"binary"`; always `[true, false]` |
| semantic_role   | string           | One of: `"external_state"`, `"internal_affect"`, `"behavioral_intention"` |
| appearance      | object           | Includes `qa_ids` and `frequency` |
| incoming_edges  | array of strings | List of edge IDs targeting this node |
| outgoing_edges  | array of strings | List of edge IDs emitted from this node |

### Node Type Guidelines

- `"binary"` nodes represent boolean propositions and must include:
  ```json
  "values": [true, false]
  ```
- `"continuous"` nodes represent real-valued quantities and must include:
  ```json
  "range": [min, max]
  ```

### Cognitive Role (semantic_role)

| semantic_role         | Description                                  | Cognitive Mapping      |
|------------------------|----------------------------------------------|-------------------------|
| external_state         | Observable or inferred world conditions      | Input / Belief          |
| internal_affect        | Internal emotional or evaluative states      | Affect / Preference     |
| behavioral_intention   | Actions, intentions, or behavioral choices   | Output / Decision       |

Example:
```json
"n2": {
  "label": "Mood",
  "type": "binary",
  "values": [true, false],
  "semantic_role": "internal_affect",
  "appearance": {
    "qa_ids": ["qa_01"],
    "frequency": 1
  },
  "incoming_edges": ["e1"],
  "outgoing_edges": ["e2"]
}
```

---

## 4. `edges`

Edges define directed causal relationships between nodes. Each edge must include a functional form and is keyed by a unique `edge_id`.

### Required Fields

| Field        | Type   | Description |
|--------------|--------|-------------|
| from         | string | Source `node_id` |
| to           | string | Target `node_id` |
| function     | object | Parameterized causal function |
| support_qas  | array  | List of supporting QA IDs |

### `function` Object

| Field         | Type             | Description |
|---------------|------------------|-------------|
| target        | string           | Target `node_id` |
| inputs        | array of strings | Parent `node_id`s |
| function_type | string           | One of: `"sigmoid"` or `"threshold"` |
| parameters    | object           | See details below |
| noise_std     | float            | Gaussian noise (≥ 0) |
| support_qas   | array            | Supporting QA IDs |
| confidence    | float (optional) | Confidence score (0.0–1.0) |

### Parameters by Function Type

- `"sigmoid"`:
```json
"parameters": {
  "weights": [-2.5],
  "bias": 1.2
}
```

- `"threshold"`:
```json
"parameters": {
  "threshold": 0.6,
  "direction": "less"
}
```

---

## 5. `qas`

Each QA record contains a question-answer pair and the extracted causal belief structure.

### Required Fields

| Field         | Type   | Description |
|---------------|--------|-------------|
| qa_id         | string | Unique identifier |
| question      | string | Interview question |
| answer        | string | Verbatim response |
| parsed_belief | object | Extracted causal belief |

### `parsed_belief` Fields

| Field            | Type   | Description |
|------------------|--------|-------------|
| belief_structure | object | Causal link between two node IDs |
| belief_strength  | object | Estimated strength and confidence |
| counterfactual   | string | Optional contrastive explanation |

**belief_structure**:
```json
{
  "from": "n1",
  "to": "n2",
  "direction": "negative"
}
```

**belief_strength**:

```json
{
  "estimated_probability": 0.75,
  "confidence_rating": 0.8
}
```

---
