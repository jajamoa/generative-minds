# Cognitive SCM (Structural Causal Model) – Minimal Per-Person Data Schema

This document defines a minimal yet expressive schema for representing individual reasoning structures as simplified structural causal models (SCMs). Each SCM is derived from a set of structured QA pairs from interviews and is designed to support causal simulation, motif extraction, and reasoning trace analysis.

---

## Top-Level Structure

Each record corresponds to one individual.

```json
{
  "agent_id": "string",
  "nodes": [...],
  "functions": [...],
  "qas": [...]
}
```

---

## 1. nodes

Each node represents a belief variable involved in the person's reasoning process.

**Required fields:**

| Field | Type | Description |
|------|------|------|
| id | string | Unique identifier for the variable |
| type | string | One of: "binary", "continuous" |
| range | array | Required if type is "continuous" (e.g., [0, 1]) |
| semantic_role | string | Optional. One of: "external_state", "internal_affect", "behavioral_intention" |

**Note on multi-category variables:**
For variables that would naturally have multiple categories (like housing types, transportation modes, etc.), represent them as multiple binary variables. For example, instead of a single `HousingType` categorical variable, use multiple binary variables like `LivesInApartment`, `LivesInTownhouse`, etc.

Example:
```json
{
  "id": "Sunlight",
  "type": "continuous",
  "range": [0.0, 1.0],
  "semantic_role": "external_state"
}
```

```json
{
  "id": "LivesInApartment",
  "type": "binary",
  "semantic_role": "external_state"
}
```

```json
{
  "id": "SupportsPolicy",
  "type": "binary",
  "semantic_role": "behavioral_intention"
}
```

---

## 2. functions

Each function defines a directed causal relationship from one or more inputs to a target variable, represented as a simplified structural function.

**Required fields:**

| Field | Type | Description |
|------|------|------|
| target | string | Target variable (must exist in nodes) |
| inputs | array of strings | Parent variables (must exist in nodes) |
| function_type | string | One of: "sigmoid", "threshold" |
| parameters | object | Structure depends on function_type |
| noise_std | float | Standard deviation of noise (≥ 0) |
| support_qas | array of strings | List of QA IDs that support this function |
| confidence | float | Optional, between 0.0 and 1.0 |

**Parameters by type:**

* sigmoid:
  * weights: array of floats, one per input
  * bias: float
* threshold:
  * threshold: float (threshold value)
  * direction: "greater" or "less" (trigger condition)

Example:
```json
{
  "target": "Mood",
  "inputs": ["Sunlight"],
  "function_type": "sigmoid",
  "parameters": {
    "weights": [-2.5],
    "bias": 1.2
  },
  "noise_std": 0.3,
  "support_qas": ["qa_01"],
  "confidence": 0.8
}
```
```json
{
  "target": "MoveIntent",
  "inputs": ["NoiseLevel"],
  "function_type": "threshold",
  "parameters": {
    "threshold": 0.7,
    "direction": "greater"
  },
  "noise_std": 0.2,
  "support_qas": ["qa_05"]
}
```

---

## 3. qas

Structured representation of the source interview questions and answers that support the belief graph.

**Required fields:**

| Field | Type | Description |
|------|------|------|
| qa_id | string | Unique identifier |
| question | string | Interview prompt |
| answer | string | Participant response |
| parsed_belief | object | Extracted belief statement with structure and strength components |

**parsed_belief fields:**

| Field | Type | Description |
|------|------|------|
| belief_structure | object | Causal structure information for motif extraction and causal graph construction |
| belief_strength | object | Quantifiable reasoning information for function fitting |
| counterfactual | string | Optional explanation of imagined alternative |

**belief_structure fields:**

| Field | Type | Description |
|------|------|------|
| from | string | Cause variable |
| to | string | Effect variable |
| direction | string | "positive", "negative", or "unknown" |

**belief_strength fields:**

| Field | Type | Description |
|------|------|------|
| estimated_probability | float | Probability estimate for CPT or function parameter construction (0.0-1.0) |
| confidence_rating | float | Subjective confidence level for noise modeling or sampling strength (0.0-1.0) |

Example:
```json
{
  "qa_id": "qa_01",
  "question": "Why do you oppose tall buildings?",
  "answer": "They block sunlight and it affects my mood.",
  "parsed_belief": {
    "belief_structure": {
      "from": "Sunlight",
      "to": "Mood",
      "direction": "negative"
    },
    "belief_strength": {
      "estimated_probability": 0.75,
      "confidence_rating": 0.8
    },
    "counterfactual": "If there were no tall buildings, I'd feel better."
  }
}
```

---

## Function Type Justification

**Kept Function Types:**

| Function Type | Kept | Reason |
|---------|------|------|
| sigmoid | Yes | Captures monotonic causal trends; supports probabilistic simulation; interpretable |
| threshold | Yes | Represents triggering behavior; suitable for rule-like beliefs |

**Removed Function Types:**

| Function Type | Removed | Reason |
|---------|------|------|
| linear | Yes | Redundant with sigmoid; harder to constrain |
| categorical-map | Yes | Inflexible; poor generalization |
| rule_based | Yes | Fragile; not reusable across individuals |
| textual | Yes | Non-executable; not usable for simulation |

Only "sigmoid" and "threshold" are retained for structural clarity, interpretability, and composability across motifs.

---

## Two-Layer Belief Design Benefits

| Design Element | Function | Supported Task |
|----------------|----------|----------------|
| belief_structure | Supports graph construction and motif extraction | Model structure learning |
| estimated_probability | Builds CPT or function parameters | Reasoning and inference |
| confidence_rating | Models noise or sampling strength | Simulation and counterfactual generation |
| counterfactual | Strengthens causal direction judgment | Motif classification, explainability |

---

## Example SCM Record
```json
{
  "agent_id": "user_001",
  "nodes": [
    { "id": "Sunlight", "type": "continuous", "range": [0, 1], "semantic_role": "external_state" },
    { "id": "Mood", "type": "continuous", "range": [0, 1], "semantic_role": "internal_affect" },
    { "id": "NoiseLevel", "type": "continuous", "range": [0, 1], "semantic_role": "external_state" },
    { "id": "MoveIntent", "type": "binary", "semantic_role": "behavioral_intention" },
    { "id": "LivesInApartment", "type": "binary", "semantic_role": "external_state" },
    { "id": "LivesInTownhouse", "type": "binary", "semantic_role": "external_state" }
  ],
  "functions": [
    {
      "target": "Mood",
      "inputs": ["Sunlight"],
      "function_type": "sigmoid",
      "parameters": { "weights": [-2.5], "bias": 1.2 },
      "noise_std": 0.3,
      "support_qas": ["qa_01"],
      "confidence": 0.8
    },
    {
      "target": "MoveIntent",
      "inputs": ["NoiseLevel"],
      "function_type": "threshold",
      "parameters": { "threshold": 0.7, "direction": "greater" },
      "noise_std": 0.2,
      "support_qas": ["qa_05"]
    }
  ],
  "qas": [
    {
      "qa_id": "qa_01",
      "question": "Why do you oppose tall buildings?",
      "answer": "They block sunlight and that affects my mood.",
      "parsed_belief": {
        "belief_structure": {
          "from": "Sunlight",
          "to": "Mood",
          "direction": "negative"
        },
        "belief_strength": {
          "estimated_probability": 0.75,
          "confidence_rating": 0.8
        },
        "counterfactual": "If there were no tall buildings, I'd feel better."
      }
    },
    {
      "qa_id": "qa_05",
      "question": "What would make you consider moving?",
      "answer": "If the noise gets too bad, I'll move out.",
      "parsed_belief": {
        "belief_structure": {
          "from": "NoiseLevel",
          "to": "MoveIntent",
          "direction": "positive"
        },
        "belief_strength": {
          "estimated_probability": 0.85,
          "confidence_rating": 0.7
        }
      }
    }
  ]
}
```





