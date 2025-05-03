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



⸻

1. nodes

Each node represents a belief variable involved in the person’s reasoning process.

Required fields:

Field	Type	Description
id	string	Unique identifier for the variable
type	string	One of: "binary", "categorical", "continuous"
range	array	Required if type is "continuous" (e.g., [0, 1])
values	array	Required if type is "categorical" (e.g., ["happy", "neutral", "sad"])

Example:

{
  "id": "Sunlight",
  "type": "continuous",
  "range": [0.0, 1.0]
}

{
  "id": "HousingType",
  "type": "categorical",
  "values": ["apartment", "townhouse", "single-family"]
}

{
  "id": "SupportsPolicy",
  "type": "binary"
}



⸻

2. functions

Each function defines a directed causal relationship from one or more inputs to a target variable, represented as a simplified structural function.

Required fields:

Field	Type	Description
target	string	Target variable (must exist in nodes)
inputs	array of strings	Parent variables (must exist in nodes)
function_type	string	One of: "sigmoid", "threshold"
parameters	object	Structure depends on function_type
noise_std	float	Standard deviation of noise (≥ 0)
support_qas	array of strings	List of QA IDs that support this function
confidence	float	Optional, between 0.0 and 1.0

Parameters by type:
	•	sigmoid:
	•	weights: array of floats, one per input
	•	bias: float
	•	threshold:
	•	threshold: float (threshold value)
	•	direction: "greater" or "less" (trigger condition)

Example:

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



⸻

3. qas

Structured representation of the source interview questions and answers that support the belief graph.

Required fields:

Field	Type	Description
qa_id	string	Unique identifier
question	string	Interview prompt
answer	string	Participant response
parsed_belief	object	Extracted belief statement

parsed_belief fields:

Field	Type	Description
from	string	Cause variable
to	string	Effect variable
direction	string	"positive", "negative", or "unknown"
confidence	float	Optional (0.0–1.0)
counterfactual	string	Optional explanation of imagined alternative

Example:

{
  "qa_id": "qa_01",
  "question": "Why do you oppose tall buildings?",
  "answer": "They block sunlight and it affects my mood.",
  "parsed_belief": {
    "from": "Sunlight",
    "to": "Mood",
    "direction": "negative",
    "confidence": 0.8
  }
}



⸻

Function Type Justification

Function Type	Kept	Reason
sigmoid	Yes	Captures monotonic causal trends; supports probabilistic simulation; interpretable
threshold	Yes	Represents triggering behavior; suitable for rule-like beliefs

Function Type	Removed	Reason
linear	Yes	Redundant with sigmoid; harder to constrain
categorical-map	Yes	Inflexible; poor generalization
rule_based	Yes	Fragile; not reusable across individuals
textual	Yes	Non-executable; not usable for simulation

Only "sigmoid" and "threshold" are retained for structural clarity, interpretability, and composability across motifs.

⸻

Example SCM Record

{
  "agent_id": "user_001",
  "nodes": [
    { "id": "Sunlight", "type": "continuous", "range": [0, 1] },
    { "id": "Mood", "type": "continuous", "range": [0, 1] },
    { "id": "NoiseLevel", "type": "continuous", "range": [0, 1] },
    { "id": "MoveIntent", "type": "binary" }
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
        "from": "Sunlight",
        "to": "Mood",
        "direction": "negative",
        "confidence": 0.8
      }
    },
    {
      "qa_id": "qa_05",
      "question": "What would make you consider moving?",
      "answer": "If the noise gets too bad, I’ll move out.",
      "parsed_belief": {
        "from": "NoiseLevel",
        "to": "MoveIntent",
        "direction": "positive"
      }
    }
  ]
}






