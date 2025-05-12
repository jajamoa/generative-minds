# Experiment Module

A framework for evaluating agent opinions on zoning proposals.

## Directory Structure

```
experiment/
├── protocols/           # Experiment protocols
├── eval/
│   ├── data/            # Simulation data
│   ├── evaluators/      # Evaluation metrics
│   └── utils/           # Utilities
├── log/                 # Results
└── run_experiment.py    # Main runner
```

## Quick Start

```bash
# Run an experiment
python -m experiment.run_experiment --protocol protocols/sf_survey_eval.yaml

# Evaluate existing experiment
python -m experiment.run_experiment --protocol protocols/sf_survey_eval.yaml --eval-only --experiment-dir log/experiment_dir
```

## Protocol Format

```yaml
name: "experiment_name"
description: "Description"

# Model configuration
model: "census"         # Model type
population: 30          # Number of agents

# Input data
input:
  proposals:
    - "path/to/proposal.json"  # Relative to data/

# Evaluation
evaluation:
  ground_truth: "path/to/ground_truth.json"
  evaluators:
    - "opinion_score"
    - "reason_match"
```

## Output

Each experiment creates a directory in `log/` containing input, output and evaluation results.

## Available Evaluators

- **Opinion Score**: Evaluates opinion score accuracy (MAE, correlation)
- **Reason Match**: Evaluates reason selection accuracy (Jaccard similarity)