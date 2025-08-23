# Method

NOTE: Working in progress.


## Folder Structure

```
method/
├── README.md                    # This file
├── data/                        # Evaluation datasets
│   └── 30ppl_eval_v1_0728/     # dataset (v1, July 28)
│       ├── [30 participant directories]/  # Individual participant data
│       └── split_log.json       # Dataset split information
├── graph_reconstruction/        # Graph reconstruction algorithms
├── motif_library/              # Motif extraction and context graph builder
├── configs/                    # YAML configs (e.g., context_graph/qwen_context_default.yaml)
├── utils/                      # Shared utilities (LLM client, logging)
└── run.sh                      # Example runner
```

## Overview

The `method/` directory is organized into three main components:

- **`data/`**: Contains source datasets
- **`graph_reconstruction/`**: Algorithms and methods for reconstructing causal graphs
- **`motif_library/`**: Components for motif extraction and library construction

## Setup

1) Python environment

```
cd method
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
# If you see missing packages at runtime:
pip install networkx requests python-dotenv pyyaml tqdm
```

2) .env configuration (LLM, optional)

Create `method/.env` (or project `.env`) with your Qwen key:

```
QWEN_API_KEY=your_api_key_here
```

You can customize the LLM prompt and API settings in:
`method/configs/context_graph/qwen_context_default.yaml`.

## How to Run

Run from the `method/` directory.

- Heuristic-only (no LLM):

```
python motif_library/context_graph_builder.py \
  --dataset data/30ppl_eval_v1_0728 \
  --out intermediate_data/30ppl_eval_v1_0728 \
  --participant 1962Michael
```

- Use Qwen LLM (requires `.env` key):

```
python motif_library/context_graph_builder.py \
  --dataset data/30ppl_eval_v1_0728 \
  --out intermediate_data/30ppl_eval_v1_0728 \
  --llm_cfg context_graph/qwen_context_default.yaml
```

- All participants: omit `--participant`.

- Or use the helper script:

```
sh run.sh
```

## Outputs

Per comment (example):

```
method/intermediate_data/30ppl_eval_v1_0728/<participant>/context_graphs/<comment_id>/
  ├─ context_graph.json   # nodes/edges JSON
  ├─ context_graph.mmd    # Mermaid graph
  └─ context_graph.md     # Markdown preview (```mermaid code block)
```

Aggregated per participant:

```
method/intermediate_data/30ppl_eval_v1_0728/<participant>/
  ├─ participant_context_graph.json
  ├─ participant_context_graph.mmd
  └─ participant_context_graph.md
```

The console shows per-participant progress bars (tqdm) and, if LLM is enabled, a one-time sample of the prompt/response (green/blue).

## Status

This methodology framework is currently under active development. Components may be incomplete or subject to change.


---
**Last Updated:** 2025-08-11
