#!/bin/bash
python evaluate_cbn.py --benchmark ../../data/sample_belief_attribution_zoning.jsonl --cbn sample_cbn.json "$@"
