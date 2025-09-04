#!/bin/bash
python evaluate_cbn.py --benchmark ../../data/sample_belief_update_zoning.jsonl --cbn sample_cbn.json "$@"
