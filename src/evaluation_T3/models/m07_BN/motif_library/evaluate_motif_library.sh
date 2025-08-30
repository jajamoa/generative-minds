#!/bin/bash

# Run motif library evaluation
echo "Evaluating motif library quality..."
python evaluation/motif_evaluator.py \
    --lib motif_library_output/motif_library.json \
    --orig agents_with_graphs.json
