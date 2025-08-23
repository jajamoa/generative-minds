#!/bin/bash

python3 merge_concepts.py --graph ../intermediate_data/30ppl_eval_v1_0728/context_graphs/1962Michael/participant_context_graph.json --out ../intermediate_data/30ppl_eval_v1_0728/context_graphs/1962Michael/participant_context_graph.merged.json --threshold 0.5 --model all-MiniLM-L6-v2