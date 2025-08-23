#!/bin/bash

python3 concept_quality_evaluator.py --graph ../intermediate_data/30ppl_eval_v1_0728/context_graphs/1962Michael/participant_context_graph.json

python3 concept_quality_evaluator.py --graph ../intermediate_data/30ppl_eval_v1_0728/context_graphs/1962Michael/participant_context_graph.merged.json
