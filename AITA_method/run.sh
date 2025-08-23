#!/bin/bash
python motif_library/context_graph_builder.py --dataset data/30ppl_eval_v1_0728 --out intermediate_data/30ppl_eval_v1_0728/context_graphs --participant 1962Michael --llm_cfg context_graph/qwen_context_default.yaml

python motif_library/extract_motifs.py --participant_dir intermediate_data/30ppl_eval_v1_0728/context_graphs/1962Michael --output_dir intermediate_data/30ppl_eval_v1_0728/motif_library