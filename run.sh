############### TRIAL ###############

# python src/phase3_replicate/answer_agent.py --json data/samples/sample_1.json --question "What would happen to stance if we significantly increase building heights?" --num_samples 1000

# python src/phase3_replicate/answer_agent_continuous.py --json data/samples/sample_1.json --question "What would happen to stance if we significantly increase building heights?" --num_samples 10000

# python src/phase2_motif/motif_library.py --input data/samples --output data/samples/output

# python src/phase2_motif/graph_reconstruction.py --motif_library data/samples/output/motif_library.json --seed_node upzoning_stance --output_dir data/samples/output/reconstruction

# python -m experiment.run_experiment --protocol experiment/protocols/SF_transcript_survey_evaluation_5.11.yaml

# python src/evaluation_T3/models/m07_BN/motif_library/graph_reconstruction.py --motif_library src/evaluation_T3/models/m07_BN/motif_library/motif_library_output/motif_library.json --seed_node upzoning_stance --output_dir src/evaluation_T3/models/m07_BN/motif_library/graph_reconstruction


############### T2 ###############
# python src/evaluation/experiment/run_experiment.py --protocol src/evaluation/experiment/protocols/SF_BN_survey_evaluation.yaml


############### T3 ###############
# python src/evaluation_T3/models/m07_BN/motif_library/extract_demo_graphs.py --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/responses_5.11_with_geo.json --graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/causal_graph_responses_5.11_with_geo.json

# python src/evaluation_T3/models/m07_BN/motif_library/motif_library.py --agents-file src/evaluation_T3/models/m07_BN/motif_library/agents_with_graphs.json 

python src/evaluation_T3/experiment/run_experiment.py --protocol src/evaluation_T3/experiment/protocols/SF_BN_survey_evaluation.yaml

