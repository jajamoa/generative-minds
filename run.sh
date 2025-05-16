########################### HEALTHCARE ##############################

# python -m src.evaluation_T3.experiment.run_experiment --protocol src/evaluation_T3/experiment/protocols/BN_healthcare.yaml

# python -m src.evaluation_T3.models.m07_BN.motif_library.motif_library --raw-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/healthcare_causal_graph.json --clean-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/healthcare_causal_graph_clean.json --agent-demographic src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --output_path src/evaluation_T3/models/m07_BN/motif_library/healthcare_motif_library.json


# ########################### HOUSING ##############################

# python -m src.evaluation_T3.experiment.run_experiment --protocol src/evaluation_T3/experiment/protocols/BN_housing.yaml 

# python -m src.evaluation_T3.models.m07_BN.motif_library.motif_library --raw-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/housing_causal_graph.json --clean-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/housing_causal_graph_clean.json --agent-demographic src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --output_path src/evaluation_T3/models/m07_BN/motif_library/housing_motif_library.json



# ########################### SURVEILLANCE ##########################

# python -m src.evaluation_T3.experiment.run_experiment --protocol src/evaluation_T3/experiment/protocols/BN_surveillance.yaml

python -m src.evaluation_T3.models.m07_BN.motif_library.motif_library --raw-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/surveillance_causal_graph.json --clean-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/surveillance_causal_graph_clean.json --agent-demographic src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/agent_20ppl.json --output_path src/evaluation_T3/models/m07_BN/motif_library/surveillance_motif_library.json




# python src/evaluation/experiment/run_experiment.py --protocol src/evaluation/experiment/protocols/SF_BN_survey_evaluation.yaml

# python src/evaluation_T3/models/m07_BN/motif_library/motif_library.py --clean-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/causal_graph_clean.json --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/responses_5.11_with_geo.json --output src/evaluation_T3/models/m07_BN/motif_library/

# python -m src.evaluation_T3.models.m07_BN.motif_library.motif_library \
#     --clean-graphs src/evaluation_T3/experiment/eval/data/sf_prolific_survey/causal_graph_clean.json \
#     --responses src/evaluation_T3/experiment/eval/data/sf_prolific_survey/responses_5.11_with_geo.json \
#     --output src/evaluation_T3/models/m07_BN/motif_library/ 


# python src/evaluation_T3/models/m07_BN/motif_library/graph_reconstruction.py --seed_node support_for_upzoning --output_dir src/evaluation_T3/models/m07_BN/motif_library/graph_reconstruction

# python -m src.evaluation_T3.models.m07_BN.model \
#     --config src/evaluation_T3/models/m07_BN/config.yaml \
#     --output_dir src/evaluation_T3/models/m07_BN/output

# python src/evaluation_T3/experiment/run_experiment.py --protocol src/evaluation_T3/experiment/protocols/SF_BN_survey_evaluation.yaml

# python -m src.evaluation_T3.models.m07_BN.model \
#     --config src/evaluation_T3/models/m07_BN/config.yaml \
#     --output_dir src/evaluation_T3/models/m07_BN/output


# python src/phase2_motif/motif_library.py --input data/samples --output data/samples/output

# python src/phase2_motif/graph_reconstruction.py --motif_library data/samples/output/motif_library.json --seed_node upzoning_stance --output_dir data/samples/output/reconstruction







# python src/evaluation_T3/models/m07_BN/motif_library/motif_library.py --input data/samples --output data/samples/output

# python src/evaluation_T3/models/m07_BN/motif_library/graph_reconstruction.py --motif_library data/samples/output/motif_library.json --seed_node upzoning_stance --output_dir data/samples/output/reconstruction