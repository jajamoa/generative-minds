#!/usr/bin/env bash

DEBUG_TRAIN_SAMPLE=-1 # Maximum number of records to extract motifs from
DEBUG_EVAL_SAMPLE=2

# MODE: pairwise split
# For each scenario, BigTom contains both see and no settings (different observations), we randomly put one scenario in train and the other in test.

# Step 1: Split dataset
python dataset.py --input data/bigtom/bigtom.csv --output_dir data/bigtom --pairwise --seed 42

# Step 2: Extract motifs from each data sample in train_pair.csv
python extract_motif.py --input data/bigtom/train_pair.csv --output_dir results/pairwise/motifs_from_bigtom --max_records $DEBUG_TRAIN_SAMPLE

# Step 3: Extract common motifs
python extract_common_motifs.py --motifs_dir results/pairwise/motifs_from_bigtom --output_dir results/pairwise/common_motifs_analysis

# Step 4: Generate and evaluate CBN on test.csv
python evaluate_cbn.py \
  --benchmark data/bigtom/test_pair.csv \
  --motifs_dir results/pairwise/motifs_from_bigtom \
  --limit $DEBUG_EVAL_SAMPLE