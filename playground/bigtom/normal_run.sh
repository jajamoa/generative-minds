#!/usr/bin/env bash

DEBUG_SAMPLE=-1 # Maximum number of records to extract motifs from

# MODE: 80/20 random split

# Step 1: Split dataset
python dataset.py --input data/bigtom/bigtom.csv --output_dir data/bigtom --train_ratio 0.8 --seed 42

# Step 2: Extract motifs from each data sample in train.csv
python extract_motif.py --input data/bigtom/train.csv --output_dir results/normal/motifs_from_bigtom --max_records $DEBUG_SAMPLE

# Step 3: Extract common motifs
python extract_common_motifs.py --motifs_dir results/normal/motifs_from_bigtom --output_dir results/normal/common_motifs_analysis

# Step 4: Generate and evaluate CBN on test.csv
python evaluate_cbn.py \
  --benchmark data/bigtom/test.csv \
  --motifs_dir results/normal/motifs_from_bigtom \
  --limit $DEBUG_SAMPLE