## BigToM – Quick Start

### Run
- One command (recommended):
```bash
bash run.sh
```

- Manual steps:
1) Split dataset (choose one)
```bash
# random 80/20 (default)
python dataset.py --input data/bigtom/bigtom.csv --output_dir data/splits --train_ratio 0.8 --seed 42
# pairwise see/no (per-row put one scenario in train, the other in test)
python dataset.py --input data/bigtom/bigtom.csv --output_dir data/splits --pairwise --seed 42
```
2) Extract motifs from train
```bash
python extract_motif.py --input data/bigtom/train.csv --output_dir results/normal/motifs_from_bigtom
```
3) Evaluate on test (dynamic CBN from motifs)
```bash
python evaluate_cbn.py --benchmark data/bigtom/test.csv --motifs_dir results/normal/motifs_from_bigtom --limit 5
```

Outputs
- Motifs: `results/normal/motifs_from_bigtom/<id>/<id>.{json,md,mmd}`
- Dynamic CBN per row: `results/normal/eval/dynamic_cbns/bigtom_rowXXXX_cbn.{json,mmd,md}`
- Eval metrics: `results/normal/eval/cbn_results_<model>.json`

### Logic
- Motifs: Qwen reads each (Story, Observation) and returns motifs (deduped) per row.
- Build CBN: use Story-only text to retrieve top-K similar motifs (sentence-transformers), then merge to a graph.
- Inference: Qwen translates the question to a do() on a variable; run simple propagation.
- Answer: score options by similarity to node labels weighted by current beliefs.
  <span style="color: red;">Note: this may not be the best way to score options, but it's a simple way to get a score.</span>

### Dataset splits
- Random 80/20: simple baseline for overall robustness.
- Pairwise see/no: same story split across train/test → tests generalization under different observations.


