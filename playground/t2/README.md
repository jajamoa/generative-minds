# CBN Agent for Theory of Mind Evaluation

## Pipeline Overview
Complete CBN analysis pipeline in execution order:

1. **Extract Motifs** - `extract_motif_from_transcript.py` - Extract reasoning motifs from transcripts
   - `--input` - Input JSONL file path
   - `--output_dir` - Output directory for motifs
   - `--max_records N` - Limit records to process

2. **Build CBN** - `build_cbn_from_motifs.py` - Convert motifs into CBN format  
   - `--input_dir` - Input motifs directory
   - `--output_dir` - Output CBN directory
   - `--limit N` - Limit participants to process
   - `--workers N` - Parallel workers (default: 4)
   - `--no-parallel` - Disable parallel processing

3. **Evaluate CBN** - `evaluate_cbn.py` - Run belief attribution/update evaluation
   - `--benchmark` - Benchmark JSONL file
   - `--cbn` - CBN JSON file
   - `--limit N` - Limit questions to evaluate
   - `--debug` - Enable detailed output
   - `--model` - LLM model choice
   - `--temperature` - LLM randomness (0.0-1.0)

## Quick Usage

### Full Pipeline
```bash
./run_extract_motif.sh              # Step 1: Extract motifs
./run_build_cbn.sh                  # Step 2: Build CBN
./run_cbn_evaluation.sh             # Step 3: Evaluate
```

### Individual Steps
```bash
python extract_motif_from_transcript.py --max_records 5
python build_cbn_from_motifs.py --limit 5 --workers 6
python evaluate_cbn.py --benchmark ../../data/sample_belief_update_zoning.jsonl --limit 3 --debug
```
