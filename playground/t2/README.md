# CBN Agent for Theory of Mind Evaluation

## Files Overview
- `cbn_agent.py` - Main CBN agent implementing causal inference pipeline
- `evaluate_cbn.py` - Evaluation script for belief attribution/update tasks
- `sample_cbn.json` - Sample causal Bayesian network data (1M+ lines)
- `llm_utils.py` - LLM utilities for API calls
- `run_cbn_evaluation.sh` - Main evaluation script

## Quick Usage

### Basic Evaluation
```bash
./run_cbn_evaluation.sh                # Full dataset
./run_cbn_evaluation.sh --limit 5      # First 5 questions only
```

### Debug Mode (recommended for development)
```bash
python evaluate_cbn.py --benchmark ../../data/sample_belief_update_zoning.jsonl --limit 3 --debug
```

### Key Arguments
- `--limit N` - Test only first N questions (default: 5)
- `--debug` - Enable detailed output and sequential processing
- `--temperature` - Control LLM randomness (0.0-1.0)
- `--model` - Choose LLM model (qwen-plus, qwen-turbo, etc.)

## Development Workflow
1. Use `--limit 3 --debug` for quick testing
2. Increase limit gradually for larger tests
3. Remove `--debug` for production runs
4. Check CBN structure in `sample_cbn.json` (nodes/edges format)
