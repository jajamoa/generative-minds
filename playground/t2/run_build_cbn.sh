#!/bin/bash
# Parallel CBN building (default: 4 workers)
# Use --workers N to adjust, --no-parallel for serial processing
python build_cbn_from_motifs.py --workers 6 "$@"
