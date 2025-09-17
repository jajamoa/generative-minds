#!/bin/bash
# Run common motifs analysis on transcript data

echo "Common Motifs Analysis"
echo "====================="
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

# Run the analysis
echo "Running common motifs extraction and analysis..."
python extract_common_motifs.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo ""
    echo "Results are available in: results/common_motifs_analysis/"
    echo ""
    echo "To view the main report:"
    echo "  open results/common_motifs_analysis/README.md"
    echo ""
    echo "To view individual motif visualizations (Mermaid diagrams):"
    echo "  ls results/common_motifs_analysis/*.mmd"
else
    echo ""
    echo "Error: Analysis failed. Please check the error messages above."
    exit 1
fi
