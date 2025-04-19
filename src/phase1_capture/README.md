# Phase 1: QA Extraction and Causal Graph Building

This module extracts question-answer pairs from interview transcripts and builds causal graphs from them.

## QA Extractor

The QA Extractor uses LLM to extract both explicit and implicit question-answer pairs from interview transcripts.

### Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your `.env` file has the required OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

#### Process all transcripts

To process all transcript files in the raw data directory:

```bash
python src/phase1_capture/run_qa_extraction.py
```

#### Process a single file

To process only a specific transcript file:

```bash
python src/phase1_capture/run_qa_extraction.py --single-file "Filename_en_subs.txt"
```

#### Custom directories

You can specify custom directories for raw and processed data:

```bash
python src/phase1_capture/run_qa_extraction.py --raw-dir "path/to/raw/data" --processed-dir "path/to/processed/data"
```

### QA Output

For each processed transcript, the following will be created in the processed directory:

1. A subdirectory named after the interview
2. A `qa_pairs.json` file containing the extracted QA pairs
3. Category-specific JSON files (e.g., `housing_experience_access_qa_pairs.json`)
4. A `metadata.json` file with processing information

## Causal Graph Builder

The Causal Graph Builder extracts causal relationships from QA pairs and builds directed acyclic graphs (DAGs).

### Usage

#### Build graphs for all interviews

To build causal graphs for all processed interviews:

```bash
python src/phase1_capture/run_causal_graph.py
```

#### Build graph for a single interview

To build a causal graph for a specific interview:

```bash
python src/phase1_capture/run_causal_graph.py --single-interview "Interview Name"
```

#### Custom directories

You can specify custom directories for input and output:

```bash
python src/phase1_capture/run_causal_graph.py --processed-dir "path/to/processed/data" --output-dir "path/to/output"
```

### Graph Output

For each processed interview, the following will be created:

1. A `causal_relationships.json` file containing extracted cause-effect relationships
2. A `causal_graph.json` file with the graph structure
3. A `causal_graph.mmd` file with Mermaid diagram code for visualization
4. A `causal_graph_metadata.json` file with graph information

## Visualization

You can visualize the Mermaid diagrams using:
- [Mermaid Live Editor](https://mermaid.live/)
- GitHub Markdown (just paste the .mmd file content into a markdown code block with mermaid tag)
- VS Code with Mermaid extension

## Flow

The complete phase 1 process includes:

1. **QA Extraction**: Process raw interview transcripts to extract QA pairs
2. **Causal Relationship Extraction**: Identify cause-effect relationships from QA pairs
3. **Graph Building**: Organize relationships into a directed graph
4. **DAG Formation**: Ensure the graph is acyclic by resolving any cycles
5. **Stance Connection**: Connect factors in the graph to policy stances
6. **Visualization**: Generate visualizations of the causal graph 