# Intelligent Survey Dialogue System

An intelligent survey system based on LLMs and probabilistic programming, supporting automatic causal inference and dynamic follow-up questions.

## Project Structure

```
causal_survey/
├── dialogue/        # Dialogue management module
├── inference/       # Causal inference module
├── visualization/   # Visualization module
├── logs/           # Log storage
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Main Features

1. Interactive Q&A System
   - User text dialogue
   - LLM automatic response analysis
   - Dynamic follow-up questioning mechanism

2. Causal Inference
   - Bayesian updates based on Pyro
   - Dynamic causal graph construction

3. Visualization
   - NetworkX static graphs
   - D3.js interactive charts

## Installation and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
Create a .env file and set:
```
ANTHROPIC_API_KEY=your_api_key
```

3. Run the system:
```bash
python main.py
``` 