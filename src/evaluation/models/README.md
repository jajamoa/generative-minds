# Opinion Simulation Models

Collection of agent-based models for simulating public opinions on urban development proposals.

## Models

- `m00_template/`: Template for creating new models
- `m01_basic/`: Basic simulation model with random opinions
- `m02_stupid/`: LLM-powered agent model with demographic-based opinions

## Model Interface

All models implement the `BaseModel` interface:

```python
async def simulate_opinions(
    self,
    region: str,
    proposal: Dict[str, Any]
) -> Dict[str, Any]
```

## Output Format

```json
{
    "summary": {
        "support": 65,
        "neutral": 15,
        "oppose": 20
    },
    "comments": [
        {
            "id": 1,
            "agent": {
                "age": "26-40",
                "income_level": "middle_income",
                "education_level": "bachelor",
                "occupation": "white_collar",
                "gender": "female"
            },
            "location": {
                "lat": 37.7371,
                "lng": -122.4887
            },
            "cell_id": "10_15",
            "opinion": "support",
            "comment": "..."
        }
    ],
    "key_themes": {
        "support": ["housing needs", "urban development"],
        "oppose": ["traffic concerns", "shadow impact"]
    }
}
```

## Creating New Models

1. Copy the template directory:
```bash
cp -r src/models/m00_template src/models/mNN_your_model_name
```

2. Implement the `simulate_opinions` method in your model class

3. Update the model registry in `run_experiment.py` 