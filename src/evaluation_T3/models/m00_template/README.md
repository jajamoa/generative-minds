# Template Model

This is a template for creating new opinion simulation models. It provides a minimal working implementation that satisfies the model interface requirements.

## Directory Structure

```
m00_template/
├── model.py          # Main model implementation
├── components/       # Model components (if needed)
├── data/            # Model-specific data (if needed)
└── README.md        # This file
```

## Usage

1. Copy this template directory to create a new model:
```bash
cp -r src/models/m00_template src/models/mNN_your_model_name
```

2. Rename the model class in `model.py`:
```python
class YourModelName(BaseModel):
    ...
```

3. Implement your model logic:
   - Add necessary components in the `components/` directory
   - Add model-specific data in the `data/` directory
   - Update the model implementation in `model.py`

4. Test your implementation:
```bash
python src/experiment/scripts/validate_model.py \
    --model-path models.mNN_your_model_name.model.YourModelName \
    --population 30
```

## Requirements

Your model implementation must:
1. Inherit from `BaseModel`
2. Implement the `simulate_opinions` method
3. Return data in the correct format
4. Handle the model configuration properly

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
            "comment": "This is a sample comment."
        }
    ],
    "key_themes": {
        "support": ["housing needs", "urban development"],
        "oppose": ["traffic concerns", "shadow impact"]
    }
}
```

## Field Specifications

### Agent Demographics
- `age`: "18-25" | "26-40" | "41-60" | "60+"
- `income_level`: "low_income" | "middle_income" | "high_income"
- `education_level`: "high_school" | "some_college" | "bachelor" | "postgraduate"
- `occupation`: "student" | "white_collar" | "service" | "retired" | "other"
- `gender`: "male" | "female" | "other"

### Location
- `lat`: Latitude (-90 to 90)
- `lng`: Longitude (-180 to 180)
- `cell_id`: String identifier of the nearest rezoning cell

### Opinion
- `opinion`: "support" | "oppose" | "neutral"
- `comment`: String explaining the agent's stance 