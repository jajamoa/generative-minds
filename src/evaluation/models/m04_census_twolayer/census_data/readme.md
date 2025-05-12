# Census Data Module

## Overview
The **Census Data Module** (`model_census`) is a structured, configurable, and extensible pipeline designed to retrieve, process, and analyze census data efficiently. It follows a **configuration-driven** approach, making it easy to modify without hardcoding, ensuring maintainability and flexibility.

## Features
- **Configuration-Based**: Uses YAML/JSON for variable definitions, making it easy to update.
- **Modular Design**: Separates concerns into retrieval, processing, and distribution layers.
- **Efficient Data Handling**: Supports structured processing using `pandas` for data aggregation.
- **Scalable & Extendable**: Allows seamless integration of new datasets and processing logic.


## Directory Structure
```
model_census/
├── __init__.py       # High-level interface
├── config_api.yaml       # Configurable dataset parameters
├── data_retriever.py      # DataRetriever for API calls
├── data_processor.py      # DataProcessor for aggregation
```

## Usage
### 1. Retrieve and Process Census Data
#### Functional Process
see the test_retriever.py


#### processor
see the test_processor.py


## Configuration Example (`config.yaml`)
```yaml
# census_config.yaml

year: 2023
dataset: "acs5"

variables:
  population:
    - code: "B11004_001E"
      description: "Estimate!!Total:	Family Type by Presence and Age of Related Children Under 18 Years"
    - code: "B11004_004E"
      description: "Estimate!!Total:!!Married-couple family:!!With related children of the householder under 18 years:!!Under 6 years only	Family Type by Presence and Age of Related Children Under 18 Years"
  mobility:
    - code: "B08006_001E"
      description: "Estimate!!Total:	Sex of Workers by Means of Transportation to Work"
    - code: "B08006_034E"
      description: "Estimate!!Total:!!Male:!!Worked from home	Sex of Workers by Means of Transportation to Work"

Zipcode:
  - code: "94102"
    description: "San Francisco ZIP near downtown | https://www.unitedstateszipcodes.org/94102/"
  - code: "94103"
    description: "Another SF ZIP with mixed residential & commercial"
  - code: "94104"
    description: "Primarily a business area of SF"
```

## Census data Output Example
```json
{
  "94102": {
    "B11004_001E": "5132",
    "B11004_004E": "344",
    "B08006_001E": "18951",
    "B08006_034E": "2653"
  },
  "94103": {
    "B11004_001E": "5281",
    "B11004_004E": "471",
    "B08006_001E": "20567",
    "B08006_034E": "3715"
  },
  "94104": {
    "B11004_001E": "114",
    "B11004_004E": "0",
    "B08006_001E": "172",
    "B08006_034E": "15"
  }
}
```

## Census data distribution  Output Example
```
==== DataFrame ====
   zipcode table_code sub_code  value
0    94102     B11004     001E   5132
1    94102     B11004     004E    344
2    94102     B08006     001E  18951
3    94102     B08006     034E   2653
4    94103     B11004     001E   5281
5    94103     B11004     004E    471
6    94103     B08006     001E  20567
7    94103     B08006     034E   3715
8    94104     B11004     001E    114
9    94104     B11004     004E      0
10   94104     B08006     001E    172
11   94104     B08006     034E     15

==== DataFrame with Ratios ====
   zipcode table_code sub_code  value     ratio
0    94102     B08006     001E  18951  1.000000
1    94102     B08006     034E   2653  0.139993
2    94102     B11004     001E   5132  1.000000
3    94102     B11004     004E    344  0.067030
4    94103     B08006     001E  20567  1.000000
5    94103     B08006     034E   3715  0.180629
6    94103     B11004     001E   5281  1.000000
7    94103     B11004     004E    471  0.089188
8    94104     B08006     001E    172  1.000000
9    94104     B08006     034E     15  0.087209
10   94104     B11004     001E    114  1.000000
11   94104     B11004     004E      0  0.000000

==== Ratio Dictionary ====
{
  "94102": {
    "B08006_034E_ratio": 0.13999261252704343,
    "B11004_004E_ratio": 0.06703039750584568
  },
  "94103": {
    "B08006_034E_ratio": 0.1806291632226382,
    "B11004_004E_ratio": 0.08918765385343685
  },
  "94104": {
    "B08006_034E_ratio": 0.0872093023255814,
    "B11004_004E_ratio": 0.0
  }
}

==== Distribution JSON ====
{
  "94102": {
    "B08006_034E_ratio": 0.13999261252704343,
    "B11004_004E_ratio": 0.06703039750584568
  },
  "94103": {
    "B08006_034E_ratio": 0.1806291632226382,
    "B11004_004E_ratio": 0.08918765385343685
  },
  "94104": {
    "B08006_034E_ratio": 0.0872093023255814,
    "B11004_004E_ratio": 0.0
  }
}
```

## License
MIT License

