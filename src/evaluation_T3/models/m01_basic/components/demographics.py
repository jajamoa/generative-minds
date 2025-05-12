import json
import logging
import os
from pathlib import Path

class DemographicsSearchEngine:
    def __init__(self):
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / "data/demographics.json"
        self.data = json.loads(open(data_path).read())
        
    def search(self, region: str) -> dict:
        if region in self.data:
            return self.data[region]
        else:
            # log message: region not found and use default region instead
            logging.warning(f"Region not found: {region}. Using default region instead.")
            return self.data["default"]