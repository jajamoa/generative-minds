import pandas as pd
from typing import Dict, Any
import json

class DataProcessor:
    """
    Used to process Census data and produce joint distribution 
    or other statistical indicators.
    """

    def __init__(self, data: Dict[str, Dict[str, str]]):
        """
        :param data: 
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
            ...
          }
        """
        self.raw_data = data

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the raw dictionary data into a DataFrame in the form of a long table for subsequent analysis(IPF).
        Example of the returned DataFrame column structure:
              zipcode    table_code   sub_code   value
            0   94102     B11004       001E       5132
            1   94102     B11004       004E        344
            2   94102     B08006       001E      18951
            3   94102     B08006       034E       2653
            ...
        """
        records = []

        for zc, var_dict in self.raw_data.items():
            for var_code, val_str in var_dict.items():
                # var_code like "B11004_001E"；divide into table_code="B11004", sub_code="001E"
                parts = var_code.split("_", maxsplit=1)
                if len(parts) == 2:
                    table_code, sub_code = parts
                else:
                    
                    table_code = var_code
                    sub_code = ""

                # string to int
                try:
                    value = int(val_str)
                except ValueError:
                    # if result is null or special tokens
                    value = 0

                records.append({
                    "zipcode": zc,
                    "table_code": table_code,
                    "sub_code": sub_code,
                    "value": value
                })

        df = pd.DataFrame(records)
        return df

    def compute_ratios(self) -> pd.DataFrame:
        """
        Example: group by (zipcode, table_code), assume *_001E is the total and the other *_xxxE are the subsets, ratio = (value of subset) / (value of table).
        ratio = (value of subset) / (value of total).
        
        Returns a long DataFrame with one column 'ratio', example:
             zipcode  table_code  sub_code  value    ratio
           0   94102     B11004      001E   5132  1.000000
           1   94102     B11004      004E    344  0.067024
           2   94102     B08006      001E  18951  1.000000
           3   94102     B08006      034E   2653  0.139943
           ...
        """
        df = self.to_dataframe()

        # groupby zipcode + table_code
        groups = df.groupby(["zipcode", "table_code"], as_index=False)

        # For each group, we find the row with sub_code = “001E”, which is the total.
        # Then the other lines / total => ratio
        def calc_ratio(group: pd.DataFrame) -> pd.DataFrame:
            # locate total
            total_row = group.loc[group["sub_code"] == "001E"]
            if total_row.empty:
                # If you don't find 001E, assume that the total number = 1 or just skip it
                group["ratio"] = None
                return group

            total_value = total_row["value"].values[0]
            if total_value == 0:
                # avoid divide 0
                group["ratio"] = None
            else:
                group["ratio"] = group["value"] / total_value
            return group

        df_with_ratio = groups.apply(calc_ratio)
        df_with_ratio = df_with_ratio.reset_index(drop=True)

        return df_with_ratio

    def get_ratio_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a JSON/dict structure for easy downstream use.
        Example.
        {
          "94102": {
            "B11004_004E_ratio": 0.067,
            "B08006_034E_ratio": 0.14,
            ...
          },
          "94103": { ... },
          ...
        }
        """
        df_with_ratio = self.compute_ratios()

        # case for 001E
        df_subset = df_with_ratio[df_with_ratio["sub_code"] != "001E"].copy()

        # new column  "B11004_004E_ratio"
        df_subset["ratio_code"] = df_subset["table_code"] + "_" + df_subset["sub_code"] + "_ratio"

        # group the zipcode
        result = {}
        for zc, group in df_subset.groupby("zipcode"):
            # group: zipcode
            ratios = {}
            for _, row in group.iterrows():
                ratio_key = row["ratio_code"]
                ratio_value = row["ratio"]
                ratios[ratio_key] = ratio_value
            result[zc] = ratios

        return result

    def get_distribution_json(self) -> str:
        """
        return json string
        """
        ratio_dict = self.get_ratio_dict()
        return json.dumps(ratio_dict, indent=2) 