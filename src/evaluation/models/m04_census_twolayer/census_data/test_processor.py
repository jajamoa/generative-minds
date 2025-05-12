import json
import pandas as pd
from data_processor import DataProcessor  # 确保你的 DataProcessor 在 data_processor.py 中

# 读取 JSON 数据
json_file_path = "test_census_data.json"  # 你的 JSON 文件路径

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 创建 DataProcessor 实例
processor = DataProcessor(data)

# 测试 1: 转换成 DataFrame
df = processor.to_dataframe()
print("==== DataFrame ====")
print(df)

# 测试 2: 计算 ratio
df_ratios = processor.compute_ratios()
print("\n==== DataFrame with Ratios ====")
print(df_ratios)

# 测试 3: 转换成字典
ratio_dict = processor.get_ratio_dict()
print("\n==== Ratio Dictionary ====")
print(json.dumps(ratio_dict, indent=2))

# 测试 4: 转换成 JSON 字符串
distribution_json = processor.get_distribution_json()
print("\n==== Distribution JSON ====")
print(distribution_json)
