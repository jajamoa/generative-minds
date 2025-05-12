from data_retriever import DataRetriever  # 确保你的类文件命名为 data_retriever.py
import json

# 测试 YAML 配置加载
config_path = "config_api.yaml"
retriever = DataRetriever(config_path)

# 检查变量加载
assert retriever.year == 2023, "Year should be 2023"
assert retriever.dataset == "acs5", "Dataset should be 'acs5'"


# 打印提取的数据
print("Extracted Variables:", retriever.var_codes)
print("Extracted ZIP Codes:", retriever.zip_list)

# 测试 API 请求（真实请求）
try:
    data = retriever.fetch_data()
    print("Fetched Data:", json.dumps(data, indent=2))
    
    # 测试保存 JSON 文件
    retriever.save_data_as_json(data, "test_census_data.json")
    print("Data successfully saved to test_census_data.json")
except Exception as e:
    print("Error fetching data:", e)
