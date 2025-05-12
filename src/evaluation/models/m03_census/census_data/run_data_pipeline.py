import argparse
import json
from data_retriever import DataRetriever
from data_processor import DataProcessor

class DataPipeline:
    def __init__(self, config_path: str, output_json: str):
        self.config_path = config_path
        self.output_json = output_json

    def fetch_and_save(self):
        # 数据获取与保存
        retriever = DataRetriever(self.config_path)
        data = retriever.fetch_data()
        retriever.save_data_as_json(data, self.output_json)
        print(f"Data successfully saved to {self.output_json}")
        return data

    def process_data(self, data):
        # 数据处理
        processor = DataProcessor(data)
        df = processor.to_dataframe()
        df_ratios = processor.compute_ratios()
        ratio_dict = processor.get_ratio_dict()
        distribution_json = processor.get_distribution_json()
        return df, df_ratios, ratio_dict, distribution_json

    def run(self):
        # 整个流程运行
        data = self.fetch_and_save()
        # 如果需要从文件重新加载也可以这样做：
        # with open(self.output_json, "r", encoding="utf-8") as f:
        #     data = json.load(f)
        df, df_ratios, ratio_dict, distribution_json = self.process_data(data)
        print("==== DataFrame ====")
        print(df)
        print("\n==== DataFrame with Ratios ====")
        print(df_ratios)
        print("\n==== Ratio Dictionary ====")
        print(json.dumps(ratio_dict, indent=2))
        print("\n==== Distribution JSON ====")
        print(distribution_json)
        return {
            "dataframe": df,
            "dataframe_ratios": df_ratios,
            "ratio_dict": ratio_dict,
            "distribution_json": distribution_json
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run data pipeline with specified YAML config and JSON output file."
    )
    parser.add_argument('--config', type=str, default='config_api.yaml',
                        help="Path to the YAML configuration file.")
    parser.add_argument('--output', type=str, default='test_census_data.json',
                        help="Output JSON file name.")
    args = parser.parse_args()

    pipeline = DataPipeline(args.config, args.output)
    pipeline.run()
