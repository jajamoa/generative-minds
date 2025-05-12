# model_census/__init__.py

from .data_retriever import DataRetriever

def get_census_data(config_path: str):
    retriever = DataRetriever(config_path)
    return retriever.fetch_data()

__all__ = ["DataRetriever", "get_census_data"]
