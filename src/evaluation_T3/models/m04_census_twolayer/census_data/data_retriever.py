import os
from urllib.parse import quote, urlencode
import yaml
import json
import math
import requests
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests_cache
    HAS_REQUESTS_CACHE = True
except ImportError:
    HAS_REQUESTS_CACHE = False


class DataRetriever:
    """
    Responsible for.
      1. loading and storing the YAML configuration
      2. Build Census API requests based on the configuration
      3. initiating requests and returning structured data
      4. (Optional) Concurrency, caching, result storage
    """

    CHUNK_SIZE = 100  # Splitting multiple parallel requests when zip_list exceeds 100

    def __init__(self, config_path: str, enable_cache: bool = False):
        """
        :param config_path: YAML config path
        :param enable_cache: Whether to enable local or memory caching
        """
        self.config_path = config_path
        self.enable_cache = enable_cache
        self._load_config()
        
        # 如果需要缓存且已安装 requests-cache，则初始化
        if self.enable_cache and HAS_REQUESTS_CACHE:
            # expire_after 可按需指定，如 3600 秒(1小时)等
            requests_cache.install_cache('census_cache', expire_after=3600)

    def _load_config(self):
        """load the yaml."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Read yaml year, dataset, variables, Zipcode
        self.year = self.config.get("year", 2023)
        self.dataset = self.config.get("dataset", "acs5")
        self.variables = self.config.get("variables", {})
        self.Zipcode = self.config.get("Zipcode", [])
        
        # Variable => self.var_codes (形如 ["B11004_004E", "B08012_003E", ...])
        self.var_codes = []
        for category, var_list in self.variables.items():
            for var_item in var_list:
                self.var_codes.append(var_item["code"])

        # ZIP  => self.zip_list (形如 ["94102", "94103", "94104", ...])
        self.zip_list = [z["code"] for z in self.Zipcode]

    def _fetch_data_for_chunk(self, zip_chunk: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Make a single request for a single ZIP batch (e.g. <=100 ZIPs) and return the result.
        Return form: { “94102”: { “B11004_004E”: “344”, ...}, “94103”: {...} }, “94103”: {...} , ... }
        """
        base_url = f"https://api.census.gov/data/{self.year}/acs/{self.dataset}"

        
        # concat variables => "NAME,B11004_004E,B08012_003E,..."
        var_str = "NAME," + ",".join(self.var_codes)

        # concat ZIP => "94102,94103,94104"
        zip_str = ",".join(zip_chunk)

        for_clause = f"zip code tabulation area:{zip_str}"
        encoded_for_clause = quote(for_clause, safe=":,")
      
         # full URL
        full_url = f"{base_url}?get={var_str}&for={encoded_for_clause}"
        
        print(f"Full API Request URL: {full_url}")

        response = requests.get(full_url)
        if response.status_code != 200:
            raise ConnectionError(f"Request failed, status code: {response.status_code}, text: {response.text}")

        data = response.json()
        header = data[0]  # e.g. ["NAME", "B11004_004E", ... ,"zip code tabulation area"]
        rows = data[1:]

        # 找到 ZIP code 在 header 中的 index
        try:
            zip_index = header.index("zip code tabulation area")
        except ValueError:
            raise ValueError("API response does not contain 'zip code tabulation area' in the header.")

        # 找到每个 var_code 在 header 中对应的下标
        var_indices = {}
        for code in self.var_codes:
            if code in header:
                var_indices[code] = header.index(code)
            else:
                raise ValueError(f"Variable code '{code}' not found in API response header: {header}")

        chunk_result = {}
        # 解析行数据
        for row in rows:
            zc = row[zip_index]  # "94102"
            var_values = {}
            for code, idx in var_indices.items():
                var_values[code] = row[idx]
            chunk_result[zc] = var_values

        return chunk_result

    def fetch_data(self) -> Dict[str, Dict[str, str]]:
        """
        Core method.
          - If the number of ZIPs <= CHUNK_SIZE, then crawl in one request.
          - Otherwise split into multiple batches and crawl in parallel
          - Finally, merge the results and return
        """
        if not self.var_codes:
            raise ValueError("No variable codes found in config.")
        if not self.zip_list:
            raise ValueError("No ZIP codes found in config.")

        # 如果 zip 数量不多，直接一次请求
        if len(self.zip_list) <= self.CHUNK_SIZE:
            return self._fetch_data_for_chunk(self.zip_list)

        # 否则，拆分并发请求
        result = {}
        # 简易的分批逻辑
        def chunkify(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i: i + size]

        # 并行
        with ThreadPoolExecutor() as executor:
            futures = []
            for zip_chunk in chunkify(self.zip_list, self.CHUNK_SIZE):
                futures.append(executor.submit(self._fetch_data_for_chunk, zip_chunk))

            for future in as_completed(futures):
                chunk_result = future.result()
                # 合并字典
                for zc, val_dict in chunk_result.items():
                    result[zc] = val_dict

        return result

    def save_data_as_json(self, data: Dict[str, Dict[str, str]], filename: str = "census_result.json"):
        """
        save the result in json file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Data has been saved to {filename}")
