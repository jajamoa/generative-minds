import requests
import json

BASE_URL = "http://localhost:5001"

def test_start_scenario():
    """测试开始新场景的端点"""
    print("\n=== 测试 start_scenario ===")
    url = f"{BASE_URL}/api/start_scenario"
    data = {
        "description": "I decided to buy an electric car because it is environmentally friendly and can save money in the long run"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print("响应数据:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.json().get("scenario_id")
    except Exception as e:
        print(f"错误: {e}")
        return None

def test_answer_question(scenario_id):
    """测试回答问题的端点"""
    print("\n=== 测试 answer_question ===")
    url = f"{BASE_URL}/api/answer_question"
    data = {
        "answer": "The main reason is environmental protection, and the cost savings are a bonus",
        "question": "Can you elaborate on your decision?"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print("响应数据:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"错误: {e}")

def test_get_scenarios():
    """测试获取所有场景的端点"""
    print("\n=== 测试 get_scenarios ===")
    url = f"{BASE_URL}/api/get_scenarios"
    
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        print("响应数据:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"错误: {e}")

def main():
    """运行所有测试"""
    print("开始 API 测试...")
    
    # 测试开始新场景
    scenario_id = test_start_scenario()
    
    if scenario_id:
        # 测试回答问题
        test_answer_question(scenario_id)
    
    # 测试获取所有场景
    test_get_scenarios()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 