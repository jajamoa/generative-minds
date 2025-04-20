import requests
import json

BASE_URL = "http://localhost:5001"

def test_start_scenario():
    """Test the start scenario endpoint"""
    print("\n=== Testing start_scenario ===")
    url = f"{BASE_URL}/api/start_scenario"
    data = {
        "description": "I decided to buy an electric car because it is environmentally friendly and can save money in the long run"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status code: {response.status_code}")
        print("Response data:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.json().get("scenario_id")
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_answer_question(scenario_id):
    """Test the answer question endpoint"""
    print("\n=== Testing answer_question ===")
    url = f"{BASE_URL}/api/answer_question"
    data = {
        "answer": "The main reason is environmental protection, and the cost savings are a bonus",
        "question": "Can you elaborate on your decision?"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status code: {response.status_code}")
        print("Response data:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

def test_get_scenarios():
    """Test the get all scenarios endpoint"""
    print("\n=== Testing get_scenarios ===")
    url = f"{BASE_URL}/api/get_scenarios"
    
    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        print("Response data:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    print("Starting API tests...")
    
    # Test starting a new scenario
    scenario_id = test_start_scenario()
    
    if scenario_id:
        # Test answering a question
        test_answer_question(scenario_id)
    
    # Test getting all scenarios
    test_get_scenarios()
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 