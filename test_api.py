import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_root():
    """Test the welcome message"""
    response = requests.get(f"{BASE_URL}/")
    print("=== ROOT ENDPOINT ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_health():
    """Test if API is healthy"""
    response = requests.get(f"{BASE_URL}/health")
    print("=== HEALTH CHECK ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction():
    """Test prediction with normal motor"""
    data = {
        "air_temp": 300,
        "process_temp": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("=== PREDICTION (Normal Motor) ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction_failure():
    """Test prediction with failing motor (high wear, high torque)"""
    data = {
        "air_temp": 305,
        "process_temp": 313,
        "rotational_speed": 1200,  # Low speed + high torque = stress
        "torque": 65,              # High torque
        "tool_wear": 240           # Very worn tool
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("=== PREDICTION (Stressed Motor) ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_invalid_data():
    """Test with bad data (should fail validation)"""
    data = {
        "air_temp": "not_a_number",  # Wrong type
        "process_temp": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("=== INVALID DATA TEST ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("Testing Motor Failure Prediction API\n")
    
    try:
        test_root()
        test_health()
        test_prediction()
        test_prediction_failure()
        test_invalid_data()
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: API is not running!")
        print("Start it first with: python -m uvicorn main:app --reload --port 8001")