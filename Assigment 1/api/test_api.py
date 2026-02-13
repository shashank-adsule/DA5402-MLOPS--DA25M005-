"""
API Smoke Tests - Phase C
Three smoke tests to verify API functionality
"""

import requests
import json
import time


API_BASE_URL = "http://127.0.0.1:5000"


def test_1_health_check():
    """
    Test 1: Health Check
    Verify that the API is running and model is loaded
    """
    print("\n" + "=" * 70)
    print("TEST 1: Health Check")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Verify structure
            assert 'status' in data, "Missing 'status' field"
            assert 'model_loaded' in data, "Missing 'model_loaded' field"
            assert data['status'] == 'healthy', "Status is not healthy"
            assert data['model_loaded'] == True, "Model is not loaded"
            
            print("\n✓ TEST 1 PASSED: API is healthy and model is loaded")
            return True
        else:
            print(f"\n✗ TEST 1 FAILED: Expected status 200, got {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ TEST 1 FAILED: Cannot connect to API")
        print("Make sure the API is running: python src/inference.py")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False


def test_2_single_prediction():
    """
    Test 2: Single Prediction
    Verify that the API can make a prediction on a single sample
    """
    print("\n" + "=" * 70)
    print("TEST 2: Single Prediction")
    print("=" * 70)
    
    # Sample input (no failure expected)
    sample_input = {
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50,
        "Type": "M"
    }
    
    print(f"Input: {json.dumps(sample_input, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=sample_input,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Verify structure
            assert 'prediction' in data, "Missing 'prediction' field"
            assert 'failure_probability' in data, "Missing 'failure_probability' field"
            assert 'prediction_class' in data, "Missing 'prediction_class' field"
            assert 'model_version' in data, "Missing 'model_version' field"
            
            # Verify types
            assert isinstance(data['prediction'], int), "prediction should be int"
            assert data['prediction'] in [0, 1], "prediction should be 0 or 1"
            assert isinstance(data['failure_probability'], float), "probability should be float"
            assert 0 <= data['failure_probability'] <= 1, "probability should be between 0 and 1"
            
            print("\n✓ TEST 2 PASSED: Prediction format is correct")
            return True
        else:
            print(f"\n✗ TEST 2 FAILED: Expected status 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False


def test_3_edge_case_prediction():
    """
    Test 3: Edge Case - High Tool Wear
    Test with conditions likely to trigger failure prediction
    """
    print("\n" + "=" * 70)
    print("TEST 3: Edge Case Prediction (High Tool Wear)")
    print("=" * 70)
    
    # Sample input with high tool wear (failure expected)
    edge_case_input = {
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 312.5,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 50.0,
        "Tool wear [min]": 230,  # Very high tool wear
        "Type": "L"
    }
    
    print(f"Input (High Tool Wear): {json.dumps(edge_case_input, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=edge_case_input,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Verify the model handles edge case
            assert 'prediction' in data, "Missing 'prediction' field"
            assert data['prediction'] in [0, 1], "Invalid prediction value"
            
            # For high tool wear, we expect higher failure probability
            if data['failure_probability'] > 0.5:
                print(f"\n✓ Model correctly identified high risk: {data['failure_probability']:.4f}")
            else:
                print(f"\n⚠ Note: Lower failure probability than expected: {data['failure_probability']:.4f}")
            
            print("\n✓ TEST 3 PASSED: API handles edge cases correctly")
            return True
        else:
            print(f"\n✗ TEST 3 FAILED: Expected status 200, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False


def test_4_batch_prediction():
    """
    Bonus Test: Batch Prediction
    Test batch prediction endpoint
    """
    print("\n" + "=" * 70)
    print("BONUS TEST: Batch Prediction")
    print("=" * 70)
    
    batch_input = {
        "samples": [
            {
                "Air temperature [K]": 298.5,
                "Process temperature [K]": 308.7,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 50,
                "Type": "M"
            },
            {
                "Air temperature [K]": 299.0,
                "Process temperature [K]": 309.2,
                "Rotational speed [rpm]": 1450,
                "Torque [Nm]": 42.5,
                "Tool wear [min]": 100,
                "Type": "L"
            },
            {
                "Air temperature [K]": 302.0,
                "Process temperature [K]": 312.5,
                "Rotational speed [rpm]": 1400,
                "Torque [Nm]": 50.0,
                "Tool wear [min]": 225,
                "Type": "H"
            }
        ]
    }
    
    print(f"Sending batch of {len(batch_input['samples'])} samples...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch-predict",
            json=batch_input,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Received {data['count']} predictions")
            
            for i, pred in enumerate(data['predictions'], 1):
                print(f"\nSample {i}:")
                print(f"  Prediction: {pred['prediction_class']}")
                print(f"  Probability: {pred['failure_probability']:.4f}")
            
            assert data['count'] == len(batch_input['samples']), "Wrong number of predictions"
            
            print("\n✓ BONUS TEST PASSED: Batch prediction works correctly")
            return True
        else:
            print(f"\n✗ BONUS TEST FAILED: Expected status 200, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n✗ BONUS TEST FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run all smoke tests"""
    print("=" * 70)
    print("MANUAL MLOPS - API SMOKE TESTS")
    print("=" * 70)
    print(f"\nTarget API: {API_BASE_URL}")
    print("\nStarting tests in 2 seconds...")
    time.sleep(2)
    
    results = []
    
    # Run required tests
    results.append(("Health Check", test_1_health_check()))
    time.sleep(1)
    
    results.append(("Single Prediction", test_2_single_prediction()))
    time.sleep(1)
    
    results.append(("Edge Case Prediction", test_3_edge_case_prediction()))
    time.sleep(1)
    
    # Run bonus test
    results.append(("Batch Prediction (Bonus)", test_4_batch_prediction()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)