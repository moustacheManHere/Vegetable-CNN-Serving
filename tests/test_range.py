import requests
import json
import pytest
import numpy as np

@pytest.mark.xfail(reason="Expected failure due to large input values")
def test_small_model_large_values(url_small):
    headers = {"Content-Type": "application/json"}

    # Generate input data with extremely large values
    input_data = {"instances": [np.ones((31, 31), dtype=np.float32) * 1e30]}
    json_data = json.dumps(input_data)

    try:
        response = requests.post(url_small, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"
        predictions = np.array(prediction_results["predictions"])
        assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

@pytest.mark.xfail(reason="Expected failure due to large input values")
def test_large_model_large_values(url_large):
    headers = {"Content-Type": "application/json"}

    # Generate input data with extremely large values
    input_data = {"instances": [np.ones((31, 31), dtype=np.float32) * 1e30]}
    json_data = json.dumps(input_data)

    try:
        response = requests.post(url_large, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"
        predictions = np.array(prediction_results["predictions"])
        assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

@pytest.mark.xfail(reason="Expected failure due to negative input values")
def test_large_model_very_negative_values(url_large):
    headers = {"Content-Type": "application/json"}

    input_data = {"instances": [np.ones((128, 128), dtype=np.float32) * -1e30]}
    json_data = json.dumps(input_data)

    try:
        response = requests.post(url_large, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"

        predictions = np.array(prediction_results["predictions"])
        assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

@pytest.mark.xfail(reason="Expected failure due to negative input values")
def test_small_model_very_negative_values(url_small):
    headers = {"Content-Type": "application/json"}

    input_data = {"instances": [np.ones((128, 128), dtype=np.float32) * -1e30]}
    json_data = json.dumps(input_data)

    try:
        response = requests.post(url_small, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"

        predictions = np.array(prediction_results["predictions"])
        assert np.all(np.isfinite(predictions)), "Predictions contain non-finite values"
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")