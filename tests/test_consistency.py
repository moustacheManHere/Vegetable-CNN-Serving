import requests
import json
import pytest
import numpy as np

def test_small_model_consistency(url_small, get_image_small):
    headers = {"Content-Type": "application/json"}
    input_data = {"instances": get_image_small}
    json_data = json.dumps(input_data)
    try:
        response1 = requests.post(url_small, data=json_data, headers=headers)
        assert response1.status_code == 200, f"Request failed with status code {response1.status_code}"
        prediction_results1 = response1.json()
        assert "predictions" in prediction_results1, "Expected 'predictions' in response"

        response2 = requests.post(url_small, data=json_data, headers=headers)
        assert response2.status_code == 200, f"Request failed with status code {response2.status_code}"
        prediction_results2 = response2.json()
        assert "predictions" in prediction_results2, "Expected 'predictions' in response"

        # Check if predictions are consistent
        predictions1 = np.array(prediction_results1["predictions"])
        predictions2 = np.array(prediction_results2["predictions"])
        assert np.array_equal(predictions1, predictions2), "Predictions are not consistent"

    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

def test_large_model_consistency(url_large, get_image_large):
    headers = {"Content-Type": "application/json"}
    input_data = {"instances": get_image_large}
    json_data = json.dumps(input_data)
    try:
        response1 = requests.post(url_large, data=json_data, headers=headers)
        assert response1.status_code == 200, f"Request failed with status code {response1.status_code}"
        prediction_results1 = response1.json()
        assert "predictions" in prediction_results1, "Expected 'predictions' in response"

        response2 = requests.post(url_large, data=json_data, headers=headers)
        assert response2.status_code == 200, f"Request failed with status code {response2.status_code}"
        prediction_results2 = response2.json()
        assert "predictions" in prediction_results2, "Expected 'predictions' in response"

        # Check if predictions are consistent
        predictions1 = np.array(prediction_results1["predictions"])
        predictions2 = np.array(prediction_results2["predictions"])
        assert np.array_equal(predictions1, predictions2), "Predictions are not consistent"

    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")
