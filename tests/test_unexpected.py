import pytest 
import requests
import json 
import numpy as np

def test_small_model_prediction(url_large, get_image_large):
    headers = {"Content-Type": "application/json"}
    input_data = {"instances": get_image_large}
    json_data = json.dumps(input_data)
    try:
        response = requests.post(url_large, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"

        predictions = np.array(prediction_results["predictions"])
        predicted_label = np.argmax(predictions)
        assert predicted_label == 12, f"Expected predicted label to be 3, but got {predicted_label}"

    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")

def test_large_model_prediction(url_small, get_image_small):
    headers = {"Content-Type": "application/json"}
    input_data = {"instances": get_image_small}
    json_data = json.dumps(input_data)
    try:
        response = requests.post(url_small, data=json_data, headers=headers)
        assert response.status_code == 200, f"Request failed with status code {response.status_code}"
        prediction_results = response.json()
        assert "predictions" in prediction_results, "Expected 'predictions' in response"

        predictions = np.array(prediction_results["predictions"])
        predicted_label = np.argmax(predictions)
        assert predicted_label == 12, f"Expected predicted label to be 3, but got {predicted_label}"
        
    except Exception as e:
        pytest.fail(f"Exception occurred: {e}")
