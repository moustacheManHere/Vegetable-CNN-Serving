import requests
import json
import pytest
import numpy as np

@pytest.mark.xfail(reason="Expected failure due to incorrect input shape")
def test_small_model_incorrect_shape(url_small, get_image_large):
    headers = {"Content-Type": "application/json"}
    
    input_data = {"instances": [get_image_large[0][0]]}  # Remove one dimension
    json_data = json.dumps(input_data)
    response = requests.post(url_small, data=json_data, headers=headers)
    assert response.status_code == 200, "Expected failure due to incorrect input shape"
   

@pytest.mark.xfail(reason="Expected failure due to incorrect input shape")
def test_large_model_incorrect_shape(get_image_small, url_large):
    headers = {"Content-Type": "application/json"}
    
    input_data = {"instances": [get_image_small]}  # Remove one dimension
    json_data = json.dumps(input_data)

    response = requests.post(url_large, data=json_data, headers=headers)
    assert response.status_code == 200, "Expected failure due to incorrect input shape"


@pytest.mark.xfail(reason="Expected failure due to missing input values")
def test_large_model_missing_values(url_large):
    headers = {"Content-Type": "application/json"}

    # Remove 'instances' key from input data
    input_data = {}
    json_data = json.dumps(input_data)
    response = requests.post(url_large, data=json_data, headers=headers)
    assert response.status_code == 200, "Expected failure due to missing input values"

@pytest.mark.xfail(reason="Expected failure due to missing input values")
def test_small_model_missing_values(url_small):
    headers = {"Content-Type": "application/json"}

    # Remove 'instances' key from input data
    input_data = {}
    json_data = json.dumps(input_data)
    response = requests.post(url_small, data=json_data, headers=headers)
    assert response.status_code == 200, "Expected failure due to missing input values"
