import requests
from PIL import Image
import json
import numpy as np

tfs_endpoint = "http://cnn_models:8501/v1/models/small:predict"
tfs_endpoint2 = "http://cnn_models:8501/v1/models/large:predict"

image_path = "/root/ca2-daaa2b01-2214618-jeyakumarsriram-dl/DL/Vegetable Images/test/Bean/0001.jpg"
def preprocess(image_path,size):
    image = Image.open(image_path).convert("RGB")
    image = image.convert("L")
    image = image.resize((size, size))  # Resize the image to 31x31 pixels
    image_array = np.array(image)[:, :, np.newaxis] / 255.0  # Normalize pixel values to [0, 1]
    return [image_array.tolist()]

input_data = {"instances": preprocess(image_path,31)}
input_data2 = {"instances": preprocess(image_path,128)}

json_data = json.dumps(input_data)
json_data2 = json.dumps(input_data2)
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(tfs_endpoint, data=json_data, headers=headers)
    response2 = requests.post(tfs_endpoint2, data=json_data2, headers=headers)
except:
    print("Error while connecting...")

if response.status_code == 200:
    prediction_results = response.json()
    print("Prediction Results:", prediction_results)
else:
    print("Error:", response.status_code, response.text)

if response2.status_code == 200:
    prediction_results = response2.json()
    print("Prediction Results:", prediction_results)
else:
    print("Error:", response2.status_code, response2.text)
