import pytest
import requests
import numpy as np
from PIL import Image


def preprocess(image_path,size):
    image = Image.open(image_path).convert("RGB")
    image = image.convert("L")
    image = image.resize((size, size))  # Resize the image to 31x31 pixels
    image_array = np.array(image)[:, :, np.newaxis] / 255.0  # Normalize pixel values to [0, 1]
    return [image_array.tolist()]

URL = "https://test-cnn.onrender.com/v1/models"

@pytest.fixture
def get_image_small():
    image_path = "/root/ca2-daaa2b01-2214618-jeyakumarsriram-dl/Vegetable Images/test/Bean/0001.jpg"
    return preprocess(image_path,31)

@pytest.fixture
def get_image_large():
    image_path = "/root/ca2-daaa2b01-2214618-jeyakumarsriram-dl/Vegetable Images/test/Bean/0001.jpg"
    return preprocess(image_path,128)

@pytest.fixture
def url_large():
    return URL+"/large:predict"

@pytest.fixture
def url_small():
    return URL+"/small:predict"