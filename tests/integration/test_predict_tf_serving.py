import pytest
pytestmark = pytest.mark.tf_serving

import os
import requests

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
TF_SERVING_HEALTH_URL = os.getenv("TF_SERVING_HEALTH_URL", "http://localhost:8501/v1/models/tomatoes_model")

def test_ping():
    res = requests.get(f"{SERVER_URL}/")
    assert res.status_code == 200
    assert res.text == '"hello"'

def test_health():
    res = requests.get(f"{TF_SERVING_HEALTH_URL}")
    assert res.status_code == 200
    data = res.json()
    assert "model_version_status" in data
    assert "status" in data["model_version_status"][0]
    assert "state" in data["model_version_status"][0]
    assert data["model_version_status"][0]["state"] in ["AVAILABLE"]

def test_model_info():
    res = requests.get(f"{SERVER_URL}/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_path" in data
    assert "num_classes" in data
    assert "use_tf_serving" in data

def test_predict_real_image():
    images_path=os.path.join('tests','integration','test_images')
    images=os.listdir(images_path)
    for image in images:
        image_path = os.path.join(images_path,image)
        assert os.path.exists(image_path), "Test image not found"
        with open(image_path, "rb") as img:
            files = {"file": (image, img, "image/jpeg")}
            res = requests.post(f"{SERVER_URL}/predict", files=files)
        assert res.status_code == 200
        data = res.json()
        assert "Class" in data
        assert "Confidence" in data
        assert isinstance(data["Confidence"], float)
