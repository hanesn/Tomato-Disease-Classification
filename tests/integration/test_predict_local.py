import pytest
pytestmark = pytest.mark.local

import os
import requests

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

def test_ping():
    res = requests.get(f"{SERVER_URL}/")
    assert res.status_code == 200
    assert res.text == '"hello"'

def test_health():
    res = requests.get(f"{SERVER_URL}/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ["ok", "fail"]
    assert "mode" in data

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
