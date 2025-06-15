import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_ping():
    res = client.get("/")
    assert res.status_code == 200
    assert res.text == '"hello"'

def test_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_path" in data
    assert "num_classes" in data
    assert "use_tf_serving" in data

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] in ["ok", "fail"]
    assert "mode" in data

@patch("api.main.predict_model")
def test_predict(mock_predict):
    # Mock prediction response
    mock_predict.return_value = {
        "Class": "Healthy",
        "Confidence": 0.987
    }

    # Create a dummy image for testing
    dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    image_bytes = io.BytesIO()
    dummy_image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    res = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")}
    )

    assert res.status_code == 200
    data = res.json()
    assert "Class" in data
    assert "Confidence" in data
    assert isinstance(data["Confidence"], float)
