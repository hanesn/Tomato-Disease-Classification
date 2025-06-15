import os
os.environ["USE_TF_SERVING"] = "True"

from fastapi.testclient import TestClient
from api.main import app
import numpy as np
from unittest.mock import patch
from PIL import Image
import io

client = TestClient(app)

@patch("api.model_tf_serving.requests.post")
def test_predict_tf_serving(mock_post):
    mock_post.return_value.json.return_value = {
        "predictions": [[0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9]]
    }

    dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    buf = io.BytesIO()
    dummy_image.save(buf, format='JPEG')
    buf.seek(0)

    res = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    print(data)
    assert isinstance(data["Confidence"], float)
    assert isinstance(data["Class"], str)
    assert data["Class"] == "healthy"
