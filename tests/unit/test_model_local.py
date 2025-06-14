import numpy as np
from api import model_local

class DummyModel:
    def predict(self, batch, verbose=0):
        return np.array([[0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5]])  # index 9 (healthy)

def test_predict_local():
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Swap in dummy model
    model_local.model = DummyModel()

    result = model_local.predict_local(dummy_image)

    assert isinstance(result, dict)
    assert result["Class"] == model_local.CLASS_NAMES[9]  # "healthy"
    assert 0.0 <= result["Confidence"] <= 1.0
