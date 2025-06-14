import requests
import numpy as np
from api.utils import CLASS_NAMES
from api.config import TF_SERVING_PREDICT_URL

def predict_tf_serving(image: np.ndarray) -> dict:
    image_batch = np.expand_dims(image, 0).tolist()
    response = requests.post(TF_SERVING_PREDICT_URL, json={"instances": image_batch})
    prediction = np.array(response.json()["predictions"])
    class_id = int(np.argmax(prediction, axis=1)[0])
    predicted_class=CLASS_NAMES[class_id]
    confidence = float(np.max(prediction))
    return {
        "Class": predicted_class,
        "Confidence": confidence
    }
