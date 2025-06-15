from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
from api.utils import CLASS_NAMES
from api.config import MODEL_DIR, MODEL_PATTERN

saved_model_dir = Path(MODEL_DIR)
model_files = sorted(saved_model_dir.glob(MODEL_PATTERN))
latest_model_file = max(model_files, key=lambda p: int(p.stem.split("-v")[-1]))
model = load_model(latest_model_file)

def predict_local(image: np.ndarray) -> dict:
    image_batch = np.expand_dims(image, 0)
    prediction = model.predict(image_batch, verbose=0)
    class_id = int(np.argmax(prediction, axis=1)[0])
    predicted_class=CLASS_NAMES[class_id]
    confidence = float(np.max(prediction))
    return {
        "Class": predicted_class,
        "Confidence": confidence
    }
