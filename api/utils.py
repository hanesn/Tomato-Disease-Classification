import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

CLASS_NAMES = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
               'Septoria_leaf_spot', 'Spider_mites_Two_spotted_spider_mite',
               'Target_Spot', 'YellowLeaf__Curl_Virus', 'mosaic_virus', 'healthy']

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)

def get_latest_model_path(saved_model_dir="saved_models", pattern="tomato_model-v*.keras"):
    saved_model_dir = Path(saved_model_dir)
    model_files = sorted(saved_model_dir.glob(pattern))
    if not model_files:
        return None
    return max(model_files, key=lambda p: int(p.stem.split("-v")[-1]))