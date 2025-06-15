import numpy as np
from api.utils import read_file_as_image, get_latest_model_path, CLASS_NAMES
from PIL import Image
import tempfile
from pathlib import Path
import os

def test_read_file_as_image():
    image = Image.new("RGB", (256, 256), color="red")
    # Create a temporary file and save image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    with open(temp_path, "rb") as f:
        image_bytes = f.read()

    os.remove(temp_path)

    result = read_file_as_image(image_bytes)
    assert isinstance(result, np.ndarray)
    assert result.shape == (256, 256, 3)

def test_get_latest_model_path():
    # Simulate model files
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "tomato_model-v1.keras").touch()
        (base / "tomato_model-v2.keras").touch()
        (base / "tomato_model-v3.keras").touch()
        (base / "tomato_model-v6.keras").touch()

        latest = get_latest_model_path(saved_model_dir=tmpdir)
        assert latest.name == "tomato_model-v6.keras"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = get_latest_model_path(saved_model_dir=tmpdir)
        assert latest is None    

def test_class_names():
    assert "Bacterial_spot" in CLASS_NAMES
    assert "Early_blight" in CLASS_NAMES
    assert "Late_blight" in CLASS_NAMES
    assert "Leaf_Mold" in CLASS_NAMES
    assert "Septoria_leaf_spot" in CLASS_NAMES
    assert "Spider_mites_Two_spotted_spider_mite" in CLASS_NAMES
    assert "Target_Spot" in CLASS_NAMES
    assert "YellowLeaf__Curl_Virus" in CLASS_NAMES
    assert "mosaic_virus" in CLASS_NAMES
    assert "healthy" in CLASS_NAMES
