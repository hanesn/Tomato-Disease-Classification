from fastapi import FastAPI,UploadFile,HTTPException
from api.utils import *
from api.config import USE_TF_SERVING,LOG_FILE,TF_SERVING_HEALTH_URL,LOG_LEVEL
from pydantic import BaseModel
import logging
import uvicorn
import time
import requests
import os
from pathlib import Path
import sys
from contextlib import asynccontextmanager

if USE_TF_SERVING:
    from api.model_tf_serving import predict_tf_serving as predict_model
else:
    from api.model_local import predict_local as predict_model

# Resolve log path relative to this file
log_file_path = Path(__file__).parent / LOG_FILE

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only create logs directory if not running tests
    if "pytest" not in sys.modules:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    yield  # Control passes to app here

app = FastAPI(lifespan=lifespan)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"api/{LOG_FILE}"),
        logging.StreamHandler()  # also keeps logging to terminal
    ]
)

latest_model_file = get_latest_model_path()

class PredictionResponse(BaseModel):
    Class: str
    Confidence: float

@app.get("/")
async def ping():
    return "hello"

@app.get("/health")
async def health():
    logging.debug("/health endpoint called")
    if USE_TF_SERVING:
        try:
            res = requests.get(TF_SERVING_HEALTH_URL)
            if res.status_code == 200:
                return {"status": "ok", "mode": "tf-serving"}
            return {"status": "fail", "details": res.json()}
        except Exception as e:
            return {"status": "fail", "error": str(e), "mode": "tf-serving"}
    else:
        if latest_model_file and os.path.exists(latest_model_file):
            return {"status": "ok", "mode": "local"}
        return {"status": "fail", "mode": "local", "error": "Model not found"}

@app.get("/model-info")
async def model_info():
    logging.info("Model info endpoint hit. Responding with model path and number of classes.")
    if not latest_model_file:
        return {"error": "No model loaded"}
    return {
        "model_path": str(latest_model_file),
        "num_classes": len(CLASS_NAMES),
        "use_tf_serving": USE_TF_SERVING
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    start = time.time()
    try:
        image = read_file_as_image(await file.read())
        prediction = predict_model(image)
        logging.info(f"Prediction done in {round(time.time() - start, 4)}s: {prediction}")
        return PredictionResponse(**prediction)
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)