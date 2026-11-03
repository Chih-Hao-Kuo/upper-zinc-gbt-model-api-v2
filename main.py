import os
import ydf
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Dynamically locate the folder next to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_dir")

API_KEY = os.environ.get("API_KEY", "my_default_secret_key")
MODEL_VERSION = "2026-11-03"

# Load using the absolute path
model = ydf.load_model(MODEL_PATH)

class Features(BaseModel):
    # accept BOTH the original Chinese keys and simple ASCII keys
    model_config = ConfigDict(populate_by_name=True)
    speed: float = Field(..., alias="綫速")
    current_ratio: float = Field(..., alias="上表面電流/鋼板寬度(A/mm)")
    be_value: float = Field(..., alias="電鍍BE值")

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(f: Features, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
    examples = {
        "綫速": [f.speed],
        "上表面電流/鋼板寬度(A/mm)": [f.current_ratio],
        "電鍍BE值": [f.be_value],
    }
    out = model.predict(examples)
    return {"prediction": float(out[0]), "model_version": MODEL_VERSION}
