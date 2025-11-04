
import os, torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    from .predict import load_model, predict_url
except ImportError:
    # Allow running with uvicorn src.serve:app
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.predict import load_model, predict_url

CKPT_PATH = os.environ.get("CKPT_PATH", "artifacts/best.pt")
model, stoi, model_args = load_model(CKPT_PATH)

app = FastAPI(title="Phishing URL CNN")

# CORS for local frontend dev
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLItem(BaseModel):
    url: str

@app.post("/predict")
def do_predict(item: URLItem):
    prob = predict_url(model, stoi, model_args, item.url)
    return {"phish_probability": float(prob), "predicted_label": int(prob >= 0.5)}
