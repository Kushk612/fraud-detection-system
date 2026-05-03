from fastapi import FastAPI
import numpy as np

from app.schema import Transaction
from app.model_loader import ModelLoader

import json
import os

app = FastAPI(title="Fraud Detection API")

# Load model once at startup
loader = ModelLoader()
model = loader.get_model()


@app.get("/health")
def health():
  return {"status": "running", "message": "Fraud API is active"}

@app.get("/test-sample")
def test_sample():
  path = os.path.join("models", "sample.json")

  with open(path, "r") as f:
    sample = json.load(f)

  return {"sample": sample}

@app.post("/predict-from-sample")
def predict_from_sample(sample: dict):
  data = np.array(sample["sample"]).reshape(1, -1)

  prob = model.predict_proba(data)[0][1]
  pred = model.predict(data)[0]

  return {
      "fraud_prediction": int(pred),
      "fraud_probability": float(prob),
      "risk": "HIGH" if prob > 0.7 else "LOW"
  }

@app.post("/predict")
def predict(transaction: Transaction):

  # Convert input to array
  data = np.array([
      transaction.Time,
      transaction.Amount,
      transaction.V1,
      transaction.V2,
      transaction.V3,
      transaction.V4,
      transaction.V5,
      transaction.V6,
      transaction.V7,
      transaction.V8,
      transaction.V9,
      transaction.V10,
      transaction.V11,
      transaction.V12,
      transaction.V13,
      transaction.V14,
      transaction.V15,
      transaction.V16,
      transaction.V17,
      transaction.V18,
      transaction.V19,
      transaction.V20,
      transaction.V21,
      transaction.V22,
      transaction.V23,
      transaction.V24,
      transaction.V25,
      transaction.V26,
      transaction.V27,
      transaction.V28
  ]).reshape(1, -1)

  # Prediction
  prob = model.predict_proba(data)[0][1]
  pred = model.predict(data)[0]

  return {
      "fraud_prediction": int(pred),
      "fraud_probability": float(prob),
      "risk_level": "HIGH" if prob > 0.7 else "LOW"
  }