from load_data import load_dataset
from preprocess import FraudPreprocessor
from models import ModelRegistry
from evaluate import Evaluator
import joblib
import os
import numpy as np
import json

class Trainer:
  def __init__(self):
    self.preprocessor = FraudPreprocessor()
    self.model_registry = ModelRegistry()
    self.evaluator = Evaluator()

  def train(self):
    # 1. Load data
    df = load_dataset()

    # 2. Preprocess
    X_train, X_test, y_train, y_test = self.preprocessor.preprocess(df)

    sample = X_test[0].tolist()
    os.makedirs("../models", exist_ok=True)
    with open("../models/sample.json", "w") as f:
      json.dump(sample, f)
    
    # 3. Get models
    models = self.model_registry.get_models()

    best_model = None
    best_score = 0

    # 4. Train all models
    for name, model in models.items():
      print(f"\n🚀 Training {name}...")

      model.fit(X_train, y_train)

      print(f"\n📊 Evaluating {name}")
      self.evaluator.evaluate(model, X_test, y_test)

      score = model.score(X_test, y_test)

      if score > best_score:
        best_score = score
        best_model = model

    # 5. Save best model
    self.save_model(best_model)

  def save_model(self, model):
    os.makedirs("../models", exist_ok=True)

    path = "../models/fraud_model.pkl"
    joblib.dump(model, path)

    print(f"\n💾 Best model saved at {path}")


if __name__ == "__main__":
  trainer = Trainer()
  trainer.train()