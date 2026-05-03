import joblib
import os

class ModelLoader:
  def __init__(self):
    model_path = os.path.join("models")

    files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]

    if not files:
      raise Exception("No trained model found!")

    self.model_path = os.path.join(model_path, files[0])
    self.model = joblib.load(self.model_path)

    print(f"✅ Model loaded: {self.model_path}")

  def get_model(self):
    return self.model