import joblib
import numpy as np
from config import MODEL_PATH

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, features):
        x = np.array(features).reshape(1, -1)
        return float(self.model.predict(x)[0])
