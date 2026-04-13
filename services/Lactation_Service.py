import joblib
import numpy as np
from config import MODEL_PATH_RF

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH_RF)

    def predict(self, features):
        x = np.array(features).reshape(1, -1)
        return float(self.model.predict(x)[0])
