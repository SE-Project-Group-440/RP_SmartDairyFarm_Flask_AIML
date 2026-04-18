import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH_RF

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH_RF)

    def predict(self, features):
        x = np.array(features).reshape(1, -1)

        if hasattr(self.model, "feature_names_in_"):
            try:
                columns = list(self.model.feature_names_in_)
                x = pd.DataFrame(x, columns=columns)
            except Exception:
                pass

        return float(self.model.predict(x)[0])
