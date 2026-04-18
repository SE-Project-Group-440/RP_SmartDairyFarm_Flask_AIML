import os
SECRET_KEY = "your-super-secret-keygggggggggggggggggggg"



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_RF = os.path.join(BASE_DIR, "Model", "milk_yield_rf_v5.pickle")
MODEL_PATH = os.path.join(BASE_DIR, "AI_Predictor", "pregnancy_model.pkl")
COX_MODEL_PATH = os.path.join(BASE_DIR, "AI_Predictor", "cox_model.pkl") 