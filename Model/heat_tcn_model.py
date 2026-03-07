import os
import tensorflow as tf
import joblib
from tcn import TCN


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "heat")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "tcn_heat_model.keras")
SCALER_X_PATH = os.path.join(ARTIFACTS_DIR, "scaler_x.save")
SCALER_Y_PATH = os.path.join(ARTIFACTS_DIR, "scaler_y.save")


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TCN": TCN}
)

scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
