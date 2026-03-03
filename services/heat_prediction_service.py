import numpy as np
from Model.heat_tcn_model import model, scaler_x, scaler_y
from services.heat_firebase_service import get_latest_readings

LOOKBACK = 6  # 2 hours (6 × 20 min)

def predict_thi_1hr(cattle_id: str):
    rows, latest  = get_latest_readings(cattle_id)

    if len(rows) < LOOKBACK:
        return None

    X = scaler_x.transform(np.array(rows))
    X = X.reshape(1, LOOKBACK, 2)

    pred_scaled = model.predict(X, verbose=0)
    thi = scaler_y.inverse_transform(pred_scaled)[0][0]

    return round(float(thi), 2), latest
