import pandas as pd
import joblib
from datetime import datetime, timedelta
import os


# XGBoost and Cox models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "AI_Predictor")

xgb_model = joblib.load(os.path.join(MODEL_DIR, "pregnancy_model.pkl"))
cox_model = joblib.load(os.path.join(MODEL_DIR, "cox_model.pkl"))

features_order = [
    "DIM", "Age_at_PD_months", "Lactation No", "AI_Count",
    "Milk_Yield", "Breed", "Milking/Dry",
    "Hormonal Treatment", "Estrus Cycle Length",
    "Estrus Signs", "Past_AI_Success_Rate", "Days_Since_Last_Estrus"
]

survival_features = [
    "DIM", "Milk_Yield", "Lactation No",
    "AI_Count", "Hormonal Treatment", "Estrus Cycle Length",
    "Estrus Signs", "Past_AI_Success_Rate", "Days_Since_Last_Estrus"
]


# def prepare_features(row: dict):
#     """Prepare features from your raw dataset row."""
#     features = {}

#     # Direct mapping
#     features["Lactation No"] = row["Lactation No"]
#     features["Milk_Yield"] = row["Milk_Yield"]
#     features["Breed"] = row["Breed"]
#     features["Milking/Dry"] = row["Milking/Dry"]
#     features["Hormonal Treatment"] = row["Hormonal Treatment"]
#     features["Estrus Cycle Length"] = row["Estrus Cycle Length"]
#     features["Milking/Dry"] = 1 if str(features["Milking/Dry"]).lower() == "yes" else 0
#     features["Hormonal Treatment"] = 1 if str(features["Hormonal Treatment"]).lower() == "yes" else 0

#     prev_ai = row.get("Previous AI Dates", "")
#     if isinstance(prev_ai, str):
#         features["AI_Count"] = len(prev_ai.split(",")) if prev_ai else 0
#     else:
#         features["AI_Count"] = len(prev_ai)

#     last_caving = pd.to_datetime(row["Last Caving Date"])
#     features["DIM"] = (pd.to_datetime("today") - last_caving).days

#     features["Age_at_PD_months"] = row["E. Age (Month)"]

#     return features
def prepare_features(row: dict):
    features = {}

    # numeric features
    features["Lactation No"] = int(row["Lactation No"])
    features["Milk_Yield"] = float(row["Milk_Yield"])
    features["Estrus Cycle Length"] = int(row["Estrus Cycle Length"])
    features["Age_at_PD_months"] = int(row["E. Age (Month)"])

    # Breed encoding
    breed_map = {
        "Jersey": 0,
        "Friesian": 1,
        "Crossbreed": 2
    }
    features["Breed"] = breed_map.get(row["Breed"], 0)

    # Milking / Dry encoding
    milking_map = {
        "Milking": 1,
        "Dry": 0
    }
    features["Milking/Dry"] = milking_map.get(row["Milking/Dry"], 0)

    # Hormonal treatment encoding
    hormonal_map = {
        "Yes": 1,
        "No": 0
    }
    features["Hormonal Treatment"] = hormonal_map.get(row["Hormonal Treatment"], 0)

    # AI count
    prev_ai = row.get("Previous AI Dates", [])
    if isinstance(prev_ai, str):
        features["AI_Count"] = len(prev_ai.split(",")) if prev_ai else 0
    else:
        features["AI_Count"] = len(prev_ai)

    # Days in milk
    last_caving = pd.to_datetime(row["Last Caving Date"])
    features["DIM"] = (pd.to_datetime("today") - last_caving).days
    
    # Defaults for new survival features if not provided by frontend
    features["Estrus Signs"] = int(row.get("Estrus Signs", 0))
    features["Past_AI_Success_Rate"] = float(row.get("Past_AI_Success_Rate", 0.0))
    features["Days_Since_Last_Estrus"] = int(row.get("Days_Since_Last_Estrus", features.get("Estrus Cycle Length", 21)))

    return features

def recommend_next_ai(row: dict):
    data = prepare_features(row)

    df = pd.DataFrame([data])
    df_cox = df[survival_features]

    surv = cox_model.predict_survival_function(df_cox)

    recommended_day = surv[surv.columns[0]][
        surv[surv.columns[0]] <= 0.5
    ].index.min()

    if pd.isna(recommended_day):
        recommended_day = surv.index.max()

    recommended_day += int(data.get("Estrus Cycle Length", 21))

    next_ai_date = datetime.today() + timedelta(days=int(recommended_day))

    if next_ai_date.date() < datetime.today().date():
        next_ai_date = datetime.today() + timedelta(days=7)

    return {
        "recommended_next_ai": next_ai_date.strftime("%Y-%m-%d")
    }

def predict_pregnancy(row: dict):
    data = prepare_features(row)

    df = pd.DataFrame([data])
    df_xgb = df[features_order]

    prob = xgb_model.predict_proba(df_xgb)[:, 1][0]

    if prob > 0.7:
        risk_level = "Low"
    elif prob > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "High"

    ai_date = pd.to_datetime(row["AI_Date"])
    days_since_ai = (datetime.today() - ai_date).days
    pregnancy_check_date = ai_date + timedelta(days=30)

    return {
        "pregnancy_probability": round(float(prob), 4),
        "risk_level": risk_level,
        "days_since_ai": days_since_ai,
        "pregnancy_check_date": pregnancy_check_date.strftime("%Y-%m-%d"),
    }