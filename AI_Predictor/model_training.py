import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import joblib
from lifelines.utils import concordance_index


#LOAD DATA
df = pd.read_excel("Reproduction_with_Milk.xlsx")
df.columns = df.columns.str.strip()


#CONVERT DATE COLUMNS
date_cols = ['Date of Birth', 'Last Caving Date', 'Caving Date', 'PD Date', 'Last Estrus/Heat Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# FEATURE ENGINEERING
df["DIM"] = (df["PD Date"] - df["Last Caving Date"]).dt.days
df["Age_at_PD_months"] = (df["PD Date"] - df["Date of Birth"]).dt.days / 30.4
df["Pregnant"] = df["P/NP"].map({"P": 1, "NP": 0})
df["AI_Count"] = df.groupby("Tag No.").cumcount() + 1
df = df[df["DIM"] >= 0]  

df["Estrus Cycle Length"] = pd.to_numeric(df["Estrus Cycle Length"], errors='coerce')
df["Estrus Cycle Length"] = df["Estrus Cycle Length"].fillna(21)

# ENCODE CATEGORICAL VARIABLES
categorical_cols = ["Breed", "Milking/Dry", "Hormonal Treatment", "Estrus Signs", "Location"]
le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le  


# Days since last estrus
df["Days_Since_Last_Estrus"] = (df["PD Date"] - df["Last Estrus/Heat Date"]).dt.days

# Fill missing with median cycle length if estrus not recorded
df["Days_Since_Last_Estrus"] = df["Days_Since_Last_Estrus"].fillna(df["Estrus Cycle Length"].median())

# PREPARE MODEL FEATURES
features = [
    "DIM", "Age_at_PD_months", "Lactation No", "AI_Count", "Milk_Yield",
    "Breed", "Milking/Dry", "Hormonal Treatment", "Estrus Cycle Length",
    "Estrus Signs", "Days_Since_Last_Estrus"
]
model_df = df[features + ["Pregnant"]].dropna()
for col in features:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')
model_df = model_df.dropna()

X = model_df.drop("Pregnant", axis=1)
y = model_df["Pregnant"]
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBOOST MODEL
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict probabilities for all cows
prediction_df = df[features].copy()
prediction_df = prediction_df.apply(pd.to_numeric, errors='coerce')
prediction_df = prediction_df.fillna(prediction_df.median())

df["Pregnancy_Prob_Today"] = xgb_model.predict_proba(prediction_df)[:, 1]


# COX SURVIVAL MODEL (TIME TO PREGNANCY)
df["Time_to_Pregnancy"] = (df["PD Date"] - df["Caving Date"]).dt.days
survival_features = ["DIM", "Milk_Yield", "Lactation No", "AI_Count", "Hormonal Treatment", "Estrus Cycle Length",
    "Estrus Signs", "Days_Since_Last_Estrus"]
survival_df = df[["Time_to_Pregnancy", "Pregnant"] + survival_features].dropna()

cph = CoxPHFitter()
cph.fit(survival_df, duration_col="Time_to_Pregnancy", event_col="Pregnant")

# Compute threshold probability from historical pregnancies
pregnant_rows = survival_df[survival_df["Pregnant"] == 1]

threshold_probs = []

for idx, row in pregnant_rows.iterrows():
    x = row[survival_features].to_frame().T
    surv_func = cph.predict_survival_function(x)
    day_of_event = row["Time_to_Pregnancy"]
    
    # survival probability on the day pregnancy occurred
    if day_of_event in surv_func.index:
        prob_at_event = surv_func.loc[day_of_event].values[0]
        threshold_probs.append(prob_at_event)

# Use median probability as threshold
threshold_prob = np.median(threshold_probs)
print(f"Data-driven threshold probability: {threshold_prob:.3f}")


# RECOMMEND NEXT AI DATE
not_pregnant_df = df[df["Pregnant"] == 0].copy()
next_ai_dates = []

for idx, row in not_pregnant_df.iterrows():
    x = row[survival_features].to_frame().T
    surv_func = cph.predict_survival_function(x)
    
    recommended_day = surv_func[surv_func.columns[0]][surv_func[surv_func.columns[0]] <= threshold_prob].index.min()
    if pd.isna(recommended_day):
        recommended_day = surv_func.index.max()
    
    estrus_cycle = row.get("Estrus Cycle Length", 21)  
    recommended_day += estrus_cycle 
    next_ai_dates.append(row["Caving Date"] + pd.Timedelta(days=int(recommended_day)))

not_pregnant_df["Recommended_Next_AI"] = next_ai_dates
not_pregnant_df.to_excel("NonPregnant_Recommended_AI.xlsx", index=False)

print("\nRecommended AI dates saved to 'NonPregnant_Recommended_AI.xlsx'")

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')

print("\n===== Cross Validation (5-Fold) =====")
print("AUC Scores:", cv_scores)
print("Mean AUC :", cv_scores.mean())

#SAVE MODEL
joblib.dump(xgb_model, "pregnancy_model.pkl")
joblib.dump(cph, "cox_model.pkl")
print("Models saved: pregnancy_model.pkl and cox_model.pkl")


# Predictions on test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# ROC Curve
plt.figure(figsize=(6,5))
RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
plt.title("ROC Curve - Pregnancy Prediction")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NP","P"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
plt.figure(figsize=(8,6))
plt.barh(X.columns, xgb_model.feature_importances_)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.show()

# Model Performance Metrics Bar Chart
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== XGBoost Model Performance =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "AUC": auc
}

plt.figure(figsize=(6,5))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.ylim(0,1)
plt.title("XGBoost Model Performance Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

c_index = concordance_index(
    event_times=survival_df["Time_to_Pregnancy"],
    predicted_scores=-cph.predict_partial_hazard(survival_df),
    event_observed=survival_df["Pregnant"]
)
print(f"Cox Model C-index: {c_index:.4f}")

# Plot survival curves for first 5 cows
cph.plot_partial_effects_on_outcome(
    covariates="AI_Count",
    values=[1, 2, 3],
    plot_baseline=True
)
plt.show()