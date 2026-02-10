from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware


# Load model & helpers
model = joblib.load("stress_model.pkl")
feature_columns = joblib.load("model_features.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (OK for hackathon)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

# -------------------------------
# Input schema (API contract)
# -------------------------------
class StressInput(BaseModel):
    gender: str
    age: int
    sleep_duration: float
    quality_of_sleep: int
    physical_activity: int
    heart_rate: int
    daily_steps: int
    systolic_bp: int
    diastolic_bp: int
    occupation: str
    bmi_category: str
    sleep_disorder: str


# -------------------------------
# Stress label logic
# -------------------------------
def stress_label(val):
    if val <= 4:
        return "Low"
    elif val <= 6:
        return "Medium"
    else:
        return "High"


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_stress(data: StressInput):
    # Encode gender
    gender_encoded = gender_encoder.transform([data.gender])[0]

    # Build input row
    input_dict = {
        "Gender": gender_encoded,
        "Age": data.age,
        "Sleep Duration": data.sleep_duration,
        "Quality of Sleep": data.quality_of_sleep,
        "Physical Activity Level": data.physical_activity,
        "Heart Rate": data.heart_rate,
        "Daily Steps": data.daily_steps,
        "Systolic_BP": data.systolic_bp,
        "Diastolic_BP": data.diastolic_bp,
    }

    df = pd.DataFrame([input_dict])

    # One-hot columns (same naming as training)
    def add_dummy(col_name):
        if col_name not in df.columns:
            df[col_name] = 0

    # Occupation
    add_dummy(f"Occupation_{data.occupation}")

    # BMI
    add_dummy(f"BMI Category_{data.bmi_category}")

    # Sleep Disorder
    add_dummy(f"Sleep Disorder_{data.sleep_disorder}")

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(df)[0]
    label = stress_label(prediction)

    return {
        "stress_score": round(float(prediction), 2),
        "stress_level": label
    }
