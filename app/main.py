from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="HR Kündigungsprognose API")

BASE_DIR = Path(__file__).resolve().parents[1]  # app/ -> Projektroot
MODEL_PATH = BASE_DIR / "model" / "logreg_pipeline.joblib"

# Modell laden (einmal beim Start)
model = joblib.load(MODEL_PATH)


class EmployeeInput(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    Department: str
    salary: str


@app.get("/")
def root():
    return {"status": "ok", "message": "HR Kündigungsprognose API läuft"}


@app.post("/predict")
def predict(inp: EmployeeInput):
    df = pd.DataFrame([inp.model_dump()])
    proba = float(model.predict_proba(df)[0][1])
    return {
        "quit_probability": proba,
        "threshold_default": 0.5,
        "prediction": int(proba >= 0.5)
    }
