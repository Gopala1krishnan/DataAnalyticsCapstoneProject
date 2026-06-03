from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Hospital Shortage Prediction API", version="1.0.0")

PRIMARY_MODEL_PATH = "artifacts/primary_shortage_model.joblib"
MENTAL_MODEL_PATH = "artifacts/mental_shortage_model.joblib"

try:
    primary_model = joblib.load(PRIMARY_MODEL_PATH)
    mental_model = joblib.load(MENTAL_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

class HospitalInput(BaseModel):
    Category: str
    LICENSED_BED_SIZE: str
    Tot_ED_NmbVsts: float = Field(..., ge=0)
    EDStations: float = Field(..., ge=0)

class PredictRequest(BaseModel):
    target: str  # PrimaryCareShortageArea or MentalHealthShortageArea
    data: HospitalInput

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    target = req.target.strip()
    if target not in ["PrimaryCareShortageArea", "MentalHealthShortageArea"]:
        raise HTTPException(status_code=400, detail="Invalid target")

    visits_per_station = req.data.Tot_ED_NmbVsts / (req.data.EDStations + 1e-5)

    row = {
        "Category": req.data.Category,
        "LICENSED_BED_SIZE": req.data.LICENSED_BED_SIZE,
        "Tot_ED_NmbVsts": req.data.Tot_ED_NmbVsts,
        "EDStations": req.data.EDStations,
        "Visits_Per_Station": visits_per_station,
    }
    X = pd.DataFrame([row])

    model = primary_model if target == "PrimaryCareShortageArea" else mental_model

    pred = model.predict(X)[0]
    conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0].max())

    return {
        "target": target,
        "prediction": int(pred),
        "confidence": conf,
        "input_used": row,
    }
