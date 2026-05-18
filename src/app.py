from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# load model
model = joblib.load("../models/anomaly_model.pkl")

@app.get("/")
def home():
    return {"message": "OpsMind AI API is running"}

@app.post("/predict")
def predict_incident(
    system: int,
    incident_type: int,
    severity: int,
    region: int,
    cpu_usage: float,
    memory_usage: float,
    latency_ms: float
):

    data = pd.DataFrame([{
        "system": system,
        "incident_type": incident_type,
        "severity": severity,
        "region": region,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "latency_ms": latency_ms
    }])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }