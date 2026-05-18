from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load trained model
model = joblib.load("models/anomaly_model.pkl")


@app.get("/")
def home():
    return {"message": "OpsMind AI API is running"}


@app.post("/predict")
def predict(cpu_usage: float, memory_usage: float, response_time: float):

    data = np.array([[cpu_usage, memory_usage, response_time]])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }