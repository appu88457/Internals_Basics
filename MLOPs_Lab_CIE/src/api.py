from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os

app = FastAPI()

import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/mediqueue_model/1")

class InputData(BaseModel):
    patients_ahead: int
    staff_count: int
    is_emergency: int
    dept_load: int

@app.get("/status")
def status():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.patients_ahead, data.staff_count, data.is_emergency, data.dept_load]])
    pred = float(model.predict(X)[0])

    # write JSON result automatically
    result = {
        "health_endpoint": "/status",
        "predict_endpoint": "/predict",
        "port": 8500,
        "health_response": {"status": "healthy", "model_loaded": True},
        "test_input": data.dict(),
        "prediction": pred
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step2_s4.json", "w") as f:
        json.dump(result, f, indent=2)

    return {"prediction": pred}