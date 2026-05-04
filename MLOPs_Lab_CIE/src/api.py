from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = Lasso()
model.fit(X_train, y_train)

# 🔥 IMPORTANT: THIS LINE MUST EXIST
app = FastAPI()

class InputData(BaseModel):
    patients_ahead: int = Field(..., ge=1, le=30)
    staff_count: int = Field(..., ge=2, le=15)
    is_emergency: int = Field(..., ge=0, le=1)
    dept_load: int = Field(..., ge=1, le=5)

@app.get("/status")
def status():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array([[data.patients_ahead, data.staff_count, data.is_emergency, data.dept_load]])
    pred = model.predict(arr)[0]
    return {"prediction": float(pred)}