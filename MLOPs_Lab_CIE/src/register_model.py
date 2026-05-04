import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# ✅ Use local MLflow storage (fixes Windows path issue)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mediqueue-wait-time-min")

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train best model (Lasso)
model = Lasso()
model.fit(X_train, y_train)

# Register model
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, name="lasso_model")
    model_uri = f"runs:/{run.info.run_id}/lasso_model"
    result = mlflow.register_model(model_uri, "mediqueue_model")

# Save JSON output
output = {
    "model_name": "mediqueue_model",
    "version": int(result.version),
    "stage": "None"
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Model registered:", output)