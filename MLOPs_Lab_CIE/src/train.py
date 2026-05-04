import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

# ✅ Fix MLflow path issue
mlflow.set_tracking_uri("file:./mlruns")

# ✅ Set experiment
mlflow.set_experiment("mediqueue-wait-time-min")

# Load data
df = pd.read_csv("data/training_data.csv")

print("Data loaded:")
print(df.head())

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("mediqueue-wait-time-min")

results = []
best_rmse = float("inf")
best_model_name = None

models = {
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"{name} -> MAE: {mae}, RMSE: {rmse}")

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("experiment_type", "baseline_comparison")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

output = {
    "experiment_name": "mediqueue-wait-time-min",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": float(best_rmse)
}

print("Final output:", output)

# Save JSON
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Step 1 completed and JSON saved")