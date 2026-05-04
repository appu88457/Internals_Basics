import mlflow
import json

mlflow.set_tracking_uri("file:./mlruns")

client = mlflow.tracking.MlflowClient()

model_name = "mediqueue_model"
version = 1

# Promote to Staging
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging"
)

output = {
    "model_name": model_name,
    "version": version,
    "new_stage": "Staging"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Model promoted:", output)