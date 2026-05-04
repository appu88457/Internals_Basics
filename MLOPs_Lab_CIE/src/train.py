import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lasso = Lasso()
lasso.fit(X_train, y_train)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Evaluate
lasso_rmse = mean_squared_error(y_test, lasso.predict(X_test)) ** 0.5
rf_rmse = mean_squared_error(y_test, rf.predict(X_test)) ** 0.5

# Select best model
if lasso_rmse < rf_rmse:
    best_model = lasso
else:
    best_model = rf

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print("✅ Model saved successfully")