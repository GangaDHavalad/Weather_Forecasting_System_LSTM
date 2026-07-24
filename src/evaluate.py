import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -------------------------
# Load model
# -------------------------
model = load_model("models/lstm_model.keras")

# -------------------------
# Load test data
# -------------------------
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

# -------------------------
# Predict
# -------------------------
predictions = model.predict(X_test)

# -------------------------
# Load scaler
# -------------------------
scaler = joblib.load("models/scaler.pkl")

# Convert back to original values
y_test_original = scaler.inverse_transform(y_test)
pred_original = scaler.inverse_transform(predictions)

feature_names = [
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind Speed"
]

# -------------------------
# Metrics
# -------------------------
for i, feature in enumerate(feature_names):

    mae = mean_absolute_error(
        y_test_original[:, i],
        pred_original[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original[:, i],
            pred_original[:, i]
        )
    )

    r2 = r2_score(
        y_test_original[:, i],
        pred_original[:, i]
    )

    print(f"\n{feature}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

# -------------------------
# Plot Temperature
# -------------------------
plt.figure(figsize=(12,5))

plt.plot(
    y_test_original[:100,0],
    label="Actual"
)

plt.plot(
    pred_original[:100,0],
    label="Predicted"
)

plt.title("Temperature Prediction")
plt.xlabel("Samples")
plt.ylabel("Temperature (°C)")
plt.legend()

plt.savefig("graphs/temperature_prediction.png")
plt.show()