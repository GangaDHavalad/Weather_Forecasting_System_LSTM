import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "weather_dataset.csv"

FEATURES = [
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed"
]

# Lazily-loaded resources
_model = None
_scaler = None
_df = None


def _load_resources():
    """Load model, scaler, and dataset on first use to avoid heavy imports at module import time."""
    global _model, _scaler, _df

    if _scaler is None:
        import joblib
        _scaler = joblib.load(str(SCALER_PATH))

    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        _df["date"] = pd.to_datetime(_df["date"])

    if _model is None:
        try:
            from tensorflow.keras.models import load_model
        except Exception:
            _model = "unavailable"
            return

        if not MODEL_PATH.exists():
            _model = "unavailable"
            return

        try:
            _model = load_model(str(MODEL_PATH))
        except Exception:
            _model = "unavailable"


def predict_weather(district):
    _load_resources()

    district_df = _df[
        _df["district"].str.lower() == district.lower()
    ].copy()

    if district_df.empty:
        return {
            "temperature": 0,
            "humidity": 0,
            "rainfall": 0,
            "wind_speed": 0
        }

    district_df = district_df.sort_values("date")

    # Scale features
    district_df[FEATURES] = _scaler.transform(
        district_df[FEATURES]
    )

    # Last 30 days
    sequence = district_df[FEATURES].tail(30).values

    if len(sequence) < 30:
        return {
            "temperature": 0,
            "humidity": 0,
            "rainfall": 0,
            "wind_speed": 0
        }

    sequence = sequence.reshape(1, 30, 4)

    # Predict
    prediction = _model.predict(
        sequence,
        verbose=0
    )

    prediction = prediction.reshape(1, 4)

    prediction = _scaler.inverse_transform(
        prediction
    )[0]

    return {
        "temperature": round(float(prediction[0]), 2),
        "humidity": round(float(max(0, prediction[1])), 2),
        "rainfall": round(float(max(0, prediction[2])), 2),
        "wind_speed": round(float(max(0, prediction[3])), 2)
    }