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

_model = None
_scaler = None
_df = None


def _load_resources():
    global _model, _scaler, _df

    if _scaler is None:
        if SCALER_PATH.exists():
            import joblib
            _scaler = joblib.load(str(SCALER_PATH))
        else:
            _scaler = None

    if _df is None:
        if DATA_PATH.exists():
            _df = pd.read_csv(DATA_PATH)
            if "date" in _df.columns:
                _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
            else:
                _df = None
        else:
            _df = None

    if _model is None:
        if not MODEL_PATH.exists():
            _model = None
            return

        try:
            from tensorflow.keras.models import load_model  # type: ignore
            _model = load_model(str(MODEL_PATH))
        except Exception:
            _model = None


def _fallback_prediction(district, current=None):
    if current:
        temperature = float(current.get("temperature", 0))
        humidity = float(current.get("humidity", 0))
        rainfall = float(current.get("rainfall", 0))
        wind_speed = float(current.get("wind_speed", 0))

        return {
            "temperature": round(temperature + 1.2, 2),
            "humidity": round(max(0, humidity + 1.5), 2),
            "rainfall": round(max(0, rainfall + 0.4), 2),
            "wind_speed": round(max(0, wind_speed + 0.7), 2),
        }

    district_name = (district or "").strip().lower()
    base_temp = 24.0 if "bengaluru" in district_name else 22.0
    base_humidity = 62.0 if "bengaluru" in district_name else 58.0

    return {
        "temperature": round(base_temp, 2),
        "humidity": round(base_humidity, 2),
        "rainfall": 2.4,
        "wind_speed": 8.5,
    }


def predict_weather(district, current=None):
    _load_resources()

    if _model is not None and _scaler is not None and _df is not None and not _df.empty:
        district_df = _df[
            _df["district"].astype(str).str.lower() == district.lower()
        ].copy()

        if district_df.empty:
            return _fallback_prediction(district, current)

        district_df = district_df.sort_values("date")
        district_df[FEATURES] = _scaler.transform(district_df[FEATURES])
        sequence = district_df[FEATURES].tail(30).values

        if len(sequence) < 30:
            return _fallback_prediction(district, current)

        sequence = sequence.reshape(1, 30, 4)
        prediction = _model.predict(sequence, verbose=0).reshape(1, 4)
        prediction = _scaler.inverse_transform(prediction)[0]

        return {
            "temperature": round(float(prediction[0]), 2),
            "humidity": round(float(max(0, prediction[1])), 2),
            "rainfall": round(float(max(0, prediction[2])), 2),
            "wind_speed": round(float(max(0, prediction[3])), 2),
        }

    return _fallback_prediction(district, current)
