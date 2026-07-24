import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.weather_api import get_current_weather
from src.predictor import predict_weather

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="AI Weather Forecast",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ AI Weather Forecasting System")
st.write("Weather Forecasting using Deep Learning (LSTM)")

# ------------------------------------
# Load Districts
# ------------------------------------

districts = pd.read_csv(BASE_DIR / "dataset" / "districts.csv")

selected = st.selectbox(
    "Select District",
    sorted(districts["district"].unique())
)

row = districts[districts["district"] == selected].iloc[0]

lat = row["latitude"]
lon = row["longitude"]

# ------------------------------------
# Prediction Button
# ------------------------------------

if st.button("🔍 Get Weather Forecast"):

    current = get_current_weather(lat, lon)

    prediction = predict_weather(selected)

    st.header("🌤 Current Weather")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Temperature",
        f"{current['temperature']} °C"
    )

    c2.metric(
        "Humidity",
        f"{current['humidity']} %"
    )

    c3.metric(
        "Rainfall",
        f"{current['rainfall']} mm"
    )

    c4.metric(
        "Wind Speed",
        f"{current['wind_speed']} km/h"
    )

    st.divider()

    st.header("🤖 Tomorrow AI Prediction")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Temperature",
        f"{prediction['temperature']} °C"
    )

    p2.metric(
        "Humidity",
        f"{prediction['humidity']} %"
    )

    p3.metric(
        "Rainfall",
        f"{prediction['rainfall']} mm"
    )

    p4.metric(
        "Wind Speed",
        f"{prediction['wind_speed']} km/h"
    )

    st.divider()

    st.header("📈 Temperature Comparison")

    fig, ax = plt.subplots(figsize=(8,4))

    labels = [
        "Current",
        "Tomorrow"
    ]

    values = [
        current["temperature"],
        prediction["temperature"]
    ]

    ax.plot(
        labels,
        values,
        marker="o",
        linewidth=3
    )

    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)

    st.pyplot(fig)

    st.divider()

    st.success(
        f"""
### AI Weather Summary

📍 **District:** {selected}

🌡 Current Temperature: {current['temperature']} °C

🤖 Predicted Temperature: {prediction['temperature']} °C

💧 Humidity: {prediction['humidity']} %

🌧 Rainfall: {prediction['rainfall']} mm

💨 Wind Speed: {prediction['wind_speed']} km/h
"""
    )