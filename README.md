# 🌦️ Weather Forecasting System Using Deep Learning (LSTM)

An AI-powered weather forecasting application that predicts future weather conditions using **Long Short-Term Memory (LSTM)** neural networks. The project leverages historical weather data from the **Open-Meteo API** to train a deep learning model capable of forecasting weather based on time-series patterns.

---

## 🚀 Project Overview

Traditional weather forecasting relies on complex atmospheric models. This project demonstrates how **Deep Learning (LSTM)** can learn temporal weather patterns from historical data and predict future weather conditions.

The application automatically downloads weather data, preprocesses it, trains an LSTM model, and provides weather predictions through an interactive **Streamlit** web interface.

---

## ✨ Features

- 🌍 Automatic weather data collection using Open-Meteo API
- 📍 Supports multiple Indian districts and cities
- 📅 Dynamic historical dataset generation (Last 5 Years)
- 🧹 Data preprocessing and normalization
- 🤖 Deep Learning model using LSTM
- 📈 Time-series weather prediction
- 🌐 User-friendly Streamlit web application
- 💾 Saved trained model for future predictions
- 📊 Interactive visualization of actual vs predicted values

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras (LSTM)

### Data Processing
- Pandas
- NumPy
- Scikit-learn

### Data Source
- Open-Meteo API

### Visualization
- Matplotlib

### Web Framework
- Streamlit

---

## 📂 Project Structure

```
weather_prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── src/
│   ├── create_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
└── graphs/
```

---

## ⚙️ Workflow

1. Collect historical weather data using Open-Meteo API.
2. Clean and preprocess the dataset.
3. Normalize the weather features.
4. Generate sequential time-series data.
5. Train an LSTM neural network.
6. Save the trained model.
7. Predict future weather conditions.
8. Display results using Streamlit.

---

## 📊 Machine Learning Pipeline

```
Open-Meteo API
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Feature Scaling
        │
        ▼
Sequence Generation
        │
        ▼
LSTM Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Weather Prediction
        │
        ▼
Streamlit Web App
```

---

## 📈 Model Architecture

- Input Layer
- LSTM Layer
- Dropout Layer
- Dense Layer
- Output Layer

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam

Evaluation Metrics:
- MAE
- RMSE

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/weather_prediction.git
```

Move into the project

```bash
cd weather_prediction
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Generate Dataset

```bash
python src/create_dataset.py
```

Preprocess Data

```bash
python src/preprocess.py
```

Train Model

```bash
python src/train_model.py
```

Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Future Enhancements

- Real-time weather forecasting
- Rainfall prediction
- Temperature prediction for multiple days
- Wind speed forecasting
- Humidity prediction
- Weather alerts
- Cloud deployment
- Mobile-friendly interface

---

## 📚 Skills Demonstrated

- Deep Learning
- LSTM Networks
- Time Series Forecasting
- TensorFlow
- Keras
- Data Preprocessing
- API Integration
- Python Programming
- Streamlit Deployment
- Machine Learning Pipeline

---

## 🎯 Learning Outcomes

- Developed an end-to-end Deep Learning project.
- Implemented time-series forecasting using LSTM.
- Automated weather data collection through APIs.
- Built a complete Machine Learning pipeline.
- Created an interactive web application for predictions.

---
