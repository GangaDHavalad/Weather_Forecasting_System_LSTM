import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/raw/weather_dataset.csv")

# -----------------------------
# Convert date
# -----------------------------
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Remove missing values
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# Sort by district and date
# -----------------------------
df = df.sort_values(["district", "date"]).reset_index(drop=True)

# -----------------------------
# Features
# -----------------------------
features = [
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed"
]

# -----------------------------
# Scale features
# -----------------------------
scaler = MinMaxScaler()

df[features] = scaler.fit_transform(df[features])

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# -----------------------------
# Create sequences
# -----------------------------
SEQUENCE_LENGTH = 30

X = []
y = []

districts = df["district"].unique()

for district in districts:

    district_df = df[df["district"]==district]

    values = district_df[features].values

    for i in range(len(values) - SEQUENCE_LENGTH):

        X.append(values[i:i+SEQUENCE_LENGTH])

        # -----------------------------
# Create sequences
# -----------------------------
SEQUENCE_LENGTH = 30

X = []
y = []

districts = df["district"].unique()

for district in districts:

    district_df = df[df["district"] == district]

    values = district_df[features].values

    for i in range(len(values) - SEQUENCE_LENGTH):

        # Last 30 days
        X.append(values[i:i+SEQUENCE_LENGTH])

        # Predict next day's
        # Temperature
        # Humidity
        # Rainfall
        # Wind Speed

        y.append(values[i+SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)


X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# -----------------------------
# Save processed arrays
# -----------------------------
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print("\nPreprocessing Completed Successfully!")