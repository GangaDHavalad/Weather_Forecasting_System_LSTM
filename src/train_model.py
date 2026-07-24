import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Create folders
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))

# 4 outputs:
# Temperature
# Humidity
# Rainfall
# Wind Speed
model.add(Dense(4))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/lstm_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -----------------------------
# Save Model
# -----------------------------
model.save("models/lstm_model.keras")

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("graphs/loss_curve.png")

plt.show()

print("\nTraining Completed Successfully!")