from .data_loader import load_faculty_data
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import os
from joblib import dump

# Load data
df = load_faculty_data().dropna()

# Features & target
X = df.drop(columns=["student_evaluation"])
y = df["student_evaluation"]

# Preprocessing
categorical = ["college","position","semester"]
numeric = ["workload_units", "num_preparations"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)

# TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

import os

# Save to the same directory as train.py
save_dir = os.path.dirname(__file__)
os.makedirs(save_dir, exist_ok=True)

dump(preprocessor, os.path.join(save_dir, "preprocessor.joblib"))

model.save(os.path.join(save_dir, "faculty_model.keras"))  # ✅ RECOMMENDED
print("✅ Training complete and model saved!")


