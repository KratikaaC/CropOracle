import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the processed dataset (must be created by step2_processing.py)
df = pd.read_csv("processed_crop_data.csv")

# Define Features (X) and Target Variable (y)
selected_features = ["P", "K", "temperature", "humidity"]
target_variable = "label"  # ← this was missing
X = df[selected_features]
y = df[target_variable]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Machine Learning Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")

# Save the trained model for later use
joblib.dump(model, "crop_recommendation_model.pkl")
print("✅ Model saved as 'crop_recommendation_model.pkl'")