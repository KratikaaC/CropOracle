import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the processed dataset
df = pd.read_csv("processed_crop_data.csv")  # Make sure this file exists from Step 2

# Step 2: Define Features (X) and Target (y)
selected_features = ['P', 'K', 'temperature', 'humidity']
X = df[selected_features]
y = df['label']

# Step 3: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Load the Trained Model
model = joblib.load("crop_recommendation_model.pkl")  # Make sure this was created by Step 3

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# Classification Report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
