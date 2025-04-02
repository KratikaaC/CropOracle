import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset (assumes file is in the same folder as this script)
file_path = "Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Step 1: Select the four chosen variables
# Based on correlation analysis:
# P, K = nutrients | temperature, humidity = climate
selected_features = ["P", "K", "temperature", "humidity"]
target_variable = "label"

# Step 2: Create a new DataFrame with selected features
df_selected = df[selected_features + [target_variable]].copy()

# Step 3: Drop missing values if any
df_selected = df_selected.dropna()

# Step 4: Save the cleaned data to a new CSV file
df_selected.to_csv("processed_crop_data.csv", index=False)
print("✅ Step 2 Complete: Processed data saved as 'processed_crop_data.csv'")
