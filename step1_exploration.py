import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (assumes the file is in the same folder as this script)
df = pd.read_csv("Crop_recommendation.csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Column names
print("\nColumn Names:", df.columns.tolist())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Show basic statistics
print("\nDataset Statistics:")
print(df.describe())

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix.round(2))  # Rounds values to 2 decimal places
