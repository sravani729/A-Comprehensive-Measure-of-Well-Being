import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("dataset/data.csv")

print("Dataset Loaded Successfully")
print(df.head())


print("\nDataset Columns:")
print(df.columns)


df = df.fillna(df.mean(numeric_only=True))


for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values handled successfully")


X = df.drop(df.columns[4], axis=1)


Y = df.iloc[:, 4]

print("\nIndependent Variables (X):")
print(X.head())

print("\nDependent Variable (Y):")
print(Y.head())


processed_data = df.copy()
processed_data.to_csv("dataset/processed_data.csv", index=False)

print("\nPreprocessing Completed Successfully")

# -----------------------------
# Step 4: Check Null Values
# -----------------------------

print("\nNull Values in X:")
print(X.isnull().sum())

# -----------------------------
# Step 5: Handle Null Values
# -----------------------------

# Fill missing values with column mean
X = X.fillna(X.mean())

print("\nAfter handling null values:")
print(X.isnull().sum())