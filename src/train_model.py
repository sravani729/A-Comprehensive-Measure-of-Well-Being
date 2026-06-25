import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("dataset/data.csv")

# -----------------------------
# KEEP ONLY SIMPLE COLUMNS (2021 data only)
# -----------------------------
df = df[[
    "Human Development Index (2021)",
    "Life Expectancy at Birth (2021)",
    "Mean Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)"
]]

# -----------------------------
# Rename columns (clean)
# -----------------------------
df.columns = ["HDI", "LifeExp", "Schooling", "GNI"]

# -----------------------------
# Remove missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# Split
# -----------------------------
X = df[["LifeExp", "Schooling", "GNI"]]
Y = df["HDI"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, Y_train)

# -----------------------------
# Save model
# -----------------------------
pickle.dump(model, open("models/model.pkl", "wb"))

print("Model trained successfully!")
print("Features used:", X.columns.tolist())