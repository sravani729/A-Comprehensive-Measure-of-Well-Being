import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("dataset/data.csv")

# -----------------------------
# Step 2: Split X and Y
# -----------------------------
X = df.drop(df.columns[4], axis=1)
Y = df.iloc[:, 4]

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
X = X.fillna(X.mean(numeric_only=True))

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 5: Load Trained Model
# -----------------------------
model = pickle.load(open("models/model.pkl", "rb"))

# -----------------------------
# Step 6: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 7: Print Predictions
# -----------------------------
print("Actual Values (y_test):")
print(Y_test.values)

print("\nPredicted Values (y_pred):")
print(y_pred)

# -----------------------------
# Step 8: R-Squared Value
# -----------------------------
r2 = r2_score(Y_test, y_pred)
print("\nR-Squared Value:", r2)

# -----------------------------
# Step 9: Mean Squared Error
# -----------------------------
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)

# -----------------------------
# Step 10: Test with Fewer Inputs
# -----------------------------
print("\nTesting first 3 samples only:")
print("Actual:", Y_test.values[:3])
print("Predicted:", y_pred[:3])