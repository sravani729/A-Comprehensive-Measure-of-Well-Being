import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("dataset/data.csv")


X = df.drop(df.columns[4], axis=1)
Y = df.iloc[:, 4]


X = X.fillna(X.mean(numeric_only=True))


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)


model = pickle.load(open("models/model.pkl", "rb"))


y_pred = model.predict(X_test)


print("Actual Values (y_test):")
print(Y_test.values)

print("\nPredicted Values (y_pred):")
print(y_pred)


r2 = r2_score(Y_test, y_pred)
print("\nR-Squared Value:", r2)


mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)


print("\nTesting first 3 samples only:")
print("Actual:", Y_test.values[:3])
print("Predicted:", y_pred[:3])