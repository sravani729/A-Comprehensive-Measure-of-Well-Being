import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/data.csv")

# Use only first 20 rows
data1 = df.head(20)


print("Unique Countries:")
print(data1["Country"].unique())


plt.figure(figsize=(8,5))
sns.stripplot(x="Mean Years of Schooling", y="HDI", data=data1)
plt.title("Mean Years of Schooling vs HDI")
plt.show()


plt.figure(figsize=(8,5))
sns.stripplot(x="Life Expectancy", y="HDI", data=data1)
plt.title("Life Expectancy vs HDI")
plt.show()

plt.figure(figsize=(10,6))

corr = data1.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()