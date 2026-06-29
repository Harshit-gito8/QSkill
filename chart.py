import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("StudentsPerformance.csv")
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.describe())

print("Average Math Score:",df["math score"].mean())
print("Average Reading Score:",df["reading score"].mean())
print("Average Writing Score:",df["writing score"].mean())

print(df["test preparation course"].value_counts())

print(df.nlargest(10, "math score"))



df.groupby("gender")["math score"].mean().plot(kind="bar")
plt.title("Average Math Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Score")
plt.show()

plt.scatter(df["math score"], df["reading score"])
plt.title("Math Score vs Reading Score")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.show()


corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")
plt.show()