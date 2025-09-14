import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------
try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # Pandas DataFrame version
    df.rename(columns={"target": "species"}, inplace=True)
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    # Data info
    print("Dataset Info:")
    print(df.info(), "\n")

    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (drop NA if any)
    df = df.dropna()

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------
print("Basic statistics:")
print(df.describe(), "\n")

# Grouping example: average petal length per species
grouped = df.groupby("species")["petal length (cm)"].mean()
print("Average petal length by species:")
print(grouped, "\n")

# Insight: pattern
print("Insights:")
print("- Virginica species has the largest petal length on average.")
print("- Setosa species generally has the smallest measurements.\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------
sns.set(style="whitegrid")

# 1. Line chart (simulate a trend by using index as x-axis for petal length)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length", color="green")
plt.title("Line Chart: Petal Length Trend (Index Order)")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, palette="viridis")
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: distribution of sepal length
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot: sepal length vs petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
