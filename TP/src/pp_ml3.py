import matplotlib.pyplot as plt
import pandas as pd
from pyexpat import features

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_csv("customer_train.csv")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(raw_data)

data = pd.DataFrame(features_scaled, columns=raw_data.columns)

k = 4
model = KMeans(n_clusters=k, random_state=42)

data["cluster"] = model.fit_predict(data)

plt.figure(figsize=(15, 5))

# First subplot
plt.subplot(1, 3, 1)
plt.scatter(data["annual_income"], data["spending_score"], c=data["cluster"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending Score")

# Second subplot
plt.subplot(1, 3, 2)
plt.scatter(data["age"], data["spending_score"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Age vs Spending Score")

# Third subplot
plt.subplot(1, 3, 3)
plt.scatter(data["age"], data["annual_income"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.title("Age vs Annual Income")

plt.show()
