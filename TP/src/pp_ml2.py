import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("salary_train.csv")
test_data = pd.read_csv("salary_test.csv")

X_train = train_data.drop("salary", axis=1)
y_train = train_data["salary"]

X_test = test_data.drop("salary", axis=1)
y_test = test_data["salary"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test["years_of_experience"], y_test)
plt.plot(X_test["years_of_experience"], y_pred, color="red")
plt.show()
