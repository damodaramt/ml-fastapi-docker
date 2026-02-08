import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("data/data.csv")

X = data[['experience']]
y = data['salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model/salary_model.pkl")

print("Model trained and saved successfully!")
