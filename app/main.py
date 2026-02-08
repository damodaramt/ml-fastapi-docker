from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML FastAPI Server")

# Load trained model
model = joblib.load("model/salary_model.pkl")

# Define input schema
class PredictionInput(BaseModel):
    experience: float

@app.get("/")
def home():
    return {"status": "FastAPI ML API running"}

@app.post("/predict")
def predict(data: PredictionInput):
    salary = model.predict(np.array([[data.experience]]))
    return {"predicted_salary": int(salary[0])}
