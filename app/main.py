from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the ML model
model = pickle.load(open('app/model.pkl', 'rb'))

# Define class label mappings
class_names = {
    0: "Healthy",
    1: "Diabetes",
    2: "Heart Disease"
}

# Define the input structure
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Disease Prediction API is running using FastAPI"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    predicted_class = class_names.get(int(prediction), "Unknown")
    return {
        "prediction": predicted_class
    }
