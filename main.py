from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_decision_tree_pipeline.joblib")

app = FastAPI(title="Iris Classifier API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris classifier is running!"}

@app.post("/predict")
def predict_species(data: IrisInput):
    X = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    pred = model.predict(X)[0]
    return {"prediction": str(pred)}
