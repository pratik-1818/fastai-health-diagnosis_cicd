from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Disease Prediction API is running using FastAPI"

def test_predict():
    # Example input (must match what your model expects)
    sample_input = {"features": [1, 2, 3, 4]}  # Adjust according to your model input
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
