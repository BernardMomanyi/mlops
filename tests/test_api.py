
from fastapi.testclient import TestClient

from src.mlops_platform.api import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "MLOps Prediction API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "features": [
            17.99, 10.38, 122.8, 1001.0, 0.1184,
            0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399,
            0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622,
            0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["predicted_class"] in ["malignant", "benign"]
    assert result["model_run_id"] == "61b032cd56a846f2ab2848ee39594a7b"

def test_prediction_rejects_wrong_feature_count():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 422
