
from pathlib import Path
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / "mlflow.db"
DATABASE_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

EXPERIMENT_NAME = "mlops-platform-classification"
PRODUCTION_MODEL_PATH = os.getenv("PRODUCTION_MODEL_PATH")

mlflow.set_tracking_uri(DATABASE_URI)


# --------------------------------------------------
# API application
# --------------------------------------------------

app = FastAPI(
    title="MLOps Prediction API",
    description="API for the breast cancer classification model.",
    version="1.1.0",
)


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features from the dataset.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    predicted_class: str
    model_run_id: str


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():
    """Load the production model or the latest MLflow run."""

    # Docker mode: load the packaged model directly.
    if PRODUCTION_MODEL_PATH:
        model_path = Path(PRODUCTION_MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production model not found: {model_path}"
            )

        loaded_model = mlflow.pyfunc.load_model(str(model_path))

        return loaded_model, "61b032cd56a846f2ab2848ee39594a7b"

    # Local mode: discover the latest model from MLflow.
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )

    if runs.empty:
        raise RuntimeError("No MLflow training runs were found.")

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        model_uri = f"runs:/{run_id}/model"

        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            return loaded_model, run_id
        except Exception:
            continue

    raise RuntimeError("Unable to load a trained MLflow model.")


try:
    model, model_run_id = load_model()
    model_load_error = None
except Exception as exc:
    model = None
    model_run_id = None
    model_load_error = str(exc)


# --------------------------------------------------
# Feature names
# --------------------------------------------------

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "MLOps Prediction API is running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_load_error}",
        )

    input_data = pd.DataFrame(
        [request.features],
        columns=FEATURE_NAMES,
    )

    try:
        prediction = int(model.predict(input_data)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        predicted_class="malignant" if prediction == 0 else "benign",
        model_run_id=model_run_id,
    )
