"""FastAPI application exposing Iris model predictions."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request

from app.model import DEFAULT_MODEL_PATH, IrisModel, load_model
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    IrisFeatures,
    IrisPrediction,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model once before serving requests."""
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    application.state.model = load_model(model_path)
    try:
        yield
    finally:
        application.state.model = None


app = FastAPI(
    title="Production-Ready ML Inference API",
    description="A validated HTTP API for predicting Iris flower species.",
    version="1.0.0",
    lifespan=lifespan,
)


def _ready_model(request: Request) -> IrisModel:
    """Return the loaded model or report that the service is not ready."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return model


@app.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    """Indicate that the API process is running."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def readiness(request: Request) -> dict[str, str]:
    """Indicate that the model has loaded and the API can serve predictions."""
    _ready_model(request)
    return {"status": "ready"}


@app.post("/api/v1/predict", response_model=IrisPrediction, tags=["inference"])
def predict(features: IrisFeatures, request: Request) -> dict[str, object]:
    """Predict the Iris species for a single flower."""
    model = _ready_model(request)
    return model.predict([features.as_model_input()])[0]


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
)
def predict_batch(
    batch: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    """Predict the Iris species for up to 128 flowers in one request."""
    model = _ready_model(request)
    predictions = model.predict(
        [features.as_model_input() for features in batch.features]
    )
    return BatchPredictionResponse(predictions=predictions)
