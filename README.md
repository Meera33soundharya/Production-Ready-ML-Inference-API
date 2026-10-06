# Production-Ready ML Inference API

A runnable machine-learning inference API built with FastAPI and scikit-learn. It trains a small Iris flower classifier from scikit-learn's bundled dataset, saves the model locally, and serves predictions over HTTP.

## Features

- Single and batch predictions with input validation.
- Liveness and readiness health endpoints.
- Model trained from a built-in dataset; no dataset download is needed.
- Model artifact created locally and excluded from Git.
- Automated API tests.
- Docker image that builds its model artifact during image creation and runs as a non-root user.

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── model.py         # Model training, persistence, and inference
│   └── schemas.py       # Validated request and response models
├── models/              # Generated model artifacts (not committed)
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

## Requirements

- Python 3.11 or newer
- pip

## Run locally on Windows

From the project directory, create and activate a virtual environment, install the development dependencies, train the model, and start the API:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m app.train
python -m uvicorn app.main:app --reload
```

The API will be available at <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

The training command writes `models/iris_model.joblib`. To use a different artifact path, set `MODEL_PATH` before training and when starting the API:

```powershell
$env:MODEL_PATH = "models\iris_model.joblib"
python -m app.train
python -m uvicorn app.main:app --reload
```

Only load model artifacts from trusted sources. Joblib model files can execute code when loaded.

## API

### `GET /health/live`

Returns `200` when the process is running.

### `GET /health/ready`

Returns `200` when the model is loaded. Returns `503` if the model is not ready.

### `POST /api/v1/predict`

Request:

```json
{
  "sepal_length_cm": 5.1,
  "sepal_width_cm": 3.5,
  "petal_length_cm": 1.4,
  "petal_width_cm": 0.2
}
```

Response:

```json
{
  "class_index": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.0
  }
}
```

The probabilities shown above are illustrative; actual values are returned by the trained model. All four measurements must be finite, positive numbers, and unknown request fields are rejected.

### `POST /api/v1/predict/batch`

Pass between 1 and 128 feature objects in a `features` array:

```json
{
  "features": [
    {
      "sepal_length_cm": 5.1,
      "sepal_width_cm": 3.5,
      "petal_length_cm": 1.4,
      "petal_width_cm": 0.2
    }
  ]
}
```

The response contains a `predictions` array with one prediction per input object.

## Run with Docker

Build and start the service:

```powershell
docker build -t production-ready-ml-inference-api .
docker run --rm -p 8000:8000 production-ready-ml-inference-api
```

The image installs the production dependencies and trains the bundled Iris model during the build. Once it is running, visit <http://127.0.0.1:8000/docs>.

## Run tests

```powershell
python -m pytest
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `models/iris_model.joblib` | Path to the model artifact used by the API and training command. |

## Notes

The Iris model is a small demonstration model, not a domain-specific or independently validated production model. Replace it with a model trained and evaluated for your use case before serving real predictions. Protect the service with authentication, rate limiting, and deployment-specific network controls before exposing it publicly.

No software license has been specified yet.
