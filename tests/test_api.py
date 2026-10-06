"""Tests for the Iris inference API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import save_model, train_model


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    """Run the API with a fresh local model artifact."""
    model_path = tmp_path / "iris_model.joblib"
    save_model(train_model(), model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_predict_returns_class_and_probabilities(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length_cm": 5.1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["class_index"] == 0
    assert result["class_name"] == "setosa"
    assert set(result["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_predict_rejects_invalid_or_unexpected_fields(client: TestClient) -> None:
    invalid_value = client.post(
        "/api/v1/predict",
        json={
            "sepal_length_cm": -1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
        },
    )
    unexpected_field = client.post(
        "/api/v1/predict",
        json={
            "sepal_length_cm": 5.1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
            "extra": True,
        },
    )

    assert invalid_value.status_code == 422
    assert unexpected_field.status_code == 422


def test_batch_predict_returns_one_result_per_input(client: TestClient) -> None:
    sample = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post(
        "/api/v1/predict/batch",
        json={"features": [sample, sample]},
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize("features", [[], [{}] * 129])
def test_batch_predict_rejects_out_of_bounds_batch(
    client: TestClient, features: list[dict[str, float]]
) -> None:
    response = client.post("/api/v1/predict/batch", json={"features": features})

    assert response.status_code == 422


def test_missing_model_fails_startup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))

    with pytest.raises(FileNotFoundError, match="python -m app.train"):
        with TestClient(app):
            pass
