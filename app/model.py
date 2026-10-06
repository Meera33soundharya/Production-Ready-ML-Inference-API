"""Iris classifier training, persistence, and prediction."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL_PATH = Path("models/iris_model.joblib")


@dataclass(frozen=True)
class IrisModel:
    """A fitted classifier and the names corresponding to its class indices."""

    estimator: Pipeline
    target_names: tuple[str, ...]

    def predict(self, features: list[list[float]]) -> list[dict[str, object]]:
        """Predict Iris classes and probabilities for one or more feature rows."""
        values: NDArray[np.float64] = np.asarray(features, dtype=np.float64)
        class_indices = self.estimator.predict(values)
        class_probabilities = self.estimator.predict_proba(values)

        predictions: list[dict[str, object]] = []
        for class_index, probabilities in zip(
            class_indices, class_probabilities, strict=True
        ):
            predicted_index = int(class_index)
            probabilities_by_name = {
                self.target_names[int(index)]: float(probability)
                for index, probability in zip(
                    self.estimator.classes_, probabilities, strict=True
                )
            }
            predictions.append(
                {
                    "class_index": predicted_index,
                    "class_name": self.target_names[predicted_index],
                    "probabilities": probabilities_by_name,
                }
            )
        return predictions


def train_model() -> IrisModel:
    """Fit a deterministic classifier using scikit-learn's bundled Iris data."""
    dataset = load_iris()
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    estimator.fit(dataset.data, dataset.target)
    return IrisModel(
        estimator=estimator,
        target_names=tuple(str(name) for name in dataset.target_names),
    )


def save_model(model: IrisModel, output_path: Path) -> None:
    """Persist a model artifact atomically at the requested path."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_model(model_path: Path) -> IrisModel:
    """Load a trusted, previously generated Iris model artifact."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python -m app.train` before starting the API."
        )

    model = joblib.load(model_path)
    if not isinstance(model, IrisModel):
        raise ValueError(f"Model artifact at {model_path} has an invalid format.")
    return model
