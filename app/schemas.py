"""Validated request and response models for the inference API."""

from pydantic import BaseModel, ConfigDict, Field


class IrisFeatures(BaseModel):
    """Four measured Iris flower features, in centimeters."""

    model_config = ConfigDict(extra="forbid")

    sepal_length_cm: float = Field(gt=0, allow_inf_nan=False)
    sepal_width_cm: float = Field(gt=0, allow_inf_nan=False)
    petal_length_cm: float = Field(gt=0, allow_inf_nan=False)
    petal_width_cm: float = Field(gt=0, allow_inf_nan=False)

    def as_model_input(self) -> list[float]:
        """Return feature values in the training dataset's column order."""
        return [
            self.sepal_length_cm,
            self.sepal_width_cm,
            self.petal_length_cm,
            self.petal_width_cm,
        ]


class BatchPredictionRequest(BaseModel):
    """A bounded collection of feature rows to predict in one request."""

    model_config = ConfigDict(extra="forbid")

    features: list[IrisFeatures] = Field(min_length=1, max_length=128)


class IrisPrediction(BaseModel):
    """Predicted species and per-class probabilities."""

    class_index: int
    class_name: str
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Predictions corresponding to the submitted feature rows."""

    predictions: list[IrisPrediction]
