from pydantic import BaseModel, ConfigDict, Field


class RidgeParams(BaseModel):
    """Hyperparameters of scikit-learn's `Ridge` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        default=1.0,
        gt=0,
        le=1_000_000,
        title="Regularisation strength (α)",
        description=(
            "How hard large coefficients are penalised. Near 0 behaves like Linear Regression; "
            "larger values shrink every coefficient towards 0, trading a little bias for "
            "stability. Try values a factor of 10 apart (0.1, 1, 10, 100)."
        ),
        json_schema_extra={"level": "beginner"},
    )
    fit_intercept: bool = Field(
        default=True,
        title="Fit intercept",
        description=(
            "Learn a constant offset (the bias, β₀). It is never penalised. Turn off only if "
            "you know the target is zero when every feature is zero."
        ),
        json_schema_extra={"level": "advanced"},
    )
