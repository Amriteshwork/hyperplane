from pydantic import BaseModel, ConfigDict, Field


class LinearRegressionParams(BaseModel):
    """Hyperparameters of scikit-learn's `LinearRegression` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    fit_intercept: bool = Field(
        default=True,
        title="Fit intercept",
        description=(
            "Learn a constant offset (the bias, β₀). Turn off only if you know the target is "
            "zero when every feature is zero."
        ),
        json_schema_extra={"level": "beginner"},
    )
    positive: bool = Field(
        default=False,
        title="Positive coefficients only",
        description=(
            "Force every coefficient to be ≥ 0, e.g. when a feature can only increase the "
            "target. Solved with non-negative least squares instead of the normal equation."
        ),
        json_schema_extra={"level": "advanced"},
    )
