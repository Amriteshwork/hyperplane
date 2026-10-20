from pydantic import BaseModel, ConfigDict, Field


class PolynomialRegressionParams(BaseModel):
    """Hyperparameters of `PolynomialFeatures` + `LinearRegression` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    degree: int = Field(
        default=2,
        ge=1,
        le=5,
        title="Degree",
        description=(
            "Highest power of the features (and of their products) the model may use. 1 is "
            "plain linear regression; higher degrees bend more, and overfit more easily."
        ),
        json_schema_extra={"level": "beginner"},
    )
    interaction_only: bool = Field(
        default=False,
        title="Interactions only",
        description=(
            "Keep only products of different features (x₁·x₂), not powers of one feature "
            "(x₁²). Models features that amplify each other without curving each one."
        ),
        json_schema_extra={"level": "advanced"},
    )
    fit_intercept: bool = Field(
        default=True,
        title="Fit intercept",
        description=(
            "Learn a constant offset (the bias, β₀). Turn off only if you know the target is "
            "zero when every feature is zero."
        ),
        json_schema_extra={"level": "advanced"},
    )
