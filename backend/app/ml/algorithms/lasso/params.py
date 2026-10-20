from pydantic import BaseModel, ConfigDict, Field


class LassoParams(BaseModel):
    """Hyperparameters of scikit-learn's `Lasso` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    # scikit-learn defaults to 1.0, which zeroes every coefficient when the target varies by
    # about 1 (e.g. California Housing); 0.1 keeps a few, so the first run shows selection.
    alpha: float = Field(
        default=0.1,
        gt=0,
        le=1_000_000,
        title="Regularisation strength (α)",
        description=(
            "How hard the absolute size of the coefficients is penalised. Larger values set "
            "more coefficients exactly to 0, dropping those features. α is in the target's "
            "units: if every coefficient comes out 0, it is too large. Try values a factor of "
            "10 apart (0.001, 0.01, 0.1, 1)."
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
    # Above scikit-learn's 1000: one-hot columns make sweeps converge slowly (Penguins: ~2400).
    max_iter: int = Field(
        default=10_000,
        ge=10,
        le=100_000,
        title="Maximum passes",
        description=(
            "Lasso has no exact formula; it updates one coefficient at a time, sweeping over "
            "all of them until they stop changing. If the run used every pass, it stopped "
            "before converging: raise this (most needed with a very small α)."
        ),
        json_schema_extra={"level": "advanced"},
    )
