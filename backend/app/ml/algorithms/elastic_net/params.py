from pydantic import BaseModel, ConfigDict, Field


class ElasticNetParams(BaseModel):
    """Hyperparameters of scikit-learn's `ElasticNet` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    # As for Lasso: scikit-learn's 1.0 zeroes every coefficient on targets that vary by ~1.
    alpha: float = Field(
        default=0.1,
        gt=0,
        le=1_000_000,
        title="Regularisation strength (α)",
        description=(
            "The total penalty strength, shared between the L1 and L2 parts by the L1 ratio. "
            "Larger values shrink every coefficient and set more of them to 0. α is in the "
            "target's units: if every coefficient comes out 0, it is too large. Try values a "
            "factor of 10 apart (0.001, 0.01, 0.1, 1)."
        ),
        json_schema_extra={"level": "beginner"},
    )
    # Below ~0.01 coordinate descent is unreliable (scikit-learn warns); use Ridge for pure L2.
    l1_ratio: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        title="L1 ratio (ρ)",
        description=(
            "How the penalty is split: 1 is pure L1 (Lasso: drops features), near 0 is almost "
            "pure L2 (Ridge: shrinks but keeps them). In between, it drops unhelpful features "
            "but keeps groups of correlated ones together. Try 0.1, 0.5, 0.9."
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
    # Above scikit-learn's 1000: one-hot columns make sweeps converge slowly (as for Lasso).
    max_iter: int = Field(
        default=10_000,
        ge=10,
        le=100_000,
        title="Maximum passes",
        description=(
            "Elastic Net has no exact formula; it updates one coefficient at a time, sweeping "
            "over all of them until they stop changing. If the run used every pass, it stopped "
            "before converging: raise this (most needed with a very small α)."
        ),
        json_schema_extra={"level": "advanced"},
    )
