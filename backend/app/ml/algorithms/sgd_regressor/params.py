from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SGDRegressorParams(BaseModel):
    """Hyperparameters of scikit-learn's `SGDRegressor` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    loss: Literal["squared_error", "huber", "epsilon_insensitive"] = Field(
        default="squared_error",
        title="Loss",
        description=(
            "What each row's error costs. squared_error: ordinary least squares, big errors "
            "cost a lot. huber: squared for small errors, linear beyond ε, so outliers pull "
            "less. epsilon_insensitive: errors smaller than ε are free (the loss of SVR)."
        ),
        json_schema_extra={"level": "beginner"},
    )
    learning_rate: Literal["invscaling", "constant", "adaptive", "optimal"] = Field(
        default="invscaling",
        title="Learning-rate schedule",
        description=(
            "How the step size changes during training. invscaling: starts at η₀ and shrinks "
            "as η₀ / t^0.25. constant: always η₀. adaptive: η₀ until the loss stops improving, "
            "then divided by 5. optimal: a 1/t schedule chosen from α (ignores η₀)."
        ),
        json_schema_extra={"level": "beginner"},
    )
    # Below scikit-learn's 0.01: rows with extreme (scaled) features, e.g. California's room
    # counts, make 0.01 overshoot on some shuffles. ADR 0018.
    eta0: float = Field(
        default=0.001,
        gt=0,
        le=10,
        title="Initial learning rate (η₀)",
        description=(
            "The size of the first steps. Too large and the loss explodes (predictions become "
            "huge); too small and training needs many epochs. Try 0.0001, 0.001, 0.01."
        ),
        json_schema_extra={"level": "beginner"},
    )
    # Not 0: the optimal schedule divides by α; penalty=None is the way to switch it off.
    alpha: float = Field(
        default=0.0001,
        gt=0,
        le=100,
        title="Regularisation strength (α)",
        description=(
            "How hard large coefficients are penalised, as in Ridge (but per row, so the same "
            "penalty is α·n in Ridge's units). The default is weak; try 0.001 or 0.01 if the "
            "model overfits."
        ),
        json_schema_extra={"level": "beginner"},
    )
    penalty: Literal["l2", "l1", "elasticnet"] | None = Field(
        default="l2",
        title="Penalty",
        description=(
            "l2 shrinks every coefficient (Ridge); l1 sets some exactly to 0 (Lasso); "
            "elasticnet mixes them by the L1 ratio. None: no penalty."
        ),
        json_schema_extra={"level": "advanced"},
    )
    l1_ratio: float = Field(
        default=0.15,
        ge=0,
        le=1,
        title="L1 ratio (ρ)",
        description=(
            "Only for the elasticnet penalty: the share of L1 in the mix. 0 is pure L2, 1 is "
            "pure L1."
        ),
        json_schema_extra={"level": "advanced"},
    )
    epsilon: float = Field(
        default=0.1,
        ge=0,
        le=10,
        title="Epsilon (ε)",
        description=(
            "Only for the huber and epsilon_insensitive losses, in standard deviations of the "
            "target (it is standardised for training): where huber switches from squared to "
            "linear, or how big an error is ignored."
        ),
        json_schema_extra={"level": "advanced"},
    )
    max_iter: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        title="Maximum epochs",
        description=(
            "An epoch is one pass over the shuffled training rows, one update per row. "
            "Training usually stops much earlier, when the loss stops improving by tol."
        ),
        json_schema_extra={"level": "advanced"},
    )
    # Below scikit-learn's 0.001: with the small default step, 0.001 stops well short of the
    # minimum; on the standardised target, epochs' noise stays far below 1e-5. ADR 0018.
    tol: float = Field(
        default=0.00001,
        gt=0,
        le=1,
        title="Stopping tolerance",
        description=(
            "Stop when the average training loss has not improved by at least this much for "
            "5 epochs in a row. Smaller values train longer and land a little closer to the "
            "minimum."
        ),
        json_schema_extra={"level": "advanced"},
    )
    early_stopping: bool = Field(
        default=False,
        title="Early stopping",
        description=(
            "Hold out 10% of the training rows and stop when the score on them stops "
            "improving, instead of watching the training loss. Guards against overfitting."
        ),
        json_schema_extra={"level": "advanced"},
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
    # SGD shuffles the rows every epoch; a fixed seed makes the script reproduce the platform.
    random_state: int = Field(
        default=0,
        ge=0,
        le=2**32 - 1,
        title="Shuffle seed",
        description=(
            "Seeds the order the rows are visited in. A different seed gives slightly "
            "different coefficients: SGD's answer depends on the path it took."
        ),
        json_schema_extra={"level": "advanced"},
    )
