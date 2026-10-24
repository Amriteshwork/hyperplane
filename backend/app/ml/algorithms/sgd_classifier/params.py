from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SGDClassifierParams(BaseModel):
    """Hyperparameters of scikit-learn's `SGDClassifier` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    # Not scikit-learn's hinge: log_loss gives probabilities (and so log-loss and ROC-AUC),
    # and is Logistic Regression's objective, which the learner has just met.
    loss: Literal["log_loss", "hinge", "modified_huber"] = Field(
        default="log_loss",
        title="Loss",
        description=(
            "log_loss: logistic regression, with class probabilities. hinge: a linear SVM, "
            "which only cares about rows near or on the wrong side of the boundary (no "
            "probabilities). modified_huber: a smoothed hinge that also gives probabilities "
            "and tolerates outliers."
        ),
        json_schema_extra={"level": "beginner"},
    )
    learning_rate: Literal["optimal", "invscaling", "constant", "adaptive"] = Field(
        default="optimal",
        title="Learning-rate schedule",
        description=(
            "How the step size changes during training. optimal: a 1/t schedule chosen from "
            "α (ignores η₀). invscaling: η₀ / √t. constant: always η₀. adaptive: η₀ until the "
            "loss stops improving, then divided by 5."
        ),
        json_schema_extra={"level": "beginner"},
    )
    eta0: float = Field(
        default=0.01,
        gt=0,
        le=10,
        title="Initial learning rate (η₀)",
        description=(
            "The size of the first steps, for every schedule except optimal. Too large and "
            "the weights jump around; too small and training needs many epochs. Try 0.001, "
            "0.01, 0.1."
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
            "How hard large weights are penalised, per row: Logistic Regression's C is "
            "1 / (α·n). Larger α means smaller weights and more cautious probabilities; it "
            "also sets the optimal schedule's step sizes."
        ),
        json_schema_extra={"level": "beginner"},
    )
    class_weight: Literal["balanced"] | None = Field(
        default=None,
        title="Class weights",
        description=(
            "None: every row counts the same. Balanced: rows of rare classes count more, in "
            "inverse proportion to how common their class is, so the model stops ignoring a "
            "minority class."
        ),
        json_schema_extra={"level": "beginner"},
    )
    penalty: Literal["l2", "l1", "elasticnet"] | None = Field(
        default="l2",
        title="Penalty",
        description=(
            "l2 shrinks every weight; l1 sets some exactly to 0 (drops features); elasticnet "
            "mixes them by the L1 ratio. None: no penalty."
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
    tol: float = Field(
        default=0.001,
        gt=0,
        le=1,
        title="Stopping tolerance",
        description=(
            "Stop when the average training loss has not improved by at least this much for "
            "5 epochs in a row. Smaller values train longer and land closer to the minimum."
        ),
        json_schema_extra={"level": "advanced"},
    )
    early_stopping: bool = Field(
        default=False,
        title="Early stopping",
        description=(
            "Hold out 10% of the training rows and stop when the accuracy on them stops "
            "improving, instead of watching the training loss. Guards against overfitting."
        ),
        json_schema_extra={"level": "advanced"},
    )
    fit_intercept: bool = Field(
        default=True,
        title="Fit intercept",
        description=(
            "Learn a constant offset per class (the bias). It is never penalised. Turn off "
            "only if you know every class is equally likely when all features are zero."
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
            "different weights: SGD's answer depends on the path it took."
        ),
        json_schema_extra={"level": "advanced"},
    )
