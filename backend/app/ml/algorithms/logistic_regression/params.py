from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LogisticRegressionParams(BaseModel):
    """Hyperparameters of scikit-learn's `LogisticRegression` that a learner can tune."""

    model_config = ConfigDict(extra="forbid")

    C: float = Field(
        default=1.0,
        gt=0,
        le=1_000_000,
        title="Inverse regularisation strength (C)",
        description=(
            "How much to trust the training data over keeping the weights small: the L2 "
            "penalty's strength is 1/C. Smaller C means a stronger penalty and smoother, more "
            "cautious probabilities; larger C fits the training data harder. Try values a "
            "factor of 10 apart (0.01, 0.1, 1, 10)."
        ),
        json_schema_extra={"level": "beginner"},
    )
    class_weight: Literal["balanced"] | None = Field(
        default=None,
        title="Class weights",
        description=(
            "None: every row counts the same. Balanced: rows of rare classes count more, in "
            "inverse proportion to how common their class is, so the model stops ignoring a "
            "minority class. Use it when one class is much rarer and missing it matters."
        ),
        json_schema_extra={"level": "beginner"},
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
    # Above scikit-learn's 100: unscaled features need ~3000 L-BFGS steps; scaled ones stop early.
    max_iter: int = Field(
        default=5000,
        ge=10,
        le=100_000,
        title="Maximum iterations",
        description=(
            "Logistic regression has no exact formula; the L-BFGS solver improves the weights "
            "step by step. Raise this if training warns that it did not converge (most "
            "likely with a large C)."
        ),
        json_schema_extra={"level": "advanced"},
    )
