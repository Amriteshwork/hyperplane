from pydantic import BaseModel, ConfigDict, Field


class GaussianNBParams(BaseModel):
    """Hyperparameters of scikit-learn's `GaussianNB` that a learner can tune.

    `priors` is left out: it needs one number per class, which the form can't know in advance,
    so the priors always come from the training rows.
    """

    model_config = ConfigDict(extra="forbid")

    # Not scikit-learn's 1e-9: a one-hot column constant within a class then rules that class out
    # for any other row (Penguins: 0.72 cross-validated). 0.01 gives 0.92 and leaves the numeric
    # samples within 0.002 (ADR 0026).
    var_smoothing: float = Field(
        default=0.01,
        gt=0,
        le=10,
        title="Variance smoothing",
        description=(
            "A share of the largest feature variance, added to every variance so that no bell "
            "curve is too narrow. Too small, and one surprising value can rule a class out; too "
            "large, and the curves flatten until features say little. The results show a "
            "cross-validated score per value to help you choose."
        ),
        json_schema_extra={"level": "beginner"},
    )
