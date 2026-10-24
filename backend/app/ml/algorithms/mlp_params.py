"""Hyperparameters shared by the MLP Classifier and MLP Regressor packs (ADR 0019)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MLPParams(BaseModel):
    """Hyperparameters of scikit-learn's `MLPClassifier` / `MLPRegressor` that a learner can tune.

    The network's shape is two numbers (layers × units per layer) rather than scikit-learn's
    `hidden_layer_sizes` tuple, so the form needs no list editor; `mlp_kwargs` maps it back.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_layers: int = Field(
        default=1,
        ge=1,
        le=5,
        title="Hidden layers",
        description=(
            "How many layers of neurons sit between the inputs and the output. One is enough "
            "for most tabular data; more layers can build more complex features but train "
            "slower and overfit more easily."
        ),
        json_schema_extra={"level": "beginner"},
    )
    hidden_units: int = Field(
        default=100,
        ge=1,
        le=1024,
        title="Neurons per layer",
        description=(
            "The width of each hidden layer. More neurons can fit more complicated patterns, "
            "and need more data (or a larger α) not to memorise the training rows."
        ),
        json_schema_extra={"level": "beginner"},
    )
    activation: Literal["relu", "tanh", "logistic", "identity"] = Field(
        default="relu",
        title="Activation",
        description=(
            "The bend each hidden neuron applies to its weighted sum. relu: max(0, z), the "
            "modern default. tanh and logistic: S-shaped curves. identity: no bend, which "
            "makes the whole network a linear model."
        ),
        json_schema_extra={"level": "beginner"},
    )
    alpha: float = Field(
        default=0.0001,
        ge=0,
        le=100,
        title="Regularisation strength (α)",
        description=(
            "An L2 penalty on every weight, as in Ridge. Raise it (try 0.01, 0.1, 1) when the "
            "training score is much better than the test score."
        ),
        json_schema_extra={"level": "beginner"},
    )
    learning_rate_init: float = Field(
        default=0.001,
        gt=0,
        le=1,
        title="Learning rate",
        description=(
            "The step size of the optimiser (adam or sgd). Too large and the loss jumps "
            "around or explodes; too small and training needs many epochs."
        ),
        json_schema_extra={"level": "beginner"},
    )
    solver: Literal["adam", "sgd", "lbfgs"] = Field(
        default="adam",
        title="Solver",
        description=(
            "adam: mini-batch steps with a per-weight adaptive step size, robust on most "
            "data. sgd: plain mini-batch SGD with momentum. lbfgs: full-batch quasi-Newton, "
            "often best on small datasets (no mini-batches, epochs or loss curve)."
        ),
        json_schema_extra={"level": "advanced"},
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        title="Batch size",
        description=(
            "Rows per gradient step for adam and sgd. Empty: min(200, number of rows). "
            "Smaller batches give more, noisier steps per epoch."
        ),
        json_schema_extra={"level": "advanced"},
    )
    # Not scikit-learn's 200: with a few batches per epoch, small samples can need more.
    max_iter: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        title="Maximum epochs",
        description=(
            "An epoch is one pass over the shuffled training rows (for lbfgs: one solver "
            "iteration). Training usually stops earlier, once the loss stops improving."
        ),
        json_schema_extra={"level": "advanced"},
    )
    # Not scikit-learn's 1e-4 (ADR 0019): on small samples the loss keeps creeping down as
    # the network memorises noise, so training ran to max_iter, warned and overfitted.
    tol: float = Field(
        default=0.001,
        gt=0,
        le=1,
        title="Stopping tolerance",
        description=(
            "Stop when the training loss (or, with early stopping, the validation score) has "
            "not improved by at least this much for more than 'Patience' epochs in a row."
        ),
        json_schema_extra={"level": "advanced"},
    )
    n_iter_no_change: int = Field(
        default=10,
        ge=1,
        le=1000,
        title="Patience (epochs)",
        description=(
            "How many epochs without improvement to tolerate before stopping. The loss of a "
            "mini-batch optimiser is noisy, so one bad epoch is no reason to stop."
        ),
        json_schema_extra={"level": "advanced"},
    )
    early_stopping: bool = Field(
        default=False,
        title="Early stopping",
        description=(
            "Hold out part of the training rows, stop when the score on them (accuracy or "
            "R²) stops improving, and keep the weights from the best epoch. Guards against "
            "overfitting on larger datasets; on a few hundred rows the held-out set is too "
            "small to judge by."
        ),
        json_schema_extra={"level": "advanced"},
    )
    validation_fraction: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        title="Validation fraction",
        description="Only with early stopping: the share of training rows held out to judge it.",
        json_schema_extra={"level": "advanced"},
    )
    # Initial weights and batch order are random; a fixed seed lets the script reproduce the run.
    random_state: int = Field(
        default=0,
        ge=0,
        le=2**32 - 1,
        title="Random seed",
        description=(
            "Seeds the starting weights and the order rows are visited in. Networks are not "
            "convex: another seed can end in a different (often similarly good) solution."
        ),
        json_schema_extra={"level": "advanced"},
    )


def mlp_kwargs(params: MLPParams) -> dict[str, Any]:
    """The scikit-learn constructor arguments; `train.py.j2` renders the same ones."""
    kwargs = params.model_dump(exclude={"hidden_layers", "hidden_units"})
    kwargs["batch_size"] = "auto" if params.batch_size is None else params.batch_size
    return {"hidden_layer_sizes": [params.hidden_units] * params.hidden_layers, **kwargs}
