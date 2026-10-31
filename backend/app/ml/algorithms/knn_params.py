"""Hyperparameters shared by the KNN Classifier and KNN Regressor packs (ADR 0025)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KNNParams(BaseModel):
    """Hyperparameters of scikit-learn's `KNeighborsClassifier` / `KNeighborsRegressor`.

    `algorithm` and `leaf_size` are left out: they change how fast neighbours are found, not
    which ones, so a form field for them would change nothing a learner can see.
    """

    model_config = ConfigDict(extra="forbid")

    n_neighbors: int = Field(
        default=5,
        ge=1,
        le=1000,
        title="Neighbours (k)",
        description=(
            "How many of the closest training rows get a say. k = 1 copies the single nearest "
            "row, noise included; a large k averages over a wide area and blurs real detail. "
            "It can't exceed the number of training rows. The results show a cross-validated "
            "score per k to help you choose."
        ),
        json_schema_extra={"level": "beginner"},
    )
    weights: Literal["uniform", "distance"] = Field(
        default="uniform",
        title="Neighbour weights",
        description=(
            "uniform: every neighbour counts the same. distance: a neighbour counts in "
            "proportion to 1/distance, so closer rows matter more and a larger k does less harm."
        ),
        json_schema_extra={"level": "beginner"},
    )
    # scikit-learn's default is minkowski with p=2, which is this metric by another name.
    # No chebyshev: scikit-learn rejects it on the sparse matrices many one-hot columns produce.
    metric: Literal["euclidean", "manhattan", "cosine"] = Field(
        default="euclidean",
        title="Distance metric",
        description=(
            "What 'close' means. euclidean: the straight-line distance. manhattan: the sum of "
            "the per-feature differences, less swayed by one large difference. cosine: the angle "
            "between the rows, ignoring their length."
        ),
        json_schema_extra={"level": "advanced"},
    )
