"""What the MLP packs report about a fitted network: its layers and how much it uses each input."""

from collections.abc import Sequence

import numpy as np
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron

from app.ml.base import ImportanceTable, Network


def mlp_network(
    model: BaseMultilayerPerceptron, feature_names: list[str], output_names: Sequence[str]
) -> Network:
    return Network(
        input_names=feature_names,
        weights=model.coefs_,
        activation=model.activation,
        output_activation=model.out_activation_,
        output_names=output_names,
    )


def input_weight_importance(
    model: BaseMultilayerPerceptron, feature_names: list[str]
) -> ImportanceTable:
    """Mean |weight| from each input into the first hidden layer.

    Inputs are standardised, so these compare; it says what the first layer listens to, not
    how a feature moves the prediction (that passes through every later layer).
    """
    strength = np.abs(model.coefs_[0]).mean(axis=1)
    return {name: float(v) for name, v in zip(feature_names, strength, strict=True)}
