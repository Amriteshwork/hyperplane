"""A fitted network as a diagram: layer sizes, plus the weights among the few units drawn."""

import numpy as np

from app.ml.base import Network
from app.ml.evaluation.models import NetworkLayer, NetworkSummary
from app.ml.evaluation.numbers import round6

#: Units drawn per layer; wider layers show a count instead (payloads stay small at 1024 units).
MAX_SHOWN = 8


def summarise_network(network: Network) -> NetworkSummary:
    weights = [np.asarray(w, dtype=float) for w in network.weights]
    # Inputs: the ones the first layer weighs most. Hidden units are interchangeable, so the
    # first few are as telling as any; outputs are few.
    strength = np.abs(weights[0]).mean(axis=1)
    shown = [np.argsort(-strength, kind="stable")[:MAX_SHOWN]]
    shown += [np.arange(min(w.shape[1], MAX_SHOWN)) for w in weights]

    hidden = len(weights) - 1
    layers = [
        NetworkLayer(
            kind="input",
            units=weights[0].shape[0],
            activation=None,
            shown=[network.input_names[i] for i in shown[0]],
        ),
        *(
            NetworkLayer(
                kind="hidden",
                units=w.shape[1],
                activation=network.activation,
                shown=[f"neuron {i + 1}" for i in shown[k + 1]],
            )
            for k, w in enumerate(weights[:hidden])
        ),
        NetworkLayer(
            kind="output",
            units=weights[-1].shape[1],
            activation=network.output_activation,
            shown=[network.output_names[i] for i in shown[-1]],
        ),
    ]
    return NetworkSummary(
        layers=layers,
        parameters=sum((w.shape[0] + 1) * w.shape[1] for w in weights),  # + 1: the biases
        weights=[
            [[round6(v) for v in row] for row in w[np.ix_(shown[i], shown[i + 1])]]
            for i, w in enumerate(weights)
        ],
    )
