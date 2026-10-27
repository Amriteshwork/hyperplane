import numpy as np
import pytest

from app.ml.base import Network, TrainingCurve
from app.ml.evaluation import EVALUATORS, Predictions, summarise_network, training_curve_plots
from app.ml.evaluation.classification import evaluate_classification
from app.ml.evaluation.network import MAX_SHOWN
from app.ml.evaluation.numbers import MAX_POINTS
from app.ml.evaluation.regression import evaluate_regression
from app.ml.types import TaskType


def test_regression_metrics() -> None:
    evaluation = evaluate_regression(Predictions([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0]))
    metrics = {m.id: m.value for m in evaluation.metrics}

    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(1 - 4 / 5)


def test_regression_plots_pair_points_and_draw_guides() -> None:
    evaluation = evaluate_regression(Predictions([1.0, 2.0], [1.5, 2.0]))
    plots = {p.id: p for p in evaluation.plots}

    assert plots["predicted-vs-actual"].points == [(1.0, 1.5), (2.0, 2.0)]
    assert plots["predicted-vs-actual"].reference == "identity"
    assert plots["residuals"].points == [(1.5, -0.5), (2.0, 0.0)]
    assert plots["residuals"].reference == "zero"


def test_plots_are_subsampled_deterministically() -> None:
    y = np.arange(5_000, dtype=float)
    first = evaluate_regression(Predictions(y, y + 1)).plots[0].points
    assert len(first) == MAX_POINTS
    assert first == evaluate_regression(Predictions(y, y + 1)).plots[0].points


def test_regression_is_registered() -> None:
    assert EVALUATORS[TaskType.REGRESSION] is evaluate_regression


def test_classification_is_registered_for_both_task_types() -> None:
    assert EVALUATORS[TaskType.BINARY] is evaluate_classification
    assert EVALUATORS[TaskType.MULTICLASS] is evaluate_classification


def test_binary_metrics_use_the_models_last_class_as_positive() -> None:
    actual = ["no", "no", "no", "yes", "yes"]
    predicted = ["no", "no", "yes", "yes", "no"]
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
    evaluation = evaluate_classification(Predictions(actual, predicted, proba, ("no", "yes")))
    metrics = {m.id: m for m in evaluation.metrics}

    assert metrics["accuracy"].value == pytest.approx(3 / 5)
    assert metrics["precision"].label == "Precision (yes)"
    assert metrics["precision"].value == pytest.approx(1 / 2)  # 1 of 2 "yes" calls right
    assert metrics["recall"].value == pytest.approx(1 / 2)  # 1 of 2 actual "yes" found
    assert metrics["f1"].value == pytest.approx(1 / 2)
    assert metrics["roc_auc"].value == pytest.approx(5 / 6)  # 5 of 6 (yes, no) pairs ranked right
    expected_loss = -np.mean(np.log([0.9, 0.8, 0.4, 0.7, 0.4]))
    assert metrics["log_loss"].value == pytest.approx(expected_loss)
    assert not metrics["log_loss"].higher_is_better


def test_classification_plots_a_confusion_matrix_and_roc_curve() -> None:
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    evaluation = evaluate_classification(Predictions([0, 1, 1, 0], [0, 1, 0, 1], proba, (0, 1)))
    plots = {p.id: p for p in evaluation.plots}

    confusion = plots["confusion-matrix"]
    assert confusion.kind == "heatmap"
    assert confusion.y_categories == ["0", "1"]
    assert confusion.values == [[1.0, 1.0], [1.0, 1.0]]  # rows: actual, columns: predicted
    roc = plots["roc"]
    assert roc.kind == "curve" and roc.reference == "diagonal"
    [series] = roc.series
    assert series.points[0] == (0.0, 0.0) and series.points[-1] == (1.0, 1.0)


def test_multiclass_metrics_are_macro_averaged_with_one_roc_curve_per_class() -> None:
    classes = ("a", "b", "c")
    actual = ["a", "a", "b", "b", "c", "c"]
    predicted = ["a", "b", "b", "b", "c", "a"]
    proba = np.full((6, 3), 0.1) + 0.7 * np.eye(3)[[0, 1, 1, 1, 2, 0]]
    evaluation = evaluate_classification(Predictions(actual, predicted, proba, classes))
    metrics = {m.id: m for m in evaluation.metrics}

    # Recall per class: a 1/2, b 2/2, c 1/2.
    assert metrics["recall"].value == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["recall"].label == "Recall (macro)"
    roc = next(p for p in evaluation.plots if p.id == "roc")
    assert [s.name for s in roc.series] == ["a", "b", "c"]


def test_classification_without_probabilities_skips_log_loss_and_roc() -> None:
    evaluation = evaluate_classification(Predictions([0, 1, 1], [0, 1, 0]))
    assert [m.id for m in evaluation.metrics] == ["accuracy", "precision", "recall", "f1"]
    assert [p.id for p in evaluation.plots] == ["confusion-matrix"]


def test_roc_is_skipped_when_a_class_is_missing_from_the_test_rows() -> None:
    proba = np.array([[0.7, 0.3], [0.6, 0.4]])
    evaluation = evaluate_classification(Predictions([0, 0], [0, 0], proba, (0, 1)))
    assert "roc_auc" not in {m.id for m in evaluation.metrics}
    assert "log_loss" in {m.id for m in evaluation.metrics}


def test_network_summary_draws_the_strongest_inputs_and_counts_every_parameter() -> None:
    inputs = [f"x{i}" for i in range(10)]
    first = np.full((10, 20), 0.1)
    first[7] = 2.0  # the input the first layer weighs most
    first[3] = -1.0  # sign doesn't matter, size does
    second = np.arange(20.0).reshape(20, 1)
    summary = summarise_network(
        Network(inputs, [first, second], "relu", "identity", ["prediction"])
    )

    assert [(layer.kind, layer.units) for layer in summary.layers] == [
        ("input", 10),
        ("hidden", 20),
        ("output", 1),
    ]
    assert summary.layers[0].shown[:2] == ["x7", "x3"]
    assert [len(layer.shown) for layer in summary.layers] == [MAX_SHOWN, MAX_SHOWN, 1]
    assert summary.layers[1].activation == "relu" and summary.layers[0].activation is None
    assert summary.parameters == (10 + 1) * 20 + (20 + 1) * 1
    assert summary.weights[0][0][0] == 2.0  # x7 → neuron 1
    assert [row[0] for row in summary.weights[1]] == list(range(MAX_SHOWN))


def test_early_stopping_marks_the_epoch_whose_weights_were_kept() -> None:
    curve = TrainingCurve(loss=[3.0, 2.0, 1.0, 0.5], validation=[0.5, 0.9, 0.9, 0.8])
    plots = training_curve_plots(curve)
    assert [p.marker.x if p.marker else None for p in plots] == [2, 2]  # the first best

    plain = training_curve_plots(TrainingCurve(loss=[3.0, 2.0]))
    assert plain[0].marker is None
