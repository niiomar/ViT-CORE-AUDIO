import numpy as np

from metrics import accuracy, auc, compute_all, eer


def test_eer_perfectly_separable_scores_is_zero():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

    value, _ = eer(y_true, y_scores)
    assert value == 0.0


def test_eer_random_scores_is_near_50_percent():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=5000)
    y_scores = rng.rand(5000)

    value, _ = eer(y_true, y_scores)
    assert 0.4 <= value <= 0.6


def test_compute_all_returns_expected_keys_and_ranges():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.6, 0.9]
    y_pred = [0, 0, 1, 1]

    metrics = compute_all(y_true, y_scores, y_pred)

    assert set(metrics) == {"accuracy", "auc", "eer", "eer_pct", "eer_threshold"}
    assert metrics["accuracy"] == accuracy(y_true, y_pred)
    assert metrics["auc"] == auc(y_true, y_scores)
    assert metrics["eer_pct"] == metrics["eer"] * 100
    assert 0.0 <= metrics["eer"] <= 1.0
