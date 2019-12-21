"""Dependency-light classification metrics for traffic-sign predictions."""

from __future__ import annotations

import numpy as np


def _validate_label_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if true.size == 0 or true.size != pred.size:
        raise ValueError("labels must be non-empty and have equal lengths")
    if not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise TypeError("labels must contain integers")
    if np.any(true < 0) or np.any(pred < 0):
        raise ValueError("labels must be non-negative")
    return true, pred


def _resolve_num_classes(true: np.ndarray, pred: np.ndarray, num_classes: int | None) -> int:
    maximum = int(max(true.max(), pred.max()))
    if num_classes is None:
        return maximum + 1
    if not isinstance(num_classes, (int, np.integer)) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes <= maximum:
        raise ValueError("num_classes must exceed every observed class id")
    return int(num_classes)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy using a transparent loop."""
    true, pred = _validate_label_pair(y_true, y_pred)
    correct = sum(int(expected == actual) for expected, actual in zip(true, pred))
    return correct / true.size


def accuracy_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return accuracy using NumPy elementwise comparison."""
    true, pred = _validate_label_pair(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Build a rows=true, columns=predicted confusion matrix."""
    true, pred = _validate_label_pair(y_true, y_pred)
    classes = _resolve_num_classes(true, pred, num_classes)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for expected, actual in zip(true, pred):
        matrix[expected, actual] += 1
    return matrix


def confusion_matrix_vectorized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Build a confusion matrix with indexed accumulation."""
    true, pred = _validate_label_pair(y_true, y_pred)
    classes = _resolve_num_classes(true, pred, num_classes)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> dict[int, dict[str, float | int]]:
    """Return precision, recall, F1, and support for each class."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    report = {}
    for class_id in range(matrix.shape[0]):
        true_positive = int(matrix[class_id, class_id])
        predicted = int(matrix[:, class_id].sum())
        actual = int(matrix[class_id, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
    return report


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Return the fraction whose label appears in the top ``k`` scores."""
    scores = np.asarray(logits)
    targets = np.asarray(labels)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("logits must be a non-empty two-dimensional array")
    if not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError("logits must contain finite numeric values")
    if targets.ndim != 1 or targets.size != scores.shape[0]:
        raise ValueError("labels must match the number of logit rows")
    if not np.issubdtype(targets.dtype, np.integer) or np.any(targets < 0):
        raise TypeError("labels must contain non-negative integers")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 1 or k > scores.shape[1]:
        raise ValueError("k must be between one and the number of classes")
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return float(np.mean(np.any(top == targets[:, None], axis=1)))
