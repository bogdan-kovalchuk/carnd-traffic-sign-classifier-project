"""Deterministic validation and splitting helpers for labeled image data."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def validate_labels(labels: np.ndarray) -> np.ndarray:
    """Validate a non-empty one-dimensional integer label array."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("labels must contain integer class ids")
    return values


def validate_split_indices(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    dataset_size: int,
) -> None:
    """Ensure two index arrays form a complete, disjoint dataset partition."""
    if not isinstance(dataset_size, Integral) or isinstance(dataset_size, bool):
        raise TypeError("dataset_size must be an integer")
    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than one")
    train = np.asarray(train_indices)
    validation = np.asarray(validation_indices)
    for name, indices in (("train_indices", train), ("validation_indices", validation)):
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"{name} must contain integers")
        if np.any(indices < 0) or np.any(indices >= dataset_size):
            raise ValueError(f"{name} contains an out-of-range index")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate indices")
    if np.intersect1d(train, validation).size:
        raise ValueError("train and validation indices must be disjoint")
    combined = np.sort(np.concatenate((train, validation)))
    if not np.array_equal(combined, np.arange(dataset_size)):
        raise ValueError("train and validation indices must cover the dataset exactly")


def stratified_split(
    labels: np.ndarray,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return seeded, stratified train and validation indices.

    Each class with at least two samples contributes at least one item to each
    partition. Singleton classes remain in training so the result is a valid
    complete partition.
    """
    values = validate_labels(labels)
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and between zero and one")

    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for class_id in np.unique(values):
        class_indices = np.flatnonzero(values == class_id)
        shuffled = rng.permutation(class_indices)
        validation_count = int(np.floor(class_indices.size * validation_fraction))
        if class_indices.size >= 2:
            validation_count = max(1, min(class_indices.size - 1, validation_count))
        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])

    train = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    validation = np.sort(np.concatenate(validation_parts).astype(np.int64, copy=False))
    validate_split_indices(train, validation, values.size)
    return train, validation


def stratified_split_round_robin(
    labels: np.ndarray,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by evenly spaced within-class positions without random state.

    This alternative preserves source order and selects validation examples at
    regular intervals.  It is useful when a caller needs repeatability across
    NumPy versions, at the cost of offering no protection against ordered data
    artifacts that a seeded shuffle can avoid.
    """
    values = validate_labels(labels)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and between zero and one")

    train_parts = []
    validation_parts = []
    for class_id in np.unique(values):
        class_indices = np.flatnonzero(values == class_id)
        validation_count = int(np.floor(class_indices.size * validation_fraction))
        if class_indices.size >= 2:
            validation_count = max(1, min(class_indices.size - 1, validation_count))
        validation_positions = np.linspace(
            0, class_indices.size, num=validation_count, endpoint=False, dtype=np.int64
        )
        validation_mask = np.zeros(class_indices.size, dtype=bool)
        validation_mask[validation_positions] = True
        validation_parts.append(class_indices[validation_mask])
        train_parts.append(class_indices[~validation_mask])

    train = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    validation = np.sort(np.concatenate(validation_parts).astype(np.int64, copy=False))
    validate_split_indices(train, validation, values.size)
    return train, validation
