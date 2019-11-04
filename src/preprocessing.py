"""Image preprocessing helpers matching the traffic-sign notebook.

The notebook converts RGB pixels to a single channel by taking their mean and
then maps the byte range ``[0, 255]`` to ``[0.1, 0.9]``.  These functions keep
that behavior independent of TensorFlow so it can be tested on small arrays.
"""

from __future__ import annotations

import numpy as np


def grayscale_mean(image_data: np.ndarray) -> np.ndarray:
    """Return channel-mean grayscale data with a trailing singleton channel."""
    image = np.asarray(image_data)
    if image.ndim != 4:
        raise ValueError("image_data must have shape (batch, height, width, channels)")
    if image.shape[-1] == 0:
        raise ValueError("image_data must contain at least one channel")
    return image.astype(np.float64, copy=False).mean(axis=3, keepdims=True)


def normalize_grayscale(
    image_data: np.ndarray,
    output_min: float = 0.1,
    output_max: float = 0.9,
    input_min: float = 0.0,
    input_max: float = 255.0,
) -> np.ndarray:
    """Map an image array from an input interval to an output interval."""
    if not np.isfinite([output_min, output_max, input_min, input_max]).all():
        raise ValueError("normalization bounds must be finite")
    if input_max <= input_min:
        raise ValueError("input_max must be greater than input_min")
    if output_max < output_min:
        raise ValueError("output_max must not be less than output_min")
    image = np.asarray(image_data)
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image_data must contain numeric values")
    return output_min + (image.astype(np.float64) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )


def preprocess_mean(image_data: np.ndarray) -> np.ndarray:
    """Apply the notebook's mean grayscale conversion and normalization."""
    return normalize_grayscale(grayscale_mean(image_data))


def grayscale_luma(image_data: np.ndarray) -> np.ndarray:
    """Return perceptual BT.601 luma for RGB image batches.

    Unlike :func:`grayscale_mean`, this method weights green more heavily and
    therefore preserves perceived brightness more closely for colored signs.
    """
    image = np.asarray(image_data)
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError("image_data must have shape (batch, height, width, 3)")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return np.tensordot(image.astype(np.float64, copy=False), weights, axes=([-1], [0]))[
        ..., np.newaxis
    ]


def preprocess_luma(image_data: np.ndarray) -> np.ndarray:
    """Apply luma grayscale conversion followed by notebook normalization."""
    return normalize_grayscale(grayscale_luma(image_data))
