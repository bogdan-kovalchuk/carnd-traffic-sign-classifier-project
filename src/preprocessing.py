"""Image preprocessing helpers matching the traffic-sign notebook.

The notebook converts RGB pixels to a single channel by taking their mean and
then maps the byte range ``[0, 255]`` to ``[0.1, 0.9]``.  These functions keep
that behavior independent of TensorFlow so it can be tested on small arrays.

Complexity and trade-offs:
    ``grayscale_mean`` is a one-pass vectorized reduction, O(NHWK) time and
    O(NHW) output space for N images of H by W pixels and K channels.
    ``grayscale_luma`` uses a tensor contraction with the same asymptotic
    cost, but its weighted arithmetic better models human brightness and is
    restricted to RGB input.  Normalization is O(M) in the number of pixels
    and returns a float64 array, trading memory for predictable numeric range.
"""

from __future__ import annotations

import numpy as np


def validate_image_batch(image_data: np.ndarray, channels: int | None = None) -> np.ndarray:
    """Validate and return an image batch without copying its values."""
    image = np.asarray(image_data)
    if image.ndim != 4:
        raise ValueError("image_data must have shape (batch, height, width, channels)")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image_data must contain numeric values")
    if any(size == 0 for size in image.shape):
        raise ValueError("image_data dimensions must be non-empty")
    if channels is not None and image.shape[-1] != channels:
        raise ValueError("image_data has an unexpected channel count")
    if not np.isfinite(image).all():
        raise ValueError("image_data must contain only finite values")
    return image


def grayscale_mean(image_data: np.ndarray) -> np.ndarray:
    """Return channel-mean grayscale data with a trailing singleton channel."""
    image = validate_image_batch(image_data)
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
    image = image.astype(np.float64)
    if not np.isfinite(image).all():
        raise ValueError("image_data must contain only finite values")
    return output_min + (image - input_min) * (
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
    image = validate_image_batch(image_data, channels=3)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return np.tensordot(image.astype(np.float64, copy=False), weights, axes=([-1], [0]))[
        ..., np.newaxis
    ]


def preprocess_luma(image_data: np.ndarray) -> np.ndarray:
    """Apply luma grayscale conversion followed by notebook normalization."""
    return normalize_grayscale(grayscale_luma(image_data))


def preprocess(image_data: np.ndarray, method: str = "mean") -> np.ndarray:
    """Preprocess images using the named ``mean`` or ``luma`` strategy."""
    if method == "mean":
        return preprocess_mean(image_data)
    if method == "luma":
        return preprocess_luma(image_data)
    raise ValueError("method must be 'mean' or 'luma'")
