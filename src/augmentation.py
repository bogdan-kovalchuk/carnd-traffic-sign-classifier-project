"""Small NumPy augmentation helpers for traffic-sign image batches."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .preprocessing import validate_image_batch


def horizontal_flip(image_data: np.ndarray) -> np.ndarray:
    """Return a copied batch with each image mirrored left-to-right."""
    return np.flip(validate_image_batch(image_data), axis=2).copy()


def translate_zero(image_data: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Translate images and fill pixels outside the frame with zero.

    Positive ``dx`` moves right and positive ``dy`` moves down.
    """
    images = validate_image_batch(image_data)
    _validate_offset(dx, "dx")
    _validate_offset(dy, "dy")
    result = np.zeros_like(images)
    height, width = images.shape[1:3]
    source_y, target_y = _translation_slices(height, dy)
    source_x, target_x = _translation_slices(width, dx)
    if source_y is not None and source_x is not None:
        result[:, target_y, target_x, :] = images[:, source_y, source_x, :]
    return result


def translate_wrap(image_data: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Translate images with periodic wrap-around at the frame boundary."""
    images = validate_image_batch(image_data)
    _validate_offset(dx, "dx")
    _validate_offset(dy, "dy")
    return np.roll(images, shift=(dy, dx), axis=(1, 2)).copy()


def _validate_offset(offset: int, name: str) -> None:
    if not isinstance(offset, Integral) or isinstance(offset, bool):
        raise TypeError(f"{name} must be an integer")


def _translation_slices(size: int, offset: int):
    if offset >= size or offset <= -size:
        return None, None
    if offset >= 0:
        return slice(0, size - offset), slice(offset, size)
    return slice(-offset, size), slice(0, size + offset)
