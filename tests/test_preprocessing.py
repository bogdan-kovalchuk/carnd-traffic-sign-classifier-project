import unittest

import numpy as np

from src.preprocessing import (
    grayscale_luma,
    grayscale_mean,
    normalize_grayscale,
    preprocess_luma,
    preprocess_mean,
)


class PreprocessingCharacterizationTests(unittest.TestCase):
    def test_grayscale_mean_matches_notebook_formula(self):
        images = np.array([[[[0, 30, 60], [90, 120, 150]]]], dtype=np.uint8)

        result = grayscale_mean(images)

        expected = np.array([[[[30.0], [120.0]]]])
        np.testing.assert_allclose(result, expected)
        self.assertEqual(result.shape, (1, 1, 2, 1))
        self.assertEqual(result.dtype, np.float64)

    def test_normalization_maps_endpoints(self):
        pixels = np.array([0.0, 127.5, 255.0])

        result = normalize_grayscale(pixels)

        np.testing.assert_allclose(result, [0.1, 0.5, 0.9])

    def test_preprocess_does_not_mutate_input(self):
        images = np.full((2, 2, 2, 3), 255, dtype=np.uint8)
        original = images.copy()

        result = preprocess_mean(images)

        np.testing.assert_array_equal(images, original)
        np.testing.assert_allclose(result, 0.9)


class PreprocessingAlternativeTests(unittest.TestCase):
    def test_luma_uses_perceptual_channel_weights(self):
        image = np.array([[[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]], dtype=np.uint8)

        result = grayscale_luma(image)

        np.testing.assert_allclose(result, [[[[76.245], [149.685], [29.07]]]])
        self.assertFalse(np.allclose(result, grayscale_mean(image)))

    def test_luma_pipeline_has_same_shape_and_normalized_range(self):
        image = np.array([[[[255, 0, 0]]]], dtype=np.uint8)

        result = preprocess_luma(image)

        self.assertEqual(result.shape, (1, 1, 1, 1))
        self.assertGreater(float(result.min()), 0.1)
        self.assertLess(float(result.max()), 0.9)


if __name__ == "__main__":
    unittest.main()
