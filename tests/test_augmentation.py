import unittest

import numpy as np

from src.augmentation import (
    horizontal_flip,
    translate,
    translate_reflect,
    translate_wrap,
    translate_zero,
)


class AugmentationCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.images = np.array([[[[1], [2], [3]], [[4], [5], [6]]]], dtype=np.uint8)

    def test_horizontal_flip_matches_expected_pixels(self):
        result = horizontal_flip(self.images)

        np.testing.assert_array_equal(result, [[[[3], [2], [1]], [[6], [5], [4]]]])
        self.assertEqual(result.dtype, self.images.dtype)

    def test_zero_translation_moves_pixels_without_wrapping(self):
        result = translate_zero(self.images, dx=1, dy=1)

        expected = np.array([[[[0], [0], [0]], [[0], [1], [2]]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_wrap_translation_retains_pixels_at_the_opposite_edge(self):
        result = translate_wrap(self.images, dx=1, dy=1)

        expected = np.array([[[[6], [4], [5]], [[3], [1], [2]]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_reflect_translation_has_different_boundary_semantics(self):
        result = translate_reflect(self.images, dx=1, dy=1)

        expected = np.array([[[[5], [4], [5]], [[2], [1], [2]]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_translation_selector_exposes_each_boundary_strategy(self):
        for mode, function in (
            ("zero", translate_zero),
            ("wrap", translate_wrap),
            ("reflect", translate_reflect),
        ):
            np.testing.assert_array_equal(
                translate(self.images, dx=1, dy=1, mode=mode),
                function(self.images, dx=1, dy=1),
            )
        with self.assertRaises(ValueError):
            translate(self.images, mode="crop")
        with self.assertRaises(TypeError):
            translate(self.images, mode=None)

    def test_invalid_shapes_offsets_and_extreme_offsets_are_handled(self):
        with self.assertRaises(ValueError):
            horizontal_flip(np.zeros((2, 3, 3), dtype=np.uint8))
        with self.assertRaises(TypeError):
            translate_zero(self.images, dx=0.5)
        with self.assertRaises(TypeError):
            translate_wrap(self.images, dy=True)
        np.testing.assert_array_equal(
            translate_zero(self.images, dx=100), np.zeros_like(self.images)
        )
        one_pixel = np.array([[[[7]]]], dtype=np.uint8)
        np.testing.assert_array_equal(translate_reflect(one_pixel, 100, -100), one_pixel)

    def test_augmentation_does_not_mutate_input(self):
        original = self.images.copy()

        horizontal_flip(self.images)
        translate_zero(self.images, dx=-1)

        np.testing.assert_array_equal(self.images, original)


if __name__ == "__main__":
    unittest.main()
