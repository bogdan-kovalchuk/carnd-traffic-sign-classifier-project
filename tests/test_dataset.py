import unittest

import numpy as np

from src.dataset import stratified_split, validate_labels, validate_split_indices


class DatasetCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_split_is_deterministic_and_stratified(self):
        first = stratified_split(self.labels, validation_fraction=1 / 3, seed=17)
        second = stratified_split(self.labels, validation_fraction=1 / 3, seed=17)

        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first[0], [1, 2, 3, 5, 6, 8])
        np.testing.assert_array_equal(first[1], [0, 4, 7])

    def test_split_preserves_each_class_in_both_partitions(self):
        train, validation = stratified_split(self.labels, 0.5, seed=3)

        self.assertEqual(set(self.labels[train]), {0, 1, 2})
        self.assertEqual(set(self.labels[validation]), {0, 1, 2})
        validate_split_indices(train, validation, self.labels.size)

    def test_label_validation_accepts_integer_ids(self):
        result = validate_labels(np.array([2, 0, 1], dtype=np.int32))

        np.testing.assert_array_equal(result, [2, 0, 1])

    def test_split_validator_requires_an_exact_partition(self):
        with self.assertRaises(ValueError):
            validate_split_indices([0, 1], [1, 2], 3)
        with self.assertRaises(ValueError):
            validate_split_indices([0, 0], [1, 2], 3)
        with self.assertRaises(ValueError):
            validate_split_indices([0], [1], 3)

    def test_invalid_split_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            stratified_split(self.labels, validation_fraction=0)
        with self.assertRaises(ValueError):
            stratified_split(self.labels, validation_fraction=1)
        with self.assertRaises(TypeError):
            stratified_split(self.labels, seed=1.5)


if __name__ == "__main__":
    unittest.main()
