import unittest

import numpy as np

from src.dataset import (
    stratified_split,
    stratified_split_round_robin,
    validate_labels,
    validate_split_indices,
)


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

    def test_label_validation_rejects_empty_float_and_negative_labels(self):
        with self.assertRaises(ValueError):
            validate_labels(np.array([], dtype=np.int64))
        with self.assertRaises(TypeError):
            validate_labels(np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            validate_labels(np.array([0, -1]))

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
        with self.assertRaises(TypeError):
            stratified_split(self.labels, validation_fraction="0.2")
        with self.assertRaises(ValueError):
            stratified_split(np.array([0, 1]), validation_fraction=0.5)

    def test_split_validator_rejects_invalid_dataset_boundaries(self):
        with self.assertRaises(ValueError):
            validate_split_indices([0], [1], 1)
        with self.assertRaises(ValueError):
            validate_split_indices([0, 3], [1], 3)
        with self.assertRaises(TypeError):
            validate_split_indices([0], [1], 3.0)


class DatasetAlternativeTests(unittest.TestCase):
    def test_round_robin_is_distinct_but_preserves_partition_invariants(self):
        labels = np.repeat(np.arange(3), 4)

        seeded = stratified_split(labels, validation_fraction=0.25, seed=17)
        round_robin = stratified_split_round_robin(labels, validation_fraction=0.25)

        validate_split_indices(*round_robin, labels.size)
        self.assertFalse(np.array_equal(seeded[1], round_robin[1]))
        self.assertEqual(set(labels[round_robin[1]]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
