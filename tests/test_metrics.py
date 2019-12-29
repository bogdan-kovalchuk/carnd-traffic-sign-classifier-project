import unittest

import numpy as np

from src.metrics import (
    accuracy,
    accuracy_from_confusion,
    accuracy_vectorized,
    classification_report,
    confusion_matrix,
    confusion_matrix_vectorized,
    top_k_accuracy,
    top_k_accuracy_partition,
)


class MetricsCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.true = np.array([0, 1, 2, 2, 1])
        self.pred = np.array([0, 2, 2, 1, 1])

    def test_accuracy_matches_notebook_style_evaluation(self):
        self.assertEqual(accuracy(self.true, self.pred), 3 / 5)

    def test_confusion_matrix_has_true_rows_and_predicted_columns(self):
        result = confusion_matrix(self.true, self.pred, num_classes=3)

        expected = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_classification_report_calculates_per_class_values(self):
        report = classification_report(self.true, self.pred, num_classes=3)

        self.assertEqual(report[0]["support"], 1)
        self.assertAlmostEqual(report[1]["precision"], 0.5)
        self.assertAlmostEqual(report[2]["recall"], 0.5)
        self.assertAlmostEqual(report[1]["f1"], 0.5)

    def test_top_k_accuracy_finds_labels_in_ranked_logits(self):
        logits = np.array([[4, 1, 0], [0, 2, 3], [3, 2, 1]])

        self.assertEqual(top_k_accuracy(logits, np.array([0, 1, 2]), k=1), 2 / 3)
        self.assertEqual(top_k_accuracy(logits, np.array([0, 1, 2]), k=2), 1.0)

    def test_perfect_predictions_and_inferred_class_count_are_exact(self):
        labels = np.array([0, 2, 2])

        self.assertEqual(accuracy(labels, labels), 1.0)
        np.testing.assert_array_equal(
            confusion_matrix(labels, labels),
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]]),
        )

    def test_top_k_accepts_all_classes_as_the_boundary(self):
        logits = np.array([[0.0, 1.0], [1.0, 0.0]])

        self.assertEqual(top_k_accuracy(logits, np.array([1, 0]), k=2), 1.0)


class MetricsAlternativeTests(unittest.TestCase):
    def test_vectorized_accuracy_matches_reference(self):
        true = np.array([0, 1, 2, 2, 1])
        pred = np.array([0, 2, 2, 1, 1])

        self.assertEqual(accuracy_vectorized(true, pred), accuracy(true, pred))

    def test_confusion_trace_accuracy_matches_direct_methods(self):
        true = np.array([0, 1, 2, 2, 1])
        pred = np.array([0, 2, 2, 1, 1])

        self.assertEqual(accuracy_from_confusion(true, pred, 3), accuracy(true, pred))

    def test_vectorized_confusion_matrix_matches_reference(self):
        true = np.array([0, 1, 2, 2, 1])
        pred = np.array([0, 2, 2, 1, 1])

        np.testing.assert_array_equal(
            confusion_matrix_vectorized(true, pred, 4), confusion_matrix(true, pred, 4)
        )

    def test_partition_top_k_matches_full_sort_without_ties(self):
        logits = np.array([[9, 1, 4, 0], [0, 8, 3, 6], [5, 2, 7, 1]])
        labels = np.array([2, 3, 0])

        self.assertEqual(top_k_accuracy_partition(logits, labels, 2), top_k_accuracy(logits, labels, 2))


class MetricsRobustnessTests(unittest.TestCase):
    def test_label_metrics_reject_bad_lengths_types_and_class_counts(self):
        with self.assertRaises(ValueError):
            accuracy([0, 1], [0])
        with self.assertRaises(TypeError):
            accuracy(np.array([0.0]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0]), np.array([0]), num_classes=0)
        with self.assertRaises(TypeError):
            confusion_matrix(np.array([0]), np.array([0]), num_classes=1.5)

    def test_top_k_rejects_bad_logits_labels_and_k_boundaries(self):
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        for invalid in (0, 3):
            with self.assertRaises(ValueError):
                top_k_accuracy(logits, labels, invalid)
        with self.assertRaises(ValueError):
            top_k_accuracy(np.array([[np.inf, 0.0]]), np.array([0]), 1)
        with self.assertRaises(TypeError):
            top_k_accuracy(logits, np.array([0.0, 1.0]), 1)
        with self.assertRaises(ValueError):
            top_k_accuracy(np.empty((0, 2)), np.array([], dtype=np.int64), 1)

    def test_report_includes_absent_class_with_zero_support(self):
        report = classification_report(np.array([0, 0]), np.array([0, 0]), num_classes=2)

        self.assertEqual(report[1]["support"], 0)
        self.assertEqual(report[1]["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
