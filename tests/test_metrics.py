import unittest

import numpy as np

from src.metrics import accuracy, classification_report, confusion_matrix, top_k_accuracy


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


if __name__ == "__main__":
    unittest.main()
