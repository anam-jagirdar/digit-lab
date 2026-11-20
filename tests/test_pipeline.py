import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from digitlab.pipeline import candidate_models, compare_models, load_data
from digitlab.train import run


class PipelineTests(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self):
        x_train, x_test, y_train, y_test = load_data()
        self.assertEqual(x_train.shape[1], 64)
        self.assertEqual(len(x_train) + len(x_test), 1797)
        self.assertEqual(set(y_train), set(range(10)))
        self.assertEqual(set(y_test), set(range(10)))

    def test_split_is_reproducible(self):
        a = load_data()
        b = load_data()
        self.assertTrue(np.array_equal(a[1], b[1]))

    def test_comparison_is_sorted_best_first(self):
        x_train, _, y_train, _ = load_data()
        models = {k: v for k, v in candidate_models().items() if k in ("Logistic regression", "SVM (RBF kernel)")}
        ranking = compare_models(models, x_train, y_train, folds=3)
        self.assertEqual(len(ranking), 2)
        self.assertGreaterEqual(ranking[0].mean, ranking[1].mean)
        self.assertGreater(ranking[0].mean, 0.9)

    def test_end_to_end_saves_a_working_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(Path(tmp))
            self.assertGreater(summary["test_accuracy"], 0.95)
            for name in ("model.joblib", "results.json", "confusion_matrix.png"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            model = joblib.load(Path(tmp) / "model.joblib")
            _, x_test, _, y_test = load_data()
            self.assertGreater((model.predict(x_test) == y_test).mean(), 0.95)


if __name__ == "__main__":
    unittest.main()
