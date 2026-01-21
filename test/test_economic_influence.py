import unittest

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.economic import ContextSplit, EconomicInfluenceEstimator, aggregate_values


class DummyEvaluator(DataEvaluator, ModelMixin):
    def train_data_values(self, *args, **kwargs):
        self._values = np.arange(len(self.x_train), dtype=float)
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self._values


class TestEconomicInfluence(unittest.TestCase):
    def test_aggregate_values_equal_contexts(self):
        values = np.array([0.5, 1.0, -0.5])
        context_values = {"a": values, "b": values}
        contexts = [
            ContextSplit(name="a", indices=np.array([0, 1]), phi=1.0),
            ContextSplit(name="b", indices=np.array([2]), phi=1.0),
        ]
        probs = {"a": 0.6, "b": 0.4}

        aggregated = aggregate_values(context_values, contexts, probs)
        self.assertTrue(np.allclose(aggregated, values))

    def test_estimator_context_values_and_weights(self):
        x_train = torch.zeros((5, 2))
        y_train = torch.zeros((5, 1))
        x_valid = torch.zeros((4, 2))
        y_valid = torch.zeros((4, 1))

        contexts = [
            ContextSplit(name="low", indices=np.array([0, 1]), phi=1.0),
            ContextSplit(name="high", indices=np.array([2, 3]), phi=2.0),
        ]

        estimator = EconomicInfluenceEstimator(dataval_factory=lambda: DummyEvaluator())
        context_values = estimator.evaluate_contexts(
            x_train, y_train, x_valid, y_valid, contexts
        )

        expected = np.arange(len(x_train), dtype=float)
        for values in context_values.values():
            self.assertTrue(np.array_equal(values, expected))

        probs = {"low": 0.5, "high": 0.5}
        aggregated = aggregate_values(context_values, contexts, probs)
        self.assertTrue(np.allclose(aggregated, 1.5 * expected))


if __name__ == "__main__":
    unittest.main()
