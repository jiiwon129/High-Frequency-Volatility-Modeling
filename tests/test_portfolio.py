import unittest

import numpy as np

from src.portfolio import portfolio_covariance, realized_variance, solve
from src.prvm_estimation import estimate_day


class PortfolioTests(unittest.TestCase):
    def test_jump_projection_preserves_continuous_risk(self):
        continuous = np.diag([2.0, 1.0])
        jump = np.diag([-3.0, 0.5])
        separate = portfolio_covariance(continuous, jump, "separate")
        combined = portfolio_covariance(continuous, jump, "combined")
        np.testing.assert_allclose(separate, np.diag([2.0, 1.5]))
        self.assertGreaterEqual(np.linalg.eigvalsh(separate - continuous).min(), -1e-12)
        self.assertLess(combined[0, 0], 1e-10)

    def test_portfolio_constraints(self):
        covariance = np.array([[2.0, 1.2], [1.2, 1.0]])
        for limit in [1.0, 1.5, 3.0]:
            weights = solve(covariance, limit)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
            self.assertLessEqual(float(abs(weights).sum()), limit + 1e-6)

    def test_realized_variance(self):
        returns = np.array([[0.01, -0.01], [0.02, 0.00]])
        self.assertAlmostEqual(realized_variance(returns, np.array([0.5, 0.5])), 0.0001)

    def test_intraday_boundary_does_not_require_next_day(self):
        prices = np.zeros((390, 2))
        prices[:, 0] = np.arange(390) * 1e-5
        prices[:, 1] = np.arange(390) * 2e-5
        _, _, returns = estimate_day(prices)
        self.assertEqual(returns.shape, (39, 2))
        np.testing.assert_allclose(returns[-1], [9e-5, 18e-5])
        with self.assertRaises(ValueError):
            estimate_day(prices[:-1])


if __name__ == "__main__":
    unittest.main()
