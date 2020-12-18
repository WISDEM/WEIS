import unittest
import numpy as np
from weis.multifidelity.models.testbed_components import (
    simple_2D_high_model,
    simple_2D_low_model,
    simple_1D_high_model,
    simple_1D_low_model,
)
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


class Test(unittest.TestCase):
    def test_optimization(self):
        np.random.seed(13)

        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {"x": np.array([0.0, 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        trust_region = SimpleTrustRegion(model_low, model_high, bounds, disp=False)

        trust_region.add_objective("y")

        results = trust_region.optimize()

        np.testing.assert_allclose(results["optimal_design"], [0.0, 0.333], atol=1e-3)

    def test_constrained_optimization(self):
        np.random.seed(13)

        bounds = {"x": np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {"x": np.array([0.0, 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        trust_region = SimpleTrustRegion(
            model_low, model_high, bounds, num_initial_points=10, disp=False
        )

        trust_region.add_objective("y")
        trust_region.add_constraint("con", equals=0.0)

        results = trust_region.optimize(plot=False, num_iterations=10)

        np.testing.assert_allclose(results["optimal_design"], [0.0, 0.10987], atol=1e-3)
        np.testing.assert_allclose(results["outputs"]["con"], 0.0, atol=1e-5)

    def test_1d_constrained_optimization(self):
        np.random.seed(13)

        bounds = {"x": np.array([[0.0, 1.0]])}
        desvars = {"x": np.array([0.25])}
        model_low = simple_1D_low_model(desvars)
        model_high = simple_1D_high_model(desvars)
        trust_region = SimpleTrustRegion(
            model_low, model_high, bounds, num_initial_points=10, disp=False
        )

        trust_region.add_objective("y")
        trust_region.add_constraint("con", equals=0.25)

        results = trust_region.optimize(plot=False, num_iterations=10)

        np.testing.assert_allclose(results["optimal_design"], 0.707105, atol=1e-3)
        np.testing.assert_allclose(results["outputs"]["con"], 0.25, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
