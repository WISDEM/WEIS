import unittest
import numpy as np
from weis.multifidelity.models.testbed_components import (
    eggcrate_high_model,
    eggcrate_low_model,
)
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


class Test(unittest.TestCase):
    def test_optimization(self):
        np.random.seed(13)

        bounds = {"x": np.array([[-5.0, 5.0], [-5.0, 5.0]])}
        desvars = {"x": np.array([0.5, 0.5])}
        model_low = eggcrate_low_model(desvars)
        model_high = eggcrate_high_model(desvars)
        trust_region = SimpleTrustRegion(model_low, model_high, bounds, disp=False, trust_radius=1., num_initial_points=40)

        trust_region.add_objective("y")
        trust_region.set_initial_point(desvars['x'])

        results = trust_region.optimize(plot=False, num_iterations=10)

        np.testing.assert_allclose(
            results["optimal_design"], [0.0, 0.0], atol=1.
        )


if __name__ == "__main__":
    unittest.main()