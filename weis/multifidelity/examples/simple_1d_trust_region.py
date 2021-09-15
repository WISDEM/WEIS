import unittest
import numpy as np
from weis.multifidelity.models.testbed_components import (
    simple_2D_high_model,
    simple_2D_low_model,
    simple_1D_high_model,
    simple_1D_low_model,
)
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


np.random.seed(13)

bounds = {"x": np.array([[0.0, 1.0]])}
desvars = {"x": np.array([0.25])}
model_low = simple_1D_low_model(desvars)
model_high = simple_1D_high_model(desvars)
trust_region = SimpleTrustRegion(
    model_low, model_high, bounds, num_initial_points=4, disp=2,
)

trust_region.add_objective("y")
# trust_region.add_constraint("con", upper=0.25)

results = trust_region.optimize(plot=True, num_iterations=1, num_basinhop_iterations=1, interp_method="linear")
