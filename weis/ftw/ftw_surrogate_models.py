import numpy as np
from smt.surrogate_models import SGP

class SGP_WT(SGP):
    # Sparse Gaussian Process surrogate model class, specifically tailored for WEIS use,
    # inherited from SGP class included in the SMT Toolbox.

    def _initialize(self):
        super()._initialize()
        self._bounds_set = False
        self.keys_in = None
        self.keys_out = None
        self.R_squared = None
        self.constant = False
        self.constant_value = 0.0

    def set_bounds(self, bounds_in, bounds_out):
        self.bounds_in = bounds_in
        self.bounds_out = bounds_out
        self._bounds_set = True

    def predict(self, x):
        # Predicts surrogate model response and variance
        # Input (x) and outputs (y_out, v_out) are denormalized (raw scale) values
        # while actual surrogate model computation is done in normalized scale.

        x_in = np.array(x)

        if not self._bounds_set:
            raise Exception('Normalizing bounds are needed before accessing surrogate model.')

        if not len(x_in.shape) == 2:
            raise Exception('Input array x needs to have shape = (:,n_dv).')

        lb_in = np.tile(self.bounds_in[0,:], (x_in.shape[0], 1))
        ub_in = np.tile(self.bounds_in[1,:], (x_in.shape[0], 1))

        lb_out = np.tile(self.bounds_out[0], (x_in.shape[0], 1))
        ub_out = np.tile(self.bounds_out[1], (x_in.shape[0], 1))

        x_in_normalized = (x_in - lb_in)/(ub_in - lb_in)
        if self.constant:
            y_out_normalized = self.constant_value*np.ones((x_in.shape[0], 1), dtype=float)
            sqrt_v_out_normalized = 0.0
        else:
            y_out_normalized = self.predict_values(x_in_normalized)
            sqrt_v_out_normalized = np.sqrt(self.predict_variances(x_in_normalized))

        y_out = lb_out + (ub_out - lb_out)*y_out_normalized
        sqrt_v_out = (ub_out - lb_out)*sqrt_v_out_normalized
        v_out = sqrt_v_out**2

        return y_out, v_out # values and variances
