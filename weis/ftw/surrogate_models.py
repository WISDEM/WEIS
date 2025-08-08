import numpy as np
from copy import copy
from smt.surrogate_models import SGP

class SM:
    def __init__(self, smtype='SGP', bounds_in=None, bounds_out=None, **kwargs):

        # Flag whether the quantity is constant or not.
        # If constant, quantity does not need SM to be trained.
        self.constant = False
        self.constant_value = np.array([[0.0]])

        # smtype: smt model type, SGP (default), KRG, KPLS, KPLSK, MGP, ...
        # See https://smt.readthedocs.io for more info
        # Defining SM object example:
        # smobj = SM(smtype='SGP', eval_noise=True, print_global=False)
        self.smtype = smtype
        self.sm_kwargs = copy(kwargs)
        mod = __import__('smt.surrogate_models', fromlist = [smtype])
        self.smclass = getattr(mod, smtype)
        self._sm = []
        # Usage: smobj = self.smclass(**self.sm_kwargs)

        # Predict variances?
        if hasattr(self.smclass, 'predict_variances'):
            self._variances = True
        else:
            self._variances = False

        # Normalizing bounds
        self.bounds_in = bounds_in
        self.bounds_out = bounds_out
        if (bounds_in != None) and (bounds_out != None):
            self._bounds_set = True
        else:
            self._bounds_set = False

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

        lb_out = np.tile(self.bounds_out[0,:], (x_in.shape[0], 1))
        ub_out = np.tile(self.bounds_out[1,:], (x_in.shape[0], 1))

        x_in_normalized = (x_in - lb_in)/(ub_in - lb_in)

        if self.constant:
        # If responses are constant valued
            y_out_normalized = np.tile(
                self.constant_value, (x_in.shape[0], 1))
            sqrt_v_out_normalized = np.zeros(shape=y_out_normalized.shape, dtype=float)
        else:
        # Otherwise, surrogate model predicts values
            if len(self._sm) == 0:
                raise Exception('Surrogate model not trained.')
            y_out_normalized = np.zeros(shape=(x_in.shape[0], 0), dtype=float)
            sqrt_v_out_normalized = np.zeros(shape=(x_in.shape[0], 0), dtype=float)
            for sm in self._sm:
                y_out_column = sm.predict_values(x_in_normalized)
                y_out_normalized = np.concatenate(
                    (y_out_normalized, y_out_column), axis=1)
                try:
                    sqrt_v_out_column = np.sqrt(sm.predict_variances(x_in_normalized))
                    sqrt_v_out_normalized = np.concatenate(
                        (sqrt_v_out_normalized, sqrt_v_out_column), axis=1)
                except:
                    sqrt_v_out_normalized = np.concatenate(
                        (sqrt_v_out_normalized, np.zeros(shape=(x_in.shape[0], 1), dtype=float)), axis=1)

        # Denormalize outputs
        y_out = lb_out + (ub_out - lb_out)*y_out_normalized
        sqrt_v_out = (ub_out - lb_out)*sqrt_v_out_normalized
        v_out = sqrt_v_out**2

        return y_out, v_out # values and variances

