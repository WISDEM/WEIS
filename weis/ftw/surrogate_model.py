import warnings
import numpy as np
from copy import copy
from wisdem.inputs import load_yaml

class surrogate_model:
    def __init__(self,
            smtype='SGP', bounds_in=None, bounds_out=None,
            constant=False, constant_value=None,
            r_inducing=0.6, n_inducing_min=80, n_inducing_max=150,
            **kwargs):
        # Flag whether the quantity is constant or not.
        # If constant, quantity does not need SM to be trained.
        self.constant = constant
        self.constant_value = constant_value
        # If constant, check dimensions of constant values
        if self.constant:
            assert type(self.constant_value) == np.ndarray
            assert len(self.constant_value.shape) == 2
            assert self.constant_value.shape[0] == 1
            assert np.prod(self.constant_value.shape) >= 1 # Not empty array
        # smtype: smt model type, SGP (default), KRG, KPLS, KPLSK, MGP, ...
        # See https://smt.readthedocs.io for more info
        # Defining SM object example:
        # smobj = SM(smtype='SGP', eval_noise=True, print_global=False)
        self.smtype = smtype
        self.sm_kwargs = copy(kwargs)
        self._r_inducing = r_inducing
        self._n_inducing_min = n_inducing_min
        self._n_inducing_max = n_inducing_max
        mod = __import__('smt.surrogate_models', fromlist = [smtype])
        self.smclass = getattr(mod, smtype)
        self._sm = []
        # Usage: smobj = self.smclass(**self.sm_kwargs)
        # Normalizing bounds
        self.bounds_in = bounds_in
        self.bounds_out = bounds_out
        if (type(bounds_in) == np.ndarray) and (type(bounds_out) == np.ndarray):
            self._bounds_set = True
        else:
            self._bounds_set = False
    def training(self, xt, yt, bounds_in=None, bounds_out=None):
        # Reset random generator
        rng = np.random.RandomState(42)
        # If bounds are provided, set bounds first
        if (type(bounds_in) == np.ndarray) and (type(bounds_out) == np.ndarray):
            self.set_bounds(bounds_in, bounds_out)
        # If constant surrogate model, return
        if self.constant:
            self.constant_value = np.mean(np.array(yt), axis=0).reshape(1,-1)
            return
        # If bounds are not set, return with warning
        if not self._bounds_set:
            warnings.warn('Set bounds before entering training values.')
            return
        xt = np.array(xt)
        yt = np.array(yt)
        lb_in = np.tile(self.bounds_in[0,:].reshape(1,-1), (xt.shape[0], 1))
        ub_in = np.tile(self.bounds_in[1,:].reshape(1,-1), (xt.shape[0], 1))
        lb_out = np.tile(self.bounds_out[0,:].reshape(1,-1), (yt.shape[0], 1))
        ub_out = np.tile(self.bounds_out[1,:].reshape(1,-1), (yt.shape[0], 1))
        n_data = xt.shape[0]
        n_x = xt.shape[1]
        n_y = yt.shape[1]
        assert n_data == yt.shape[0]
        assert n_x == lb_in.shape[1]
        assert n_y == lb_out.shape[1]
        # Normalize
        xt_normalized = (xt - lb_in)/(ub_in - lb_in)
        yt_normalized = (yt - lb_out)/(ub_out - lb_out)
        # Inducing points
        n_inducing = np.min([int(self._r_inducing*float(n_data)), self._n_inducing_max])
        n_inducing = np.min([np.max([n_inducing, self._n_inducing_min]), n_data])
        random_idx = rng.permutation(n_data)[:n_inducing]
        Z = xt_normalized[random_idx].copy()
        for idx in range(n_y):
            # Create surrogate model object for each column
            if len(self._sm) > idx:
                sm = self.smclass(**self.sm_kwargs)
                self._sm[idx] = sm
            else:
                sm = self.smclass(**self.sm_kwargs)
                self._sm.append(sm)
            # Training surrogate model
            sm.set_training_values(xt_normalized, yt_normalized[:,idx:(idx+1)])
            if hasattr(sm, 'set_inducing_inputs'): # e.g., if SGP, ...
                # Use inducing points
                sm.set_inducing_inputs(Z=Z)
            sm.train()
    def set_bounds(self, bounds_in, bounds_out):
        self.bounds_in = bounds_in
        self.bounds_out = bounds_out
        self._bounds_set = True
    def predict(self, x):
        # Predicts surrogate model response and variance
        # Input (x) and outputs (y_out, v_out) are denormalized (raw scale) values
        # while actual surrogate model computation is done in normalized scale.
        x_in = np.array(x)
        if not len(x_in.shape) == 2:
            warnings.warn('Input array x needs to have shape = (:,n_dv).')
            return None, None
        # If constant surrogate model, return constant_value
        if self.constant:
            y_out = np.tile(self.constant_value, (x_in.shape[0], 1))
            v_out = np.zeros(shape=y_out.shape, dtype=float)
            return y_out, v_out
        if not self._bounds_set:
            warnings.warn('Set bounds before predicting values.')
            return None, None
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



# Demonstration script (for low-level access only)
if __name__ == '__main__':
    fname_doedata = '/Users/yhlee/work/WEIS-FTW-DEV/examples/99_design_coupling_analysis/outputs/umaine_semi_raft_dc/log_opt-doedata.yaml'
    doedata = load_yaml(fname_doedata)
    inputs = doedata['input']
    n_vars_inputs = len(inputs)
    n_real_inputs = sum([i['len'] for i in inputs])
    outputs = doedata['output']
    n_vars_outputs = len(outputs)
    n_real_outputs = sum([i['len'] for i in outputs])
    n_data = np.array(inputs[0]['data']).shape[0]

    # Prepare dataset
    input_array = np.zeros((n_data, 0), dtype=float)
    for idx in range(n_vars_inputs):
        input_array = np.concatenate(
            (input_array, np.array(inputs[idx]['data'])), axis=1)

    input_labels = [
        [i['name'] for i in inputs],
        [i['len'] for i in inputs],
    ]
    input_labels.append([sum(input_labels[1][:i]) for i in range(len(input_labels[1]))])

    output_array = np.zeros((n_data, 0), dtype=float)
    for idx in range(n_vars_outputs):
        output_array = np.concatenate(
            (output_array, np.array(outputs[idx]['data'])), axis=1)

    output_labels = [
        [i['name'] for i in outputs],
        [i['len'] for i in outputs],
    ]
    output_labels.append([sum(output_labels[1][:i]) for i in range(len(output_labels[1]))])

    input_chunks = list([list(range(n_vars_outputs))])
    index_list = input_chunks[0]
    # Setting bounds and normalizing data
    lb_in = np.min(input_array, axis=0).reshape(1,-1)
    ub_in = np.max(input_array, axis=0).reshape(1,-1)
    for idx in range(lb_in.shape[1]):
        if (ub_in[0,idx] - lb_in[0,idx]) < 10.0*np.finfo(np.float64).eps:
            lb_in[0,idx] = lb_in[0,idx] - 0.5*np.abs(lb_in[0,idx])
            ub_in[0,idx] = ub_in[0,idx] + 0.5*np.abs(ub_in[0,idx])

    lb_out = np.min(output_array, axis=0).reshape(1,-1)
    ub_out = np.max(output_array, axis=0).reshape(1,-1)
    const_out = np.zeros(shape=lb_out.shape, dtype=int)
    for idx in range(lb_out.shape[1]):
        if (ub_out[0,idx] - lb_out[0,idx]) < 10.0*np.finfo(np.float64).eps:
            lb_out[0,idx] = lb_out[0,idx] - 0.5*np.abs(lb_out[0,idx])
            ub_out[0,idx] = ub_out[0,idx] + 0.5*np.abs(ub_out[0,idx])
            const_out[0,idx] = 1 # mark constant

    bounds_in = np.concatenate((lb_in, ub_in), axis=0)
    bounds_out = np.concatenate((lb_out, ub_out), axis=0)

    # Training
    sm_list = []

    for idx in index_list:
        idx_start = output_labels[2][idx]
        idx_endp1 = idx_start + output_labels[1][idx]
        output_array_train = output_array[:,idx_start:idx_endp1]
        bounds_out_train = bounds_out[:,idx_start:idx_endp1]
        const_out_train = const_out[:,idx_start:idx_endp1]
        if const_out_train.all():
            # If constant output, the object is just returning constant array regardless of input values.
            sm = surrogate_model(constant=True,
                constant_value=np.mean(output_array_train, axis=0).reshape(1,-1),
                bounds_in=bounds_in, bounds_out=bounds_out_train)
        else:
            # Surrogate model training for non-constant outputs
            sm = surrogate_model(smtype='SGP', # SGP: Sparse Gaussian Process
                bounds_in=bounds_in, bounds_out=bounds_out_train,
                eval_noise=True, print_global=False)
            sm.training(input_array, output_array_train)
            print('surrogate model trained for output variable {:}'.format(idx+1))
        sm_list.append(sm)

