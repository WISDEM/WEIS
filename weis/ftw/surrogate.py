import time, os, warnings
import numpy as np
import pickle as pkl
import openmdao.api as om
from copy                       import copy
from openmdao.utils.mpi         import MPI
from wisdem.inputs              import load_yaml
from openfast_io.FileTools      import save_yaml
from weis.ftw.ftw_tools         import split_list_chunks
from weis.ftw.surrogate_model   import surrogate_model


def ftw_surrogate_modeling(fname_doedata=None, fname_smt=None,
        doedata=None, WTSM=None, skip_training_if_sm_exist=False):
    # MPI
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        n_cores = MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        n_cores = 1
    # Broadcast arguments (trust data from rank 0)
    if MPI and n_cores > 1:
        MPI.COMM_WORLD.bcast(fname_doedata, root=0)
        MPI.COMM_WORLD.bcast(fname_smt, root=0)
        MPI.COMM_WORLD.bcast(doedata, root=0)
        MPI.COMM_WORLD.bcast(WTSM, root=0)
        MPI.COMM_WORLD.bcast(skip_training_if_sm_exist, root=0)
    # Parse arguments
    if skip_training_if_sm_exist: # Try to read WTSM and/or smt
        # Check if WTSM is provided
        if type(WTSM) == WTsurrogate:
            # Try to write to file if rank == 0
            if rank == 0:
                try:
                    # Write smt only if file does not exist
                    if (type(fname_smt) == str) and (not os.path.isfile(fname_smt)):
                        WTSM._write_sm(fname_smt)
                except:
                    pass
            return WTSM # Must return at here as WTSM is already provided
        elif type(WTSM) == None:
            # Read smt file
            if type(fname_smt) == str:
                smt_loaded = False
                if rank == 0: # Read and broadcast from rank 0
                    if os.path.isfile(fname_smt):
                        try:
                            WTSM = WTsurrogate()
                            WTSM._read_sm(fname_smt)
                            if WTSM._sm_trained: # If successfully loaded
                                smt_loaded = True
                        except:
                            pass
                if MPI:
                    # Broadcast flag
                    MPI.COMM_WORLD.bcast(smt_loaded, root=0)
                if smt_loaded:
                    if MPI:
                        # Broadcast WTSM
                        MPI.COMM_WORLD.bcast(WTSM, root=0)
                    return WTSM
    # WTSM is not privided and smt reading is not successful.
    # Generate WTSM based on doedata
    if type(doedata) == dict: # doedata provided
        WTSM = WTsurrogate(doedata=doedata)
    elif type(fname_doedata) == str: # file path for doedata provided
        WTSM = WTsurrogate(doedata=fname_doedata)
    if rank == 0:
        try:
            # If filename is provided, write smt
            if (type(fname_smt) == str):
                WTSM._write_sm(fname_smt)
        except:
            pass
    return WTSM


class WTsurrogate:
    # Read DOE sql files, process data, create and train surrogate models, and save them in smt file.
    def __init__(self, doedata=None):
        self._doe_loaded = False
        self._sm_trained = False
        # MPI cores and rank number
        if MPI:
            self._rank = MPI.COMM_WORLD.Get_rank()
            self._n_cores = MPI.COMM_WORLD.Get_size()
        else:
            self._rank = 0
            self._n_cores = 1
        # Broadcast doedata to all cores
        if MPI and self._n_cores > 1:
            MPI.COMM_WORLD.bcast(doedata, root=0)
        # If doedata=None, sm training performed manually
        # If doedata is provided, sm training performed automatically
        if type(doedata) == dict:
            try:
                self._doedata = doedata
                self._doe_loaded = True
            except:
                self._doe_loaded = False
        elif type(doedata) == str:
            try:
                # Loading DOE data from yaml file
                doedata = load_yaml(os.path.realpath(doedata))
                self._doedata = doedata
                self._doe_loaded = True
            except:
                self._doe_loaded = False
        else:
            self._doe_loaded = False
        # Automatic training if _doe_loaded == True
        if self._doe_loaded:
            try:
                self.train_sm()
                # _sm_trained will be flagged in the train_sm function
            except:
                self._sm_trained = False
    def _read_sm(self, fname_smt):
        try:
            with open(fname_smt, 'rb') as fid:
                self._sm = pkl.load(fid)
                self._sm_trained = True
        except:
            self._sm_trained = False
            warnings.warn('smt file reading failed.')
    def _write_sm(self, fname_smt):
        sm = copy(self._sm)
        try:
            with open(fname_smt, 'wb') as fid:
                pkl.dump(sm, fid)
        except:
            warnings.warn('smt file writing failed.')
    def train_sm(self):
        # Check if doedata is loaded
        if not self._doe_loaded:
            warnings.warn('DOE data not loaded.')
            return
        # Prepare dataset
        inputs = self._doedata['input']
        n_vars_inputs = len(inputs)                         # Number of variables
        n_real_inputs = sum([i['len'] for i in inputs])     # Number of real numbers
        outputs = self._doedata['output']
        n_vars_outputs = len(outputs)                       # Number of variables
        n_real_outputs = sum([i['len'] for i in outputs])   # Number of real numbers
        n_data = np.array(inputs[0]['data']).shape[0]       # Number of DOE data
        # Prepare dataset in array and nested list forms
        # Inputs
        input_array = np.zeros((n_data, 0), dtype=float)
        for idx in range(n_vars_inputs):
            input_array = np.concatenate(
                (input_array, np.array(inputs[idx]['data'])), axis=1)
        # Input labels (with vector length and starting index)
        input_labels = [
            [i['name'] for i in inputs],
            [i['len'] for i in inputs],
        ]
        input_labels.append([sum(input_labels[1][:i]) for i in range(len(input_labels[1]))])
        # Outputs
        output_array = np.zeros((n_data, 0), dtype=float)
        for idx in range(n_vars_outputs):
            output_array = np.concatenate(
                (output_array, np.array(outputs[idx]['data'])), axis=1)
        # Output labels (with vector length and starting index)
        output_labels = [
            [i['name'] for i in outputs],
            [i['len'] for i in outputs],
        ]
        output_labels.append([sum(output_labels[1][:i]) for i in range(len(output_labels[1]))])
        # Distribute data array variable indices through MPI
        if self._rank == 0:
            # Indices list for each MPI core
            index_chunks = list(split_list_chunks(range(n_vars_outputs), self._n_cores))
            if self._n_cores > n_vars_outputs:
                index_chunks += [[] for i in range(self._n_cores - n_vars_outputs)]
        else:
            index_chunks = None
        if MPI:
            index_list = MPI.COMM_WORLD.scatter(index_chunks, root=0)
        else:
            index_list = index_chunks[0] # If non-MPI
        # Setting bounds and normalizing data
        lb_in = np.min(input_array, axis=0).reshape(1,-1)
        ub_in = np.max(input_array, axis=0).reshape(1,-1)
        for idx in range(lb_in.shape[1]):
            # If all values are (nearly) identical, use the scale of values
            if (ub_in[0,idx] - lb_in[0,idx]) < 10.0*np.finfo(np.float64).eps:
                lb_in[0,idx] = lb_in[0,idx] - 0.5*np.abs(lb_in[0,idx])
                ub_in[0,idx] = ub_in[0,idx] + 0.5*np.abs(ub_in[0,idx])
        lb_out = np.min(output_array, axis=0).reshape(1,-1)
        ub_out = np.max(output_array, axis=0).reshape(1,-1)
        const_out = const_out = np.zeros(shape=lb_out.shape, dtype=int)
        for idx in range(lb_out.shape[1]):
            # If all values are (nearly) identical, use the scale of values
            if (ub_out[0,idx] - lb_out[0,idx]) < 10.0*np.finfo(np.float64).eps:
                lb_out[0,idx] = lb_out[0,idx] - 0.5*np.abs(lb_out[0,idx])
                ub_out[0,idx] = ub_out[0,idx] + 0.5*np.abs(ub_out[0,idx])
                # mark constant
                const_out[0,idx] = 1
        bounds_in = np.concatenate((lb_in, ub_in), axis=0)
        bounds_out = np.concatenate((lb_out, ub_out), axis=0)
        self._bounds_in = bounds_in
        self._bounds_out = bounds_out
        self._input_array = input_array
        self._output_array = output_array
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
            sm_list.append({
                    'serial_number': idx,
                    'surrogate_model_object': sm,
                    'input': {
                        'label': input_labels[0],
                        'len': input_labels[1],
                        'index': input_labels[2],
                        'bounds': bounds_in.tolist(),
                    },
                    'output': {
                        'label': output_labels[0][idx],
                        'len': output_labels[1][idx],
                        'index': output_labels[2][idx],
                        'bounds': bounds_out_train.tolist(),
                    }
                })
        # MPI Gather
        if MPI and self._n_cores > 1:
            sm_list_gathered = MPI.COMM_WORLD.gather(sm_list, root=0)
            if self._rank == 0:
                sm_list_flattened = [i for sl in sm_list_gathered for i in sl]
                sm_list_new = [None for i in range(len(sm_list_flattened))]
                for idx in range(len(sm_list_flattened)):
                    sm_dict = copy(sm_list_flattened[idx])
                    sn = sm_dict['serial_number']
                    sm_list_new[sn] = sm_dict
                print('rank={:}, sm_list_new={:}'.format(self._rank, sm_list_new))
            else:
                sm_list_new = []
                print('rank={:}, len(sm_list_new)={:}'.format(self._rank, len(sm_list_new)))
            # MPI Re-broadcast full list to all cores
            MPI.COMM_WORLD.bcast(sm_list_new, root=0)
        else:
            sm_list_new = sm_list
        # Save to class object, finalize training
        self._sm = sm_list_new
        self._sm_trained = True




