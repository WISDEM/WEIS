import numpy as np
import pickle as pkl
import openmdao.api as om
from openmdao.utils.mpi import MPI
from smt.surrogate_models import SGP

class WindTurbineDOE2SM():
    # Read DOE sql files, process data, create and train surrogate models, and save them in smt file.

    def __init__(self):
        self._doe_loaded = False
        self._sm_trained = False

    def read_doe(self, sql_file, modeling_options, opt_options):
        # Read DOE sql files. If MPI, read them in parallel.
        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else:
            rank = 0
            n_cores = 1

    def write_sm(self, sm_filename):
        # Write trained surrogate models in the smt file in pickle format
        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else
            rank = 0
            n_cores = 1

    def train_sm(self):
        # Surrogate model training
        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else
            rank = 0
            n_cores = 1



if __name__ == '__main__':
    # Demonstration script
