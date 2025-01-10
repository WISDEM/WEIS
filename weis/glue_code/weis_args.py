import argparse
from openmdao.utils.mpi  import MPI

def weis_args():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run WEIS driver with flag prepping for MPI run.")
    # Add the flag
    parser.add_argument("--preMPI", type=bool, default=False, help="Flag for preprocessing MPI settings (True or False).")
    parser.add_argument("--maxnP", type=int, default=1, help="Maximum number of processors available.")
    # Parse the arguments
    args, _ = parser.parse_known_args()
    # Use the flag in your script
    if args.preMPI:
        print("Preprocessor flag is set to True. Running preprocessing setting up MPI run.")
    else:
        print("Preprocessor flag is set to False. Run WEIS now.")

    return args

def get_max_procs(args):
    # Set max number of processes, either set by user or extracted from MPI
    if args.preMPI:
        maxnP = args.maxnP
    else:
        if MPI:
            maxnP = MPI.COMM_WORLD.Get_size()
        else:
            maxnP = 1
    return maxnP

def set_modopt_procs(modeling_options):
    print('Applying the modeling option updates as overrides.')
    modeling_override = {}
    modeling_override['General'] = {}
    modeling_override['General']['openfast_configuration'] = {}
    modeling_override['General']['openfast_configuration']['nFD'] = modeling_options['General']['openfast_configuration']['nFD']
    modeling_override['General']['openfast_configuration']['nOFp'] = modeling_options['General']['openfast_configuration']['nOFp']
    return modeling_override
