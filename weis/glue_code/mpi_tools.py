import os
import sys
import numpy as np
from openmdao.utils.mpi import MPI

def compute_optimal_nP(nFD, nOF, modeling_options, opt_options, maxnP=1):
    
    
    fd_methods = ['SLSQP','SNOPT', 'LD_MMA']
    evolutionary_methods = ['DE', 'NSGA2']
    
    print("\nYour problem has %d design variable(s) and %d OpenFAST run(s)\n"%(nFD, nOF))

    if modeling_options['Level3']['flag']:
        # If we are running an optimization method that doesn't use finite differencing, set the number of DVs to 1
        if not (opt_options['driver']['design_of_experiments']['flag']) and (opt_options['driver']['optimization']['solver'] in evolutionary_methods):
            print("You are running an optimization based on an evolutionary solver. The number of parallel finite differencing is now multiplied by 5\n")
            nFD *= 5  # targeting 10*n_DV population size... this is what the equivalent FD coloring would take
        elif not (opt_options['driver']['design_of_experiments']['flag'] or opt_options['driver']['optimization']['solver'] in fd_methods):
            print("You are not running a design optimization, a design of experiment, or your optimizer is not gradient based. The number of parallel function evaluations is set to 1\n")
            nFD = 1
    elif modeling_options['Level2']['flag']:
        if not (opt_options['driver']['design_of_experiments']['flag'] or opt_options['driver']['optimization']['solver'] in fd_methods):
            print("You are not running a design optimization, a design of experiment, or your optimizer is not gradient based. The number of parallel function evaluations is set to 1\n")
            nFD = 1
    # # if we're doing a GA or such, "FD" means "entities in epoch"
    # if opt_options['driver']['optimization']['solver'] in evolutionary_methods:
    #     nFD = maxnP


    print("To run the code in parallel with MPI, execute one of the following commands\n")

    if maxnP != 1:
        print("You have access to %d processors. Please call WEIS as:"%maxnP)
        # Define the color map for the parallelization, determining the maximum number of parallel finite difference (FD)
        # evaluations based on the number of design variables (DV). OpenFAST on/off changes things.
        if modeling_options['Level3']['flag']:
            # If openfast is called, the maximum number of FD is the number of DV, if we have the number of processors available that doubles the number of DVs,
            # otherwise it is half of the number of DV (rounded to the lower integer).
            # We need this because a top layer of processors calls a bottom set of processors where OpenFAST runs.
            if maxnP > 2. * nFD:
                nFD = nFD
            else:
                nFD = int(np.floor(maxnP / 2))
            # Get the number of OpenFAST runs from the user input and the max that can run in parallel given the resources
            # The number of OpenFAST runs is the minimum between the actual number of requested OpenFAST simulations, and
            # the number of processors available (minus the number of DV, which sit and wait for OF to complete)
            nFD = max([nFD, 1])
            max_parallel_OF_runs = max([int(np.floor((maxnP - nFD) / nFD)), 1])
            nOFp = min([int(nOF), max_parallel_OF_runs])
        elif modeling_options['Level2']['flag']:
            if maxnP > 2. * nFD:
                nFD = nFD
            else:
                nFD = int(np.floor(maxnP / 2))
            nFD = max([nFD, 1])
            max_parallel_OF_runs = max([int(np.floor((maxnP - nFD) / nFD)), 1])
            nOFp = min([int(nOF), max_parallel_OF_runs])
        else:
            # If OpenFAST is not called, the number of parallel calls to compute the FDs is just equal to the minimum of processors available and DV
            nFD = min([maxnP, nFD])
            nOFp = 1
        nP = nFD + nFD * nOFp
    
    else:
        nOFp = nOF
        nP = nFD + nFD * nOFp
        print("If you have access to (at least) %d processors, please call WEIS as:"%nP)
    
    print("mpirun -np %d python weis_driver.py\n"%nP)

    if maxnP == 1:
        print("\nIf you do not have access to %d processors"%nP)
        print("please provide your maximum available number of processors by typing:")
        print("python weis_driver.py --preMPI=True --maxnP=xx")
        print("And substitute xx with your number of processors\n")

    modeling_options['General']['openfast_configuration']['nP'] = nP
    modeling_options['General']['openfast_configuration']['nFD'] = nFD
    modeling_options['General']['openfast_configuration']['nOFp'] = nOFp

    return modeling_options

def under_mpirun():
    """Return True if we're being executed under mpirun."""
    # this is a bit of a hack, but there appears to be
    # no consistent set of environment vars between MPI
    # implementations.
    for name in os.environ.keys():
        if (
            name == "OMPI_COMM_WORLD_RANK"
            or name == "MPIEXEC_HOSTNAME"
            or name.startswith("MPIR_")
            or name.startswith("MPICH_")
            or name.startswith("INTEL_ONEAPI_MPI_")
            or name.startswith("I_MPI_")
        ):
            return True
    return False


if under_mpirun():

    def debug(*msg):  # pragma: no cover
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write("\n")
        sys.stdout.flush()

else:
    MPI = None


def map_comm_heirarchical(nFD, n_OF):
    """
    Heirarchical parallelization communicator mapping.  Assumes a number of top level processes
    equal to the number of finite differences, each
    with its associated number of openfast simulations.
    """
    N = nFD + nFD * n_OF
    comm_map_down = {}
    comm_map_up = {}
    color_map = [0] * nFD

    for i in range(nFD):
        comm_map_down[i] = [nFD + j + i * n_OF for j in range(n_OF)]
        color_map.extend([i + 1] * n_OF)

        for j in comm_map_down[i]:
            comm_map_up[j] = i

    return comm_map_down, comm_map_up, color_map


def subprocessor_loop(comm_map_up):
    """
    Subprocessors loop, waiting to receive a function and its arguements to evaluate.
    Output of the function is returned.  Loops until a stop signal is received

    Input data format:
    data[0] = function to be evaluated
    data[1] = [list of arguments]
    If the function to be evaluated does not fit this format, then a wrapper function
    should be created and passed, that handles the setup, argument assignment, etc
    for the actual function.

    Stop sigal:
    data[0] = False
    """
    # comm        = impl.world_comm()
    rank = MPI.COMM_WORLD.Get_rank()
    rank_target = comm_map_up[rank]

    keep_running = True
    while keep_running:
        data = MPI.COMM_WORLD.recv(source=(rank_target), tag=0)
        if data[0] == False:
            break
        else:
            func_execution = data[0]
            args = data[1]
            output = func_execution(args)
            MPI.COMM_WORLD.send(output, dest=(rank_target), tag=1)


def subprocessor_stop(comm_map_down):
    """
    Send stop signal to subprocessors
    """
    # comm = MPI.COMM_WORLD
    for rank in comm_map_down.keys():
        subranks = comm_map_down[rank]
        for subrank_i in subranks:
            MPI.COMM_WORLD.send([False], dest=subrank_i, tag=0)
        print("All MPI subranks closed.")


if __name__ == "__main__":

    (
        _,
        _,
        _,
    ) = map_comm_heirarchical(2, 4)
