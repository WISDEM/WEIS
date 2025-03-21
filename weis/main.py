import os
import sys
import time
import warnings
import numpy as np

from weis.inputs.gui import run as guirun
from weis.glue_code.runWEIS import run_weis
import weis.inputs as sch
from openmdao.utils.mpi  import MPI

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning, lineno=175)
warnings.simplefilter("ignore", RuntimeWarning, lineno=177)

def get_max_procs():
    return MPI.COMM_WORLD.Get_size() if MPI else 1

def set_modopt_procs(modeling_options,modeling_override):
    print('Applying the modeling option updates as additional overrides.')
    mpi_modeling_override = {}
    mpi_modeling_override['General'] = {}
    mpi_modeling_override['General']['openfast_configuration'] = {}
    mpi_modeling_override['General']['openfast_configuration']['nFD'] = modeling_options['General']['openfast_configuration']['nFD']
    mpi_modeling_override['General']['openfast_configuration']['nOFp'] = modeling_options['General']['openfast_configuration']['nOFp']

    modeling_override.update(mpi_modeling_override)
    return mpi_modeling_override

def set_modopt_test_runs(fname_input_modeling, modeling_override, fname_input_analysis, analysis_override):
    # Load modeling options
    modeling_options = sch.load_modeling_yaml(fname_input_modeling)
    analysis_options = sch.load_analysis_yaml(fname_input_analysis)

    # Options to speed up tests:

    # Shorten all DLC runs to single metocean condition
    for dlc_option in modeling_options['DLC_driver']['DLCs']:
        dlc_option['transient_time'] = 0
        dlc_option['analysis_time'] = 0.5
        dlc_option['n_seeds'] = 1
        dlc_option['wind_speed'] = [10]
        if dlc_option['wave_period']: # if user specified array is used
            dlc_option['wave_period'] = [8]
        if dlc_option['wave_height']: # if user specified array is used
            dlc_option['wave_height'] = [2]

    # OpenFAST modeling_overrides are not honored if an OpenFAST model is read, but we can set options in openmdao_openfast for now if this flag is enabled
    modeling_options['General']['test_mode'] = True

    # Cp surface generation: default is 20, ROSCO can struggle with fewer than 10
    modeling_options['WISDEM']['RotorSE']['n_pitch_perf_surfaces'] = 10
    modeling_options['WISDEM']['RotorSE']['n_tsr_perf_surfaces'] = 10

    # Solver, max iterations
    analysis_options['driver']['optimization']['max_iter'] = 1
    analysis_options['driver']['optimization']['solver'] = 'LN_COBYLA'   # Gradient free

    modeling_override.update(modeling_options)  # not sure whether to return all options as overrides
    analysis_override.update(analysis_options) 


def weis_main(fname_wt_input, fname_modeling_options, fname_analysis_options,
              geometry_override={}, modeling_override={}, analysis_override={}, test_run = False):

    tt = time.time()
    maxnP = get_max_procs()

    if test_run:
        set_modopt_test_runs(fname_modeling_options, 
                            modeling_override,
                            fname_analysis_options,
                            analysis_override
                            )

    # If running in parallel, pre-compute number of cores needed in this run
    if MPI:
        _, modeling_options, _ = run_weis(fname_wt_input,
                                          fname_modeling_options, 
                                          fname_analysis_options, 
                                          geometry_override=geometry_override,
                                          modeling_override=modeling_override,
                                          analysis_override=analysis_override,
                                          prepMPI=True, 
                                          maxnP=maxnP)

        modeling_override = set_modopt_procs(modeling_options)

    # Run WEIS for real now
    wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, 
                                                     fname_modeling_options, 
                                                     fname_analysis_options,
                                                     geometry_override=geometry_override,
                                                     modeling_override=modeling_override,
                                                     analysis_override=analysis_override,
                                                     prepMPI=False,
                                                     maxnP=maxnP)

    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0
        
    if rank == 0:
        output_name = os.path.split(opt_options['general']['folder_output'])[-1]  # maybe temporary hack to profile examples/tests
        print(f"Run time ({output_name}): {time.time()-tt}")
        sys.stdout.flush()
    
    return wt_opt, modeling_options, opt_options

    
def weis_cmd():
    usg_msg = "WEIS command line launcher\n    Arguments: \n    weis : Starts GUI\n    weis geom.yaml modeling.yaml analysis.yaml : Runs specific geometry, modeling, and analysis files\n"

    # Look for help message
    help_flag = False
    for k in range(len(sys.argv)):
        if sys.argv[k] in ["-h", "--help"]:
            help_flag = True

    if help_flag:
        print(usg_msg)

    elif len(sys.argv) == 1:
        # Launch GUI
        guirun()

    elif len(sys.argv) == 4:
        check_list = ["geometry", "modeling", "analysis"]
        for k, f in enumerate(sys.argv[1:]):
            if not os.path.exists(f):
                raise FileNotFoundError("The " + check_list[k] + " file, " + f + ", cannot be found.")

        # Run WEIS (also saves output)
        wt_opt, modeling_options, opt_options = weis_main(sys.argv[1], sys.argv[2], sys.argv[3])

    else:
        # As if asked for help
        print("Unrecognized set of inputs.  Usage:")
        print(usg_msg)

    sys.exit(0)


if __name__ == "__main__":
    weis_cmd()
