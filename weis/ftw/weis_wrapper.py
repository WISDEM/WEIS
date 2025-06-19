import numpy as np
import os, sys
import warnings
import openmdao.api as om
from weis.glue_code.gc_LoadInputs       import WindTurbineOntologyPythonWEIS
from weis.glue_code.gc_PoseOptimization import PoseOptimizationWEIS
from openmdao.utils.mpi                 import MPI
from openfast_io.FileTools              import save_yaml
from wisdem.inputs.validation           import simple_types
from weis import weis_main

def ftw_training(fname_wt_input, fname_modeling_options, fname_opt_options, geometry_override=None, modeling_override=None, analysis_override=None, test_run=False):
    # Load all yaml inputs and validate (also fills in defaults)
    wt_initial = WindTurbineOntologyPythonWEIS(
        fname_wt_input,
        fname_modeling_options,
        fname_opt_options,
        modeling_override=modeling_override,
        analysis_override=analysis_override
        )
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()

    # Initialize openmdao problem. If running with multiple processors in MPI, use parallel finite differencing equal to the number of cores used.
    # Otherwise, initialize the WindPark system normally. Get the rank number for parallelization. We only print output files using the root processor.
    myopt = PoseOptimizationWEIS(wt_init, modeling_options, opt_options)    

    # Raise error when a proper simulation level is not specified.
    if modeling_options['OpenFAST']['flag']:
    	raise NotImplementedError('Design coupling analysis with OpenFAST response is not implemented yet.')
    elif modeling_options['OpenFAST_Linear']['flag']:
    	raise NotImplementedError('Design coupling analysis with OpenFAST_Linear response is not implemented yet.')
    elif modeling_options['RAFT']['flag']:
    	pass
    else:
    	raise Exception('A simulation level (RAFT, OpenFAST_Linear, OpenFAST) should be enabled for design coupling analysis.')

    # Enable design of experiment driver to obtain training sample data
    opt_options['driver']['optimization']['flag'] = False
    opt_options['driver']['design_of_experiments']['flag'] = True
    if opt_options['driver']['design_of_experiments']['generator'] != 'LatinHypercube':
    	warnings.warn('LatinHypercube generator with sufficient number of samples is recommended.')

    wt_opt_doe, modeling_options_doe, opt_options_doe = weis_main(
    	fname_wt_input, fname_modeling_options, fname_opt_options,
        geometry_override, modeling_override, analysis_override, test_run)

    return wt_opt_doe, modeling_options_doe, opt_options_doe

