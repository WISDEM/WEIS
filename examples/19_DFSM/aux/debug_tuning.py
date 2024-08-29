import os 
import sys 
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
import shutil
import matplotlib.pyplot as plt

# ROSCO toolbox modules 
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox.utilities import write_DISCON
from rosco import discon_lib_path
from rosco.toolbox.inputs.validation import load_rosco_yaml


# DFSM modules
from weis.dfsm.ode_algorithms import RK4


# WEIS specific modules
import weis.inputs as sch
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.dfsm.generate_wave_elev import generate_wave_elev
import time as timer


this_dir = os.path.dirname(os.path.realpath(__file__))
weis_dir = os.path.dirname(os.path.dirname(this_dir))


if __name__ == '__main__':

    # Write parameter input file
    cp_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_Cp_Ct_Cq.txt')
    param_filename = os.path.join(this_dir,'IEA_15_MW','IEA_w_TMD_DISCON.IN')

    tuning_yaml = os.path.join(this_dir,'IEA_15_MW','IEA15MW-tuning.yaml')
    inputs = load_rosco_yaml(tuning_yaml)
    path_params         = inputs['path_params']
    turbine_params      = inputs['turbine_params']
    controller_params   = inputs['controller_params']


    turbine = ROSCO_turbine.Turbine(turbine_params)
    turbine.v_rated = 10.555555555555555
    #breakpoint()
    turbine.load_from_fast(
        path_params['FAST_InputFile'],
        os.path.join(this_dir,path_params['FAST_directory']),
        rot_source='txt',txt_filename=cp_filename
        )
    
    # Tune controller 
    controller      = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)

    # Write parameter input file
    param_filename = os.path.join(this_dir,'DISCON_1.IN')
    write_DISCON(
      turbine,controller,
      param_file=param_filename, 
      txt_filename=cp_filename
      )
    
    #-------------------------------------------------------------
    tuning_yaml2 = os.path.join(this_dir,'../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi/IEA15MW-UMaineSemi.yaml')
    inputs2 = load_rosco_yaml(tuning_yaml2)

    path_params2         = inputs2['path_params']
    turbine_params2      = inputs2['turbine_params']
    controller_params2   = inputs2['controller_params']
    cp_filename = os.path.join(this_dir,'../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi',path_params2['rotor_performance_filename'])


    turbine2 = ROSCO_turbine.Turbine(turbine_params2)
    turbine2.v_rated = 10.64
    turbine2.load_from_fast(
        path_params2['FAST_InputFile'],
        path_params2['FAST_directory'],
        rot_source='txt',txt_filename=cp_filename
        )
    
    controller2      = ROSCO_controller.Controller(controller_params2)
    controller2.tune_controller(turbine2)

    param_filename2 = os.path.join(this_dir,'DISCON_2.IN')
    write_DISCON(
      turbine2,controller2,
      param_file=param_filename2, 
      txt_filename=cp_filename
      )

    breakpoint()
