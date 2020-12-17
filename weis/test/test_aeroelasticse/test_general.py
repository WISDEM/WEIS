"""

Example script to compute the steady-state performance in OpenFAST

"""

import copy
import os
import platform
import sys
import unittest

import numpy as np

import weis
from openmdao.utils.assert_utils import assert_near_equal
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from weis.aeroelasticse.runFAST_pywrapper import (runFAST_pywrapper,
                                                  runFAST_pywrapper_batch)
from weis.test.utils import compare_regression_values


class TestGeneral(unittest.TestCase):
    def test_run(self):
        # Paths calling the standard modules of WEIS
        fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
        run_dir1 = (
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                )
            )
            + os.sep
        )
        fastBatch.FAST_exe          = os.path.join(run_dir1, 'local','bin','openfast')   # Path to executable
        run_dir2 = (
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                )
            )
            + os.sep
            + "examples"
            + os.sep
            + "01_aeroelasticse"
            + os.sep
        )
        fastBatch.FAST_directory    = os.path.join(run_dir2, 'OpenFAST_models','IEA-15-240-RWT','IEA-15-240-RWT-Monopile')   # Path to fst directory files
        fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
        fastBatch.FAST_runDirectory = 'steady_state/iea15mw'
        fastBatch.debug_level       = 2

        # User settings
        n_cores     = 1     # Number of available cores
        TMax        = 1.  # Length of wind grids and OpenFAST simulations, suggested 720 s
        cut_in      = 3.    # Cut in wind speed
        cut_out     = 25.   # Cut out wind speed
        n_ws        = 2    # Number of wind speed bins
        wind_speeds = np.linspace(cut_in, cut_out, int(n_ws)) # Wind speeds to run OpenFAST at
        Ttrans      = max([0., TMax - 60.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
        TStart      = max([0., TMax - 300.]) # Start of the recording of the channels of OpenFAST

        # Initial conditions for ElastoDyn
        # Initial conditions to start the OpenFAST runs
        u_ref = np.arange(8.0, 10.0)  # Wind speed
        pitch_ref = [0.0, 0.5]  # Pitch values in deg
        omega_ref = [5.5, 6.0]  # Rotor speeds in rpm
        pitch_init = np.interp(wind_speeds, u_ref, pitch_ref)
        omega_init = np.interp(wind_speeds, u_ref, omega_ref)

        # Settings passed to OpenFAST
        case_inputs = {}
        case_inputs[("Fst","TMax")]             = {'vals':[TMax], 'group':0}
        case_inputs[("Fst","DT")]               = {'vals':[0.01], 'group':0}
        case_inputs[("Fst","CompInflow")]       = {'vals':[1], 'group':0}
        case_inputs[("Fst","CompServo")]        = {'vals':[1], 'group':0}
        case_inputs[("Fst","OutFileFmt")]       = {'vals':[1], 'group':0}
        case_inputs[("Fst","DT_Out")]           = {'vals':[0.02], 'group':0}
        case_inputs[("ElastoDyn","GenDOF")]     = {'vals':['True'], 'group':0}
        case_inputs[("ServoDyn","PCMode")]      = {'vals':[5], 'group':0}
        case_inputs[("ServoDyn","VSContrl")]    = {'vals':[5], 'group':0}
        case_inputs[("InflowWind","WindType")]  = {'vals':[1], 'group':0}
        case_inputs[("InflowWind","HWindSpeed")]= {'vals': wind_speeds, 'group': 1}
        case_inputs[("Fst","OutFileFmt")]       = {'vals':[0], 'group':0}
        case_inputs[("ElastoDyn","RotSpeed")]   = {'vals': omega_init, 'group': 1}
        case_inputs[("ElastoDyn","BlPitch1")]   = {'vals': pitch_init, 'group': 1}
        case_inputs[("ElastoDyn","BlPitch2")]   = case_inputs[("ElastoDyn","BlPitch1")]
        case_inputs[("ElastoDyn","BlPitch3")]   = case_inputs[("ElastoDyn","BlPitch1")]

        # Find the controller
        if platform.system() == 'Windows':
            sfx = 'dll'
        elif platform.system() == 'Darwin':
            sfx = 'dylib'
        else:
            sfx = 'so'
        path2dll = os.path.join(run_dir1, 'local','lib','libdiscon.'+sfx)

        case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}

        # Generate the matrix of cases
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase='iea15mw')

        fastBatch.case_list = case_list
        fastBatch.case_name_list = case_name_list

        # Run OpenFAST, either serially or sequentially
        out = fastBatch.run_serial()
            
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        compare_regression_values(out, 'general_regression_values.pkl', directory=this_file_dir, tol=1e-3, train=False)

if __name__ == "__main__":
    unittest.main()
