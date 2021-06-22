import numpy as np
import os
import weis.control.LinearModel as lin_mod
from weis.aeroelasticse.LinearFAST import LinearFAST


def gen_level2_model(dofs=['GenDOF']):
    lin_fast = LinearFAST(dev_branch=True)

    # fast info
    lin_fast.weis_dir                 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
    
    lin_fast.FAST_InputFile           = self.FAST_InputFile   # FAST input file (ext=.fst)
    lin_fast.FAST_directory           = self.FAST_directory
    lin_fast.FAST_linearDirectory     = os.path.join(lin_fast.weis_dir,'outputs','iea_semi_lin')
    lin_fast.FAST_steadyDirectory     = os.path.join(lin_fast.weis_dir,'outputs','iea_semi_steady')
    self.FAST_level2_Directory        = lin_fast.FAST_linearDirectory
    lin_fast.debug_level              = 2
    lin_fast.dev_branch               = True
    lin_fast.write_yaml               = True
    
    lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
    lin_fast.WindSpeeds                 = self.level2_wind_speeds
    lin_fast.DOFs                       = dofs  # enable with 
    lin_fast.TMax                       = 1600   # should be 1000-2000 sec or more with hydrodynamic states
    lin_fast.NLinTimes                  = 12

    # lin_fast.FAST_exe                   = '/Users/dzalkind/Tools/openfast-dev/install/bin/openfast'

    # simulation setup
    lin_fast.cores                      = self.n_cores
    
    self.GBRatio          = 1.0
    self.fst_vt           = fastRead.fst_vt

    # overwrite steady & linearizations
    # lin_fast.overwrite        = True
    
    # run OpenFAST linearizations
    lin_fast.gen_linear_model()

    LinearTurbine = lin_mod.LinearTurbineModel(mf_turb.FAST_level2_Directory,reduceStates=False)
    
    
get_level2_model()