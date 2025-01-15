There are multiple examples in this folder focused on running WEIS at level 2 (linearized OpenFAST) using the IEA 15MW wind turbine as the testbed geometry.

 


# WEIS Linearizations
To linearize a WEIS model using OpenFAST, run
 ```
 python weis_driver.py
 ```
 
 No `analysis_options.yaml` design variable flags should be active and the analysis drivers should be disabled. 
 In `modeling_options.yaml`, a few linearization options can be set up:
```
OpenFAST_Linear:
    flag: True          # Enable to run linearizations
    simulation:        
        flag: True
        TMax: 600.              # for running a linear simulation using the linearized models
    linearization:
        TMax: 700.              # Maximum time that linearization simulation will be run. More states (DOFs) take more time to converge
        DT: 0.01
        TrimGain: 1e-4          # OpenFAST option that affects steady state trim solution convergence
        TrimTol: 1e-3           # OpenFAST option that affects steady state trim solution convergence
        wind_speeds: [14.,16.]  # Wind speeds where linearizations occur
        rated_offset: 0.75      # WEIS computes the rated wind speed assuming a stiff rotor.  When more DOFs are enabled, the rated wind speed could be higher and this value will increase the rated wind speed used to select torque/pitch for trimming the steady state system
        DOFs: ['GenDOF','TwFADOF1','PtfmPDOF']      # DOFs used in OpenFAST simulation.  Will become linear system states
        NLinTimes: 2            # Number of azimuth positions linearizations are peformed (12 is recommended)
```

The class used to set up OpenFAST linearizations is in `weis/aerelasticse/LinearFAST.py`; it can be set up to run more general linearization cases.
The class used to post process and join linearization is in `weis/control/LinearModel.py`; it can be used to run linear simulations.
Linearizations will be generated in the directory specified by the `OF_run_dir` modeling option.
 
 # Linearizations design of experiments
 Parameter sweeps of different turbine model options can be set up and linearized using 
  ```
 python doe_driver.py
  ```
  
The design variables determine the parameters to be swept.
