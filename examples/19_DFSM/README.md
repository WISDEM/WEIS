## Requirements
The DFSM construction uses matlab, so the matlab engine is requred: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.  The MATLAB optimization toolbox is required.

## Usage

0. Run OpenFAST simulations in operational DLCs (ideally 1.6, 6 seeds), across the wind speeds you want to simulate. Only enable the DOFs that you want to include in the DFSM.  Ensure that NcIMURAys is in the openfast outputs so it can be used by the rosco control interface.

1. Run python construct_LPV_matlab.py.  The MATLAB optimization toolbox is required.  This process is not yet tested, but several DFSM pickles are available in ``dfsm_models/``.

2. Use DFSM models within WEIS.  See the modeling options under ``DFSM`` for an example.  


## Overview
This example introduces the basic workings of the derivative function surrogate modeling (DFSM) approach, and demonstrates a usecase for closed-loop simulations.

## What is the DFSM
The DFSM is a low-fidelity model that is built to approximate OpenFAST response, and can be used to simulate the turbine.
The DFSM is a linear parameter varying (LPV) state-space model with the following structure:

dx/dt = A(w)x + B(w)u

y = C(w)x + D(w)u

The states (x) considered in the model are platform pitch (PtfmPitch), platform heave (PtfmHeave), generator speed (GenSpeed), and their first-time derivatives. 

The controls/inputs (u) are rotor average wind speed (w) (RtVAvgxh), generator torque (GenTq), blade pitch (BldPitch1), and wave elevation (Wave1Elev).

The outputs (y) are tower-base fore-aft shear force (TwrBsFxt), side-to-side moment (TwrBsMyt), tower top translational and rotational accelerations (YawBrTAxp, NcIMURAys), generator power (GenPwr), and Fluid Cp and Ct.

## Constructing the DFSM

The DFSM is constructed using OpenFAST simulations.
Linear state-space matrices (A,B,C,D) are identified for different wind/current speeds (w), and the matrices are interpolated over this to get a continuous model.
Suppose we want the DFSM to predict the system response for w = [2.5,2.75,3.0] [m/s], we need to simulate the turbine model using OpenFAST for these current speeds.
Typically 5 different seeds for each wind/current speed are used to construct the DFSM.
An optimization problem is formulated to identify the model parameters associated with the A,B,C,D matrices.
Currently we use the optimization toolbox in MATLAB to solve this problem.


More details regarding the preprocessing and contstruction steps are available in the `Preprocessing.ipynb` and `DFSM.ipynb` notebooks.
Check these notebooks before running `construct_DFSM.py`.

## dfsm_mhk.pkl
A pre constructed model is available in the `dfsm_mhk.pkl`. The states, controls and outputs included as part of this model are listed above.

The core of the DFSM model are five arrays, containing the list of current speeds, and the A,B,C,D matrices corresponding to them.
Using these arrays, and an interpolating scheme, the LPV model can be set up. The default interpolating scheme is linear.


Once constructed the DFSM can be used for closed-loop simulations along with ROSCO, and it can be used for controller optimization studies like the one in `example 03_control_opt`.
Checkout `weis_driver_dfsm_mhk.py` use the DFSM with ROSCO for closed-loop simulation.
Change the optimization flags in `analysis_options_dfsm_mhk.yaml` to run a controller optimization study.

## Developer Notes
- DFSM libraries are located in ``weis/dfsm``
- DFSM pickles contain the class object (weis.dfsm.construct_dfsm.DFSM)

 
