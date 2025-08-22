## Requirements
The DFSM construction uses matlab, so this is requred: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html


1. Run OpenFAST simulations in operational DLCs, usually 6 seeds, across all wind speeds, or just the ones that are needed (14,16,18).  Only enable the DOFs that you want to include in the DFSM.  Ensure that NcIMURAys is in the openfast outputs



2. 


## Overview
This example introduces the basic workings of the derivative function surrogate modeling (DFSM) approach, and demonstrates a usecase for closed-loop simulations.

Checkout `run_simulation.py` on how to use the DFSM with ROSCO for closed-loop simulation.

The DFSM is available in the `dfsm_1p6.pkl` file. This model has been specifically built for simulating load cases from DLC 1.6.

## Model Description
The DFSM approximates the system response as a linear parameter varying (LPV) state-space model with the following structure:

dx/dt = A(w)x + B(w)u

y = C(w)x + D(w)u

The states (x) considered in the model are platform pitch (PtfmPitch), tower top displacement (fore-aft) (TTDspFA), generator speed (GenSpeed), and their first-time derivatives. 

The controls/inputs (u) are rotor average wind speed (w) (RtVAvgxh), generator torque (GenTq), blade pitch (BldPitch1), and wave elevation (Wave1Elev).

The outputs (y) are tower-base fore-aft shear force (TwrBsFxt), side-to-side moment (TwrBsMyt), tower top translational and rotational accelerations (YawBrTAxp, NcIMURAys), and the generator power (GenPwr).

## Model Development

 
 
