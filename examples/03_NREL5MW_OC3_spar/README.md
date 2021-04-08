# 03_NREL5MW_OC3_spar

In this example, you can optimize the ballast volume of the OC3 spar along with the pitch control bandwidth of the ROSCO controller on the NREL-5MW reference turbine.  

Currently, the inputs of this example are set up to test the software using short OpenFAST simulations.  

To run the full optimization, make sure that you install the NLOpt solvers in your weis environment using 

```
conda install nlopt
```

and set the following inputs in the `modeling_options.yaml` input:

```
Level3: # Options for WEIS fidelity level 3 = nonlinear time domain
    flag: True
    simulation:
        TMax: 720.0
        TStart: 120.0
```