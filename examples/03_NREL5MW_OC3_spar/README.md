# 03_NREL5MW_OC3_spar

In this example, you can optimize the ballast volume of the OC3 spar along with the pitch control bandwidth of the ROSCO controller on the NREL-5MW reference turbine.  

Currently, the inputs of this example are set up to test the software using short OpenFAST simulations. 
To run a full optimization

1. Change the following entries in the `modeling_options.yaml` input:

```
    DLCs:
        - DLC: "1.1"
          ws_bin_size: 2
          wind_speed: [14.]
          wave_height: [7.]
          wave_period: [1.]
          n_seeds: 1
          analysis_time: 600.   # These inputs control the simulation length   
          transient_time: 120.    # These inputs control the simulation length
```

2. Change the following entries in the `analysis_options.yaml` input:

```
driver:
  optimization:
    max_iter: 40           # Maximum number of iterations (SLSQP)
```


# Design of Experiments (DOE)
To run a DOE, or parameter sweep, change the following entries in the `analysis_options.yaml` input:

```
driver:
  optimization:
    flag: False
  design_of_experiments:
    flag: True           # Flag to enable design of experiments
```