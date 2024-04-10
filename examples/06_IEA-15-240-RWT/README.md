This example runs several IEA-15MW RWT-related cases:
- `weis_driver.py` runs an analysis (not optimization) of the monopile version of the IEA 15MW. 
- `weis_driver_umaine_semi.py` performs an optimization of the IEA 15MW equipped with the UMaine VolturnUS-S floating platform.
- `weis_driver_TMDs.py` performs an optimization of the tuned mass damper (TMD) properties for the floating platform version of IEA 15MW.
- `weis_driver_tower_DVs.py` performs a tower sizing optimization of the floating platform version of IEA 15MW with TMDs included.

Currently, the inputs of this example are set up to test the software using short OpenFAST simulations. 
To run a full optimization

1. Change the following entries in the `modeling_options*.yaml` input:

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

2. (If optimizing) Change the following entries in the `analysis_options*.yaml` input:

```
driver:
  optimization:
    max_iter: 40           # Maximum number of iterations (SLSQP)
```


