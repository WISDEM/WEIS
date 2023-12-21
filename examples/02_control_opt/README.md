These examples all run a previously generated OpenFAST model.  The models are not automatically generated from WISDEM geometry and will not change, except for different load cases.

Most of the geometry options are unused in these examples.  However, you may want to set some of the `assembly` options so that turbulent wind inputs are properly generated for the desired class and rotor diameter.

The OpenFAST model is specified in the following modeling options:
```
Level3: # Options for WEIS fidelity level 3 = nonlinear time domain
    flag: True
    from_openfast: True
    openfast_file: IEA-15-240-RWT-UMaineSemi.fst
    openfast_dir: examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi
```
Further options provided in `Level3` will override the existing OpenFAST model.  Some inputs will be changed depending on the design load case (DLC).

DLCs are specified in the `DLC_driver` modeling options, e.g.,
```
DLC_driver:
    DLCs:
        - DLC: "1.1"
          wind_speed: [16]
          n_seeds: 1
          analysis_time: 10.
          transient_time: 0.1
          turbulent_wind:
              HubHt: 142
              RefHt: 142
              GridHeight: 220
              GridWidth: 220
        - DLC: "6.1"
          analysis_time: 10.
          transient_time: 0.1
          turbulent_wind:
              HubHt: 142
              RefHt: 142
              GridHeight: 220
              GridWidth: 220
              URef: 46.789
              PLExp: 0.14
              IECturbc: 0.12
```
The exact DLC cases run can be found in the `case_matrix.txt` file where the OpenFAST files are written and run.

The analysis options determine what kind of WEIS run is performed.  `analysis_options_loads.yaml` will not optimize any parameters; it will only run the load cases specified. `analysis_options.yaml` is configured to optimize a floating controller in ROSCO. The example is only configured to run 2 iterations, but this can be increased with the `max_iter` option.  `analysis_options_sm.yaml` will optimze a controller based on its robust stability margin.


To run these examples, activate your WEIS conda environment and run
```
python weis_driver.py
```
or the appropriate `weis_driver<script>.py`
