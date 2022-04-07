To run these examples (and dqtpy, in general) you will need to install `pyoptsparse` using `conda`.

Running DTQP and the open-loop to closed-loop control optimization is currently a two step process.

## Running DTQP in WEIS
To run OpenFAST linearizations and DTQP optimizations to genenerate open loop control signals, run
```
python weis_driver.py
```

The `modeling_options.yaml` are configured to linearize across the relevant wind speeds: 
```
Level2:
    flag: True
    simulation:         # could these be included in openfast options?
        flag: False
        TMax: 100.      # run simulations using IEC standards, could share that info
    linearization:
        TMax: 1600.
        DT: 0.01
        TrimGain: 1e-4
        TrimTol: 1e-3
        wind_speeds: [3, 4, 5, 6, 7, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.625, 9.75, 9.875, 10, 10.12, 10.25, 10.38, 10.5, 10.62, 10.75, 10.88, 11, 11.12, 11.25, 11.38, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        rated_offset: 0.75
        DOFs: ['GenDOF','TwFADOF1','PtfmPDOF']
        NLinTimes: 12
    DTQP:
        flag: True
```

The DLCs used in the DTQP optimizations are defined in
```
DLC_driver:
    DLCs:
        - DLC: "1.1"
          ws_bin_size: 2
          wind_speed: [15,18,21]
          n_seeds: 1
          analysis_time: 600.
          transient_time: 100.
          turbulent_wind:
              HubHt: 150.0
              Clockwise: True
              IECturbc: B
              RefHt: 150.0
              PLExp: 0.11
```

If linearizations are already generated, `gen_oloc.py` can be used to run a similar procedure with the same modeling options.
The open-loop trajectories will be placed in `OF_run_dir`, specified in the modeling options.

## Open-loop to closed-loop optimizations
To run an optimization that tracks the open loop trajectories using ROSCO, run 
```
python weis_driver_ol2cl.py
```

The following `modeling_options_ol2cl.yaml` are required:
```
OL2CL:
    flag: True
    trajectory_dir: /Users/dzalkind/Tools/WEIS-4/outputs/13_DTQP_1
```
where the `trajectory_dir` is the directory of the open loop control trajectories and the `DLC_Driver` modeling options must match the ones used to generate the trajectories.

The current example configures `analysis_options_ol2cl.yaml` to optimize the following design variables:
```
design_variables:
  control:
    servo:
      pitch_control:
          omega:
            flag: True
            min: 0.1
            max: 0.5
          zeta:
            flag: True
            min: 0.1
            max: 3.0
          Kp_float:
            flag: False
            min: -40
            max: 0
          ptfm_freq:
            flag: False
            max: 0.4

merit_figure: OL2CL_pitch
```
where the merit figure `OL2CL_pitch` is the average RMS error across DLCs between the open- and closed-loop pitch controls.
