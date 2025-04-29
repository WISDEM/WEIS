.. _section-weis_outputs:

WEIS Outputs
=============

File Structure & Location
--------------------------

The WEIS outputs are determined by the way the software configures ``analysis_options`` and ``modeling_options`` files.
The name and path of WEIS output files are determined by the variables ``general/folder_output`` and ``general/fname_output`` from the ``analysis_options`` file.
Below is an example of file structure from WEIS outputs with OpenFAST-based optimization.

::

   WEIS Outputs
        ├── problem_vars.yaml               # WEIS - Design variables, constraints, and objectives
        ├── refturb_output-analysis.yaml    # WEIS - Final analysis options
        ├── refturb_output-modeling.yaml    # WEIS - Final modeling options
        ├── refturb_output.yaml             # WEIS - Final turbine model
        ├── refturb_output.mat              # WISDEM/OpenMDAO - Results in MATLAB format
        ├── refturb_output.npz              # WISDEM/OpenMDAO - Results in NumPy archive format
        ├── refturb_output.pkl              # WISDEM/OpenMDAO - Results in Python pickle format
        ├── refturb_output.xlsx             # WISDEM/OpenMDAO - Results in Excel format
        ├── refturb_output.csv              # WISDEM/OpenMDAO - Results in CSV format
        ├── case_matrix_combined.txt        # DLC - Combined definitions (text)
        ├── case_matrix_combined.yaml       # DLC - Combined definitions (YAML)
        ├── case_matrix_DLC1.6_0.txt        # DLC - Specific definitions (text)
        ├── case_matrix_DLC1.6_0.yaml       # DLC - Specific definitions (YAML)
        ├── log_opt.sql                     # Optimization history database
        └── openfast_runs/                  # OpenFAST simulation directory
                 ├── DLC1.6_0_AeroDyn_blade.dat             # Blade aerodynamics
                 ├── DLC1.6_0_AeroDyn.dat                   # General aerodynamics
                 ├── DLC1.6_0_ElastoDyn_tower.dat           # Tower structural properties
                 ├── DLC1.6_0_ElastoDyn.dat                 # Main structural properties
                 ├── DLC1.6_0_ElastoDynBlade.dat            # Blade structural properties
                 ├── DLC1.6_0_HydroDyn.dat                  # Hydrodynamic properties
                 ├── DLC1.6_0_InflowWind.dat                # Wind conditions
                 ├── DLC1.6_0_MoorDyn.dat                   # Mooring system properties
                 ├── DLC1.6_0_SeaState.dat                  # Ocean conditions
                 ├── DLC1.6_0_ServoDyn.dat                  # Controller configuration
                 ├── Airfoils/                              # Airfoil data
                 │      ├── DLC1.6_0_AeroDyn_Polar_00.dat   # Airfoil polar data
                 │      └── DLC1.6_0_AeroDyn_Polar_01.dat   # Airfoil polar data
                 ├── wind/                                  # Wind field data
                 ├── DLC1.6_0.fst                           # Main OpenFAST input file
                 ├── DLC1.6_0.out                           # Text output results
                 ├── DLC1.6_0.outb                          # Binary output results
                 ├── DLC1.6_0.sum                           # Summary results
                 │
                 ├── iteration_0/                           # Optimization iteration output
                 │      ├── DELs.p                          # Damage Equivalent Loads pickle file
                 │      ├── fst_vt.p                        # OpenFAST variable tree pickle file
                 │      ├── summary_stats.p                 # Summary statistics pickle file
                 │      └── timeseries/                     # OpenFAST Timeseries Output
                 │
                 └── iteration_1/                           # Optimization iteration output
                        ├── DELs.p
                        ├── fst_vt.p
                        ├── summary_stats.p
                        └── timeseries/
 

OpenFAST-related
-----------------

The OpenFAST-related files are located under the ``openfast_runs/`` folder, containing the dynamic simulation outputs from aero-servo-hydro-elastic modeling.
OpenFAST requires multiple input files to run the simulations, which are automatically generated within WEIS.

Input Files
^^^^^^^^^^^

    * *.fst
        * This is the main OpenFAST input file that serves as the entry point for simulations. It references all other module input files and specifies general simulation settings including time step, simulation duration, and enabled modules.
    
    * *.dat
        * These are module-specific input files that contain parameters for different subsystems of the wind turbine model, a few examples include:
            * *_AeroDyn.dat - Aerodynamic settings and parameters
            * *_ElastoDyn.dat - Structural properties and initial conditions
            * *_ServoDyn.dat - Control system configuration
            * *_HydroDyn.dat - Hydrodynamic loading parameters
            * *_MoorDyn.dat - Mooring system configuration
            * *_InflowWind.dat - Wind conditions specification

    * Airfoils/*.dat
        * These input files define the airfoil lift, drag, and moment coefficients as functions of angle of attack, Reynolds number, and other parameters. These data are crucial for accurate aerodynamic calculations.

    * wind/*.in
        * Input files for TurbSim/deterministic inflow that define the wind field characteristics including mean wind speed, turbulence intensity, wind shear, and spatial coherence parameters.


Output Files
^^^^^^^^^^^

OpenFAST generates several output files containing simulation results:

    * *.out
        * Text-based time-series output containing columns of data for requested output channels. These files provide detailed information on turbine states and loads throughout the simulation.
    
    * *.outb
        * Binary version of the .out file with the same data but in a more compact and efficient format for large simulations.
    
    * *.sum
        * This file contains a summary of the various modules used in the OpenFAST simulation. It serves as a quick reference for the configuration and settings of each module.


For more information on OpenFAST input and output files, refer to the `OpenFAST documentation <https://openfast.readthedocs.io/en/main/>`_.


WISDEM/WEIS-related
--------------------

These files contain the integrated outputs from the WEIS framework, providing comprehensive results on the wind turbine's performance, structural properties, and system characteristics.

Output Files
^^^^^^^^^^^

    * refturb_output.yaml, refturb_output-analysis.yaml, refturb_output-modeling.yaml
        * These are the final updated versions of the WEIS input files, reflecting any changes made during the analysis or optimization process. The base filename (refturb_output) is determined by the ``general/fname_output`` setting in the ``analysis_options.yaml`` file.

    * problem_vars.yaml
        * This file documents the optimization problem formulation, including design variables with their bounds, constraints with their values and margins, and objectives with their weights.

    * refturb_output.* (mat/npz/pkl/xlsx/csv)
        * These files contain the same output data in different formats for compatibility with various analysis tools:
            * .mat - MATLAB format
            * .npz - NumPy compressed archive
            * .pkl - Python pickle format
            * .xlsx - Excel spreadsheet
            * .csv - Comma-separated values text file


DLC-related
------------

Design Load Case (DLC) files document the simulation cases that were run:

    * case_matrix_combined.txt/yaml
        * These files contain the complete set of all DLC simulations performed, with each row describing a unique combination of wind speed, wave conditions, fault scenarios, and other environmental parameters.

    * case_matrix_DLC*.txt/yaml
        * Individual case matrix files for specific DLCs (e.g., DLC1.6_0), containing only the cases relevant to that particular design load case.

For more information on how DLCs are configured and generated, please refer to the :doc:`Design Load Cases page <dlc_generator>`.


Optimization-related
--------------------

When optimization is enabled by setting ``recorder/flag = True`` in the ``analysis_options.yaml`` file, additional outputs are generated to track the optimization process:

    * log_opt.sql
        * An SQLite database file containing the complete history of the optimization process. This file can be visualized using OpenMDAO's built-in visualization tools to show the convergence history, constraint violations, and objective improvement over iterations. The filename is configured via the ``recorder/file_name`` setting.

    * iteration_*/
        * Directories containing results from each optimization iteration:

            * timeseries/
                * Contains raw OpenFAST time series outputs for all simulations run during the specific iteration. These files capture the detailed dynamic response of the system for each design point.

            * summary_stats.p
                * A Python pickle file containing statistical summaries (min, max, mean, standard deviation, median, absolute values, integrated values) of key output channels across all simulations in the iteration. This provides a quick overview of system performance without needing to process the raw time series.

            * DELs.p
                * Contains Damage Equivalent Loads calculated from the fatigue analysis of time series data. These loads represent the cumulative damage effect from variable amplitude loading converted to equivalent constant amplitude loads.

            * fst_vt.p
                * Contains the complete OpenFAST input parameter set for this iteration in a FAST variable tree format ref: `openfast_io <https://openfast.readthedocs.io/en/main/>`_.. This preserves the exact model configuration used for the simulations in this iteration.

RAFT-based Optimization
^^^^^^^^^^^^^^^^^^^^^^^

For optimizations using the RAFT (Ranging And Fatigue Tool) module instead of OpenFAST:

    * raft/raft_designs/
        * This directory contains RAFT time series outputs and design information, available in both pickle (.p) and YAML (.yaml) formats. These files provide computationally efficient approximations of turbine dynamics and loads for optimization purposes.
