WEIS Models
==============

WISDEM 
-------


RAFT 
-------


OpenFAST
-----------

OpenFAST is an open-source wind turbine simulation tool developed and maintained by the National Renewable Energy Laboratory (NREL). Within WEIS, OpenFAST serves as the high-fidelity physics-based solver for wind turbine aeroelastic simulation.

Key Functions in WEIS
^^^^^^^^^^^^^^^^^^^^^

OpenFAST in WEIS provides:

* Time-domain aeroelastic simulations of wind turbines.
* Detailed loading analysis under various operating conditions
* Fatigue and extreme load calculations
* Linearized turbine models for controls design and optimization
* Support for Design Load Cases (DLCs) defined by IEC standards
* Integration with the ROSCO controller for turbine control
* Capability to model advanced features like active trailing edge flaps and tuned mass dampers (TMDs)

Integration and Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary integration point for OpenFAST in WEIS is through the ``FASTLoadCases`` class within the ``openmdao_openfast.py`` file. This class serves as an OpenMDAO ExplicitComponent that:

1. Sets up the OpenFAST simulation environment
2. Configures input files and directories
3. Executes the simulations (serially or in parallel)
4. Post-processes the results, extracting key performance metrics and loads data

WEIS can run OpenFAST in different modes:

* **Model generation only**: Create OpenFAST input files without executing the solver
* **From existing model**: Run OpenFAST using an existing set of input files
* **WISDEM-generated model**: Generate OpenFAST inputs from a WISDEM model and run simulations
* **Linearization**: Extract linearized representations of the turbine model
* **Parallel execution**: Distribute simulations across multiple cores for computational efficiency

Input/Output Structure
^^^^^^^^^^^^^^^^^^^^^^

**Inputs**:

The OpenFAST component in WEIS accepts a wide range of inputs including:

* Environmental conditions (wind speeds, turbulence parameters, water depths)
* Turbine geometric properties (blade, tower, platform definitions)
* Operational parameters (controller settings, rotor speeds)
* Simulation settings (time steps, analysis duration, output channels)
* Design Load Case specifications

**Outputs**:

Key outputs from OpenFAST simulations include:

* Power production metrics (AEP, Cp, Ct curves)
* Blade, tower, and platform loads
* Damage equivalent loads (DELs)
* Maximum deflections and stresses
* Control performance metrics (overspeed, pitch rates, etc.)
* Platform motions for floating systems

Example Configuration
^^^^^^^^^^^^^^^^^^^^

OpenFAST settings are defined in the modeling options YAML file, for example:

.. code-block:: yaml

    OpenFAST:
      flag: True
      simulation:
        DT: 0.01
        CompElast: 1
        CompInflow: 1
        CompAero: 2
        CompServo: 1
        CompHydro: 1
        CompSub: 0
        CompMooring: 3
        
      # Module-specific configurations
      AeroDyn:
        WakeMod: 1  # Set to 3 for OLAF (free vortex wake)
      
      ElastoDyn:
        # Flags for DOFs
        FlapDOF1: True
        FlapDOF2: True
        EdgeDOF: True
        
      # ... Additional module settings

Design Load Cases
^^^^^^^^^^^^^^^^^

WEIS implements a system for managing Design Load Cases (DLCs) according to IEC standards through the ``DLCGenerator`` class (``weis/dlc_driver/dlc_generator.py``). This framework automatically configures appropriate wind and wave conditions based on the specified DLC numbers and is integrated with OpenFAST through the ``FASTLoadCases`` component.

The DLC framework currently includes:

**Power Production DLCs (1.x)**

* **DLC 1.1**: Normal turbulence model (NTM) with normal sea state for ultimate loads
* **DLC 1.2**: NTM with normal sea state for fatigue analysis
* **DLC 1.3**: Extreme turbulence model (ETM) for ultimate loads
* **DLC 1.4**: Extreme coherent gust with direction change (ECD)
* **DLC 1.5**: Extreme wind shear (EWS) with horizontal/vertical profiles
* **DLC 1.6**: NTM with severe sea state

**Fault Conditions DLCs (2.x)**

* **DLC 2.1**: Power production with grid loss
* **DLC 2.2**: Power production with pitch system fault
* **DLC 2.3**: Power production with control system fault
* **DLC 2.4**: Fatigue load cases with faults

**Startup DLCs (3.x)**

* **DLC 3.1**: Turbine startup during normal wind/sea conditions

**Shutdown DLCs**

* **DLC 4.1**: Normal shutdown
* **DLC 5.1**: Emergency shutdown

**Parked/Standstill DLCs (6.x)**

* **DLC 6.1**: Parked with extreme wind model (50-year return)
* **DLC 6.2**: Parked with grid loss and extreme yaw misalignment
* **DLC 6.3**: Parked with extreme wind model (1-year return)
* **DLC 6.4**: Parked with NTM for fatigue analysis

**Parked with Fault DLCs (7.x)**

* **DLC 7.1**: Parked with fault in extreme conditions (1-year return)
* **DLC 7.2**: Parked with fault in normal conditions for fatigue

**Special Case DLCs**

* **AEP**: Special DLC for Annual Energy Production calculation
* **freedecay**: Platform free decay tests for floating systems

DLC configuration is handled in the modeling options YAML file:

.. code-block:: yaml

    DLC_driver:
        DLCs:
            - DLC: "1.1"
              ws_bin_size: 2
              wind_speed: [4,6,8,10,12,14,16,18,20,22,24] 
              wave_height: [4.0]
              wave_period: [10.0]
              n_seeds: 6
              analysis_time: 600.
              transient_time: 120.
              turbulent_wind:
                  HubHt: 150.0
                  RefHt: 150.0
                  PLExp: 0.11
            - DLC: "6.1"
              yaw_misalign: [-8, 8]
              wind_speed: [50.0]  # V_e50 (Class I)
              sea_state: "50-year"
              analysis_time: 600.
              transient_time: 120.

For each DLC, the framework:

1. Creates appropriate environmental conditions (wind/waves)
2. Sets up proper initial conditions and control states
3. Configures precise fault settings when applicable
4. Manages appropriate simulation length for each analysis type
5. Handles multi-seed simulations for statistical validity

The DLC generator applies partial safety factors automatically based on the specific DLC, and handles metocean conditions according to the specified turbine class and site characteristics.

Post-Processing
^^^^^^^^^^^^^^

After simulations, WEIS processes OpenFAST outputs to provide:

* Blade loading distributions
* Tower/monopile load profiles
* Fatigue damage calculations using rainflow counting
* Damage equivalent loads (DELs)
* Peak loads for ultimate strength checks
* Control performance metrics
* Maximum values for design constraints

These results are formatted for integration with optimization workflows and automated reporting.

Further Information
^^^^^^^^^^^^^^^^^^

* For detailed OpenFAST documentation, visit the `OpenFAST documentation <https://openfast.readthedocs.io/>`_
* Examples of OpenFAST usage in WEIS can be found in the ``examples`` directory
* For advanced features like free-vortex wake modeling (OLAF), dedicated examples are provided
