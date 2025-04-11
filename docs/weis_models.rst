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
* **WISDEM-generated model**: Generate OpenFAST inputs from a geometry file and run simulations

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

WEIS implements a framework for managing Design Load Cases (DLCs) according to IEC standards through the ``DLCGenerator`` class (``weis/dlc_driver/dlc_generator.py``). This powerful framework is tightly integrated with OpenFAST through the ``FASTLoadCases`` component.

For a detailed description of all supported DLCs and their specific configuration options, please refer to the dedicated :doc:`dlc_generator` documentation.

The framework supports standard DLCs such as:

* Power production (DLC 1.x)
* Power production with fault (DLC 2.x) 
* Startup conditions (DLC 3.x)
* Normal shutdown (DLC 4.x)
* Emergency shutdown (DLC 5.x)
* Parked/idling (DLC 6.x)
* Parked with fault (DLC 7.x)

Post-Processing
^^^^^^^^^^^^^^

After simulations, WEIS processes OpenFAST outputs to provide:

* Blade loading distributions
* Tower/monopile load profiles
* Fatigue damage calculations using rainflow counting
* Damage equivalent loads (DELs) for fatigue analysis
* Peak loads for ultimate strength checks
* Control performance metrics
* Maximum values for design constraints

These results are formatted for integration with optimization workflows and automated reporting.

Further Information
^^^^^^^^^^^^^^^^^^

* For detailed OpenFAST documentation, visit the `OpenFAST documentation <https://openfast.readthedocs.io/>`_
* Examples of OpenFAST usage in WEIS can be found in the ``examples`` directory
* For advanced features like free-vortex wake modeling (OLAF), dedicated examples are provided
