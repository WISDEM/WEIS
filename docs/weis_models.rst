WEIS Models
==============

WISDEM 
-------


RAFT 
-------


OpenFAST
-----------

OpenFAST is a high-fidelity physics-based tool for wind turbine time-domain aeroelastic simulation. In WEIS, OpenFAST serves as the primary dynamic analysis engine for comprehensive turbine modeling and load calculation.

OpenFAST is integrated into WEIS through the ``FASTLoadCases`` class in ``weis/aeroelasticse/openmdao_openfast.py`` that connects the static design models from WISDEM with OpenFAST's dynamic analysis capabilities. This integration enables the inclusion of dynamic effects, controller interactions, and transient loading in the turbine design optimization process.

Within WEIS, OpenFAST receives structural, aerodynamic, and controller properties from WISDEM and translates them into full dynamic simulation models. This integration is achieved through multiple workflow paths:

- **Direct model creation**: WEIS can generate all OpenFAST input files from WISDEM outputs, automatically configuring the structural properties, aerodynamic surfaces, control settings, and environmental conditions
- **Existing model modification**: WEIS can modify existing OpenFAST models, adjusting specific parameters according to the design variables in the optimization process

The OpenFAST simulation process within WEIS is managed by the ``FASTLoadCases`` component, which handles:

1. Translation of design variables into OpenFAST model parameters
2. Setup of OpenFAST's modular structure (ElastoDyn, AeroDyn, ServoDyn, HydroDyn, etc.)
3. Parallel execution of multiple simulations across different load cases
4. Post-processing of simulation outputs to extract design-relevant quantities

For blade modeling, WEIS configures both simple (ElastoDyn) and advanced (BeamDyn) structural formulations, with automatic conversion of WISDEM-generated beam properties to the appropriate format. Aerodynamic modeling is handled through AeroDyn, with support for both Blade Element Momentum theory and higher-fidelity options like the free vortex wake model (OLAF). Offshore applications utilize integrated hydrodynamic and mooring modules (HydroDyn, SubDyn, and MoorDyn) that receive platform and mooring properties from WISDEM.

WEIS uses OpenFAST's simulation capabilities through automated load analysis, extracting statistics, damage equivalent loads (DELs), and extreme values across simulations. These outputs are directly incorporated into the optimization process, enabling design constraints based on ultimate loads, operational performance, or other user defined metrics. This comprehensive analysis capability allows designers to create turbines that not only maximize energy production and minimize cost, but also maintain structural integrity.

For floating offshore applications, OpenFAST's HydroDyn and MoorDyn modules simulate the complex interactions between the turbine, platform, and ocean environment. WEIS can evaluate platform motion statistics and stability metrics, enabling the design of integrated floating wind systems that balance turbine performance with platform stability.

In summary, OpenFAST in WEIS provides:

1. High-fidelity time-domain simulations capturing the full dynamic behavior of wind turbines
2. Modeling of various wind tubine configurations
3. Load calculation for design verification and optimization
5. Support for standard design load cases following IEC guidelines
6. Capabilities for controller design and evaluation
7. Performance metrics for optimization-based design

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
* Fatigue damage equivalent loads (DELs)
* Maximum deflections and stresses
* Control performance metrics (overspeed, pitch rates, etc.)
* Platform motions for floating systems

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
