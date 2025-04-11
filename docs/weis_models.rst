WEIS Models
==============

WISDEM 
-------

WEIS wraps the systems engineering framework WISDEM. 
Both WISDEM and WEIS are built on top of `OpenMDAO <https://openmdao.org>_` and therefore share much of the same architecture and syntax. 
OpenMDAO is a generic open-source optimization framework implemented in Python by NASA. 
WISDEM has been developed by NREL researchers on top of OpenMDAO since 2014. 
Both WISDEM and WEIS allow users to optimize the design of single turbines. 
The key difference between WISDEM and WEIS is that WISDEM implements steady-state models, such as the steady-state blade element momentum solver CCBlade, the cross-sectional solver PreComp, and the steady-state beam solver frame3dd, whereas WEIS includes solvers capable of capturing dynamic effects using OpenFAST and RAFT. 
WEIS also includes the turbine controller ROSCO.
WISDEM and WEIS share the same three input files: the geometry yaml, which describes the turbine, the modeling yaml, which lists all modeling-related options, and the analysis yaml, which builds the optimization setup and lists design variables, figure of merit, constraints, and optimization settings. 
The only difference in inputs files between WISDEM and WEIS is that the modeling options yaml in WEIS has many more optional inputs, as it contains all the flags and settings that go into OpenFAST, RAFT, and ROSCO.

Within WEIS, WISDEM models the turbine and generates all those quantities that are needed by OpenFAST and RAFT. 
For the turbine rotor, WISDEM interpolates aerodynamic quantities and builds the lifting line and associated polars. 
WISDEM runs PreComp and computes the equivalent elastic properties of the blades. 
These are then formatted for the structural dynamic solvers BeamDyn and ElastoDyn, including the mode shapes. 
WISDEM computes the equivalent elastic properties of the nacelle components, of the tower, and of the offshore substructure. 
WISDEM connects floating platform joints using members along with the parameters needed for the hydrodynamic solvers in OpenFAST and RAFT. 
WISDEM finally computes the regulation trajectory for the turbine, which is used by ROSCO, so that WEIS can automatically create the DISCON input file.
In simple terms, WISDEM can be seen as the first tool for the conceptual design of a wind turbine. 
Once a first conceptual design is generated, WEIS becomes a critical tool for more sophisticated studies investigating the dynamic performance of the turbine, whether land-based or offshore.

WEIS users are encouraged to familiarize themselves with WISDEM through its documentation, which is available at `https://wisdem.readthedocs.io/en/master/ <https://wisdem.readthedocs.io/en/master/>_`. 
The list of WISDEM examples is often a good starting point.


RAFT 
-------


OpenFAST
-----------

OpenFAST is a high-fidelity physics-based tool for wind turbine aeroelastic simulation. In WEIS, OpenFAST serves as the primary time-domain dynamic analysis engine for comprehensive turbine modeling and load calculation.

OpenFAST is integrated into WEIS through the ``FASTLoadCases`` class in ``weis/aeroelasticse/openmdao_openfast.py`` that connects the static design models from WISDEM with OpenFAST's dynamic analysis capabilities. This integration enables the inclusion of dynamic effects, controller interactions, and transient loading in the turbine design optimization process.

Within WEIS, OpenFAST receives structural, aerodynamic, and controller properties from WISDEM and translates them into full dynamic simulation models. This integration is achieved through multiple workflow paths:

- **Direct model creation**: WEIS can generate all OpenFAST input files from WISDEM outputs, automatically configuring the structural properties, aerodynamic surfaces, control settings, and environmental conditions
- **Existing model modification**: WEIS can modify existing OpenFAST models, adjusting specific parameters according to the design variables in the optimization process

The OpenFAST simulation process within WEIS is managed by the ``FASTLoadCases`` component, which handles:

1. Translation of design variables into OpenFAST model parameters
2. Setup of OpenFAST's modular input structure (ElastoDyn, AeroDyn, ServoDyn, HydroDyn, etc.)
3. Parallel execution of multiple simulations across different load cases
4. Post-processing of simulation outputs to extract design-relevant quantities

For blade modeling, WEIS configures both simple (ElastoDyn) and advanced (BeamDyn) structural formulations, with automatic conversion of WISDEM-generated beam properties to the appropriate format. Aerodynamic modeling is handled through AeroDyn, with support for both Blade Element Momentum theory and higher-fidelity options like the free vortex wake model (OLAF). Offshore applications utilize integrated hydrodynamic and mooring modules (HydroDyn, SubDyn, and MoorDyn) that receive platform and mooring properties from WISDEM.

WEIS uses pCrunch to post-process OpenFAST's simulation results, providing automated load analysis, extracting statistics, damage equivalent loads (DELs), and extreme values across simulations. These outputs are directly incorporated into the optimization process, enabling design constraints based on ultimate loads, operational performance, or other user defined metrics. This comprehensive analysis capability allows designers to create turbines that not only maximize energy production and minimize cost, but also maintain structural integrity.

For floating offshore applications, OpenFAST's HydroDyn and MoorDyn modules simulate the complex interactions between the turbine, platform, and ocean environment. WEIS can evaluate platform motion statistics and stability metrics, enabling the design of integrated floating wind systems that balance turbine performance with platform stability.

In summary, OpenFAST in WEIS provides:

1. Mid-fidelity time-domain simulations capturing the full dynamic behavior of wind turbines
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

After simulations, WEIS processes OpenFAST outputs using pCrunch to provide:

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
