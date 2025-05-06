WEIS Models
==============

WISDEM 
-------

WEIS wraps the systems engineering framework WISDEM. 
Both WISDEM and WEIS are built using `OpenMDAO <https://openmdao.org>`_ and therefore share much of the same architecture and syntax. 
OpenMDAO is a generic open-source optimization framework implemented in Python by NASA. 
WISDEM has been developed by NREL researchers using OpenMDAO since 2014. 
Both WISDEM and WEIS allow users to optimize the design of single turbines. 
The key difference between WISDEM and WEIS is that WISDEM implements steady-state models, such as the steady-state blade element momentum solver CCBlade, the cross-sectional solver PreComp, and the steady-state beam solver frame3dd, whereas WEIS includes solvers capable of capturing dynamic effects using OpenFAST and RAFT. 
WEIS also includes the turbine controller ROSCO.
WISDEM and WEIS share the same three input files: the geometry yaml, which describes the turbine, the modeling yaml, which lists all modeling-related options, and the analysis yaml, which builds the optimization setup and lists design variables, figure of merit, constraints, and optimization settings. 
The only difference in inputs files between WISDEM and WEIS is that the modeling options yaml in WEIS has many more optional inputs, as it contains all the flags and settings that configure OpenFAST, RAFT, and ROSCO.

Within WEIS, WISDEM models the turbine and generates all those quantities that are needed by OpenFAST and RAFT. 
For the turbine rotor, WISDEM interpolates aerodynamic quantities and builds the lifting line and associated polars. 
WISDEM runs PreComp and computes the equivalent elastic properties of the blades. 
These are then formatted for the structural dynamic solvers BeamDyn and ElastoDyn, including the mode shapes. 
WISDEM computes the equivalent elastic properties of the nacelle components, of the tower, and of the offshore substructure. 
WISDEM connects floating platform joints using members along with the parameters needed for the hydrodynamic solvers in OpenFAST and RAFT. 
WISDEM finally computes the regulation trajectory for the turbine, which is used by ROSCO, so that WEIS can automatically create the DISCON input file.
In simple terms, WISDEM can be seen as the first tool for the conceptual design of a wind turbine. 
Once a first conceptual design is generated, WEIS becomes a critical tool for more sophisticated studies investigating the dynamic performance of the turbine, whether land-based or offshore.

WEIS users are encouraged to familiarize themselves with WISDEM through its documentation, which is available at `https://wisdem.readthedocs.io/en/master/ <https://wisdem.readthedocs.io/en/master/>`_. 
The list of WISDEM examples is often a good starting point.


RAFT 
-------
`RAFT <https://github.com/WISDEM/RAFT>`_ (Response Amplitudes of Floating Turbines) is an open-source Python-based software for frequency-domain analysis of floating wind turbines. The software includes quasi-static mooring reactions, strip-theory and potential-flow hydrodynamics (first and second order), blade element momentum aerodynamics, and linear turbine control in a way that avoids any time-domain preprocessing. RAFT constitutes the "Level 1" of modeling fidelity in the WEIS toolset for floating wind turbine controls co-design, providing rapid evaluation (in the order of seconds) of the coupled responses of the system.

As a frequency-domain code, RAFT solves for the steady state response of the system assuming linearity. In the frequency domain, time-varying quantities (such as external loads and system responses) are modeled as a linear superposition of a mean and a Fourier series of complex amplitudes. Under these assumptions, RAFT can compute:

- Mean responses (mean displacements, mean mooring line tensions, mean rotor thrust, ...) by solving the static equilibrium of the system under a given environmental condition
- Dynamic responses by solving the linearized equations of motion of the system (around an equilibrium position corresponding to the mean response) under a given environmental condition. The dynamic response is given in the form of power spectral densities. Based on the power spectral density, RAFT computes the standard deviation and estimates extreme values of the system responses
- The eigenmodes and eigenfrequencies of the system around the equilibrium position
- Inertial and hydrostatic properties of the system

For a more detailed description of RAFT, see `https://openraft.readthedocs.io/ <https://openraft.readthedocs.io/>`_.

Using RAFT through WEIS
~~~~~~~~~~~~~~~~
A RAFT model requires information about the floating platform, turbine, mooring system, site, environmental conditions, and numerical settings. For a standalone RAFT model (i.e., to be run outside of WEIS), these information are usually provided by a `yaml` file, as illustrated by the `examples <https://github.com/WISDEM/RAFT/tree/master/examples>`_ available in the RAFT GitHub repo.

When used through WEIS, RAFT receives the necessary data directly from WEIS. The platform, turbine, and site information are specified in the `geometry file`, from which RAFT uses only a subset of the data. RAFT specific inputs are provided in the `RAFT` section of the `modeling file`, as described in :ref:`modeling-options`. The RAFT-specific part of the schema that describes the modeling options is given below:

.. literalinclude:: ../weis/inputs/modeling_schema.yaml
   :language: yaml
   :lines: 103-184

WEIS can output a RAFT `.yaml` input file by setting the flag `save_designs` to `True` in the `RAFT` section of the `modeling file`.


.. _section-BEM_modeling:
BEM modeling
~~~~~~~~~~~~

RAFT uses `pyHAMS <https://github.com/WISDEM/pyHAMS>`_, a potential-flow code based on the Boundary Element Method (BEM), to obtain first-order frequency-dependent potential flow coefficients for the platform. To use BEM modeling in WEIS/RAFT, there are two ways:

1. First, if you want to mesh and use BEM modeling for all members of the platform, you can just specify ``potential_model_override: 2``. This will enable BEM modeling for **all** members, overwriting the value of ``potential_bem_member`` specified to each member. Note that RAFT uses strip theory for hydrostatics irrespective of BEM modeling options.
2. Another approach is to specify the member names of the floating platform that are intended for BEM modeling in the *potential_bem_member* option in modeling options and use *potential_model_override: 0*. This allows using BEM modeling for the listed members; for the members not listed, strip theory will be used to provide wave excitation and added mass. Note that RAFT uses strip theory for hydrostatics irrespective of BEM modeling options.

This table below summmarizes the options for *potential_model_override* in RAFT. Note that hydrostatics (buoyancy) is always computed using strip theory, regardless of the potential_model_override option.

+--------------------------+-----------------------------------------------------+----------------------------------------------------------------------------------------+
| potential_model_override | BEM modeling                                        | Strip theory                                                                           | 
+==========================+=====================================================+========================================================================================+
| 0                        | Wave excitation and added mass for selected members | Wave excitation and added mass for selected members, drag and buoyancy for all members |
+--------------------------+-----------------------------------------------------+----------------------------------------------------------------------------------------+
| 1                        | No for all members                                  | All forces from strip theory for all members                                           |
+--------------------------+-----------------------------------------------------+----------------------------------------------------------------------------------------+
| 2                        | Wave excitation and added mass for all members      | Only drag and buoyancy from strip theory for all members                               | 
+--------------------------+-----------------------------------------------------+----------------------------------------------------------------------------------------+

WEIS now includes the capability to create intersected meshes for the BEM modeling. This capability relies on using external packages, `pygmsh <https://pygmsh.readthedocs.io/en/latest/>`_ and `meshmagick <https://github.com/LHEEA/meshmagick>`_. Please make sure you install these separately. 

pyHAMS
^^^^^^

pyHAMS is a python wrapper for HAMS, a boundary element method for hydrodynamic analysis of submerged structures. This is the current default BEM solver used in WEIS/RAFT to obtain potential flow solutions for analyses and optimizations. Check out the `pyHAMS GitHub page <https://github.com/WISDEM/pyHAMS>`_ for more details. If you set up the options correctly in WEIS/RAFT to request potential flow solutions, the BEM meshes will be generated internally and pyHAMS will be run.


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

For blade modeling, WEIS configures both simple (ElastoDyn) and advanced (BeamDyn) structural formulations, with automatic conversion of WISDEM-generated beam properties to the appropriate format. Aerodynamic modeling is handled through AeroDyn, with support for both Blade Element Momentum theory and higher-fidelity options like the free vortex wake model (OLAF). Offshore applications utilize integrated hydrodynamic and mooring modules (HydroDyn, SeaState, SubDyn, and MoorDyn) that receive platform and mooring properties from WISDEM.

WEIS uses pCrunch to post-process OpenFAST's simulation results, providing automated load analysis, extracting statistics, damage equivalent loads (DELs), and extreme values across simulations. These outputs are directly incorporated into the optimization process, enabling design constraints based on ultimate loads, operational performance, or other user defined metrics. This comprehensive analysis capability allows designers to create turbines that not only maximize energy production and minimize cost, but also maintain structural integrity.

For floating offshore applications, OpenFAST's HydroDyn and MoorDyn modules simulate the complex interactions between the turbine, platform, and ocean environment. WEIS can evaluate platform motion statistics and stability metrics, enabling the design of integrated floating wind systems that balance turbine performance with platform stability.

In summary, OpenFAST in WEIS provides:

1. Mid-fidelity time-domain simulations capturing the full dynamic behavior of wind turbines
2. Modeling of various wind turbine configurations
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
* Control performance metrics (over-speed, pitch rates, etc.)
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
^^^^^^^^^^^^^^^^^^^^

* For detailed OpenFAST documentation, visit the `OpenFAST documentation <https://openfast.readthedocs.io/>`_
* Examples of OpenFAST usage in WEIS can be found in the ``examples`` directory
* For advanced features like free-vortex wake modeling (OLAF), dedicated examples are provided





pCrunch
----------
WEIS uses pCrunch internally to process the time series data from OpenFAST to provide merit of figures and constraints in the optimization. pCrunch can perform post-processing for any time series data from wind turbine simulations. Readers are provided for OpenFAST outputs, but the analysis tools are equally applicable to HAWC2, Bladed, QBlade, ADAMS, or other tools. pCrunch attempts to capture the primary functions of legacy tools like MCrunch, MLife, and MExtremes, while using open source utilities.  Some statistics pCrunch provides are: extreme values, fatigue calculation based on rainflow cycle counting and miner’s law, and AEP calculation. pCrunch is usually installed as part of WEIS. For separate installation guide and more details on pCrunch, check out the `pCrunch github page 
<https://github.com/NREL/pCrunch/tree/master>`_. pCrunch provides examples using the two primary classes AeroelasticOutputs and Crunch, see 
`AeroelasticOutputs example <https://github.com/NREL/pCrunch/blob/master/examples/aeroelastic_output_example.ipynb>`_ and 
`Crunch example <https://github.com/NREL/pCrunch/blob/master/examples/crunch_example.ipynb>`_.


ROSCO
---------

WEIS contains the Reference Open Source Controller for wind turbines (`link to paper <https://wes.copernicus.org/articles/7/53/2022/>`_).
The ROSCO dynamic library links with OpenFAST, receiving sensor signals and providing control inputs to the turbine.
The ROSCO toolbox is used to automatically tune controllers using turbine information and a few input parameters.
Both the ROSCO dynamic library and python toolbox are installed via ``conda`` (or ``pip``) with WEIS.
Within WEIS, an OpenMDAO component wraps ROSCO and connects the turbine information from WISDEM (or an OpenFAST model) as inputs, and saves controller parameters that will be used in the ROSCO DISCON.IN input file.
The DISCON.IN file contains detailed parameters, including gains, set points, and limits specific to a turbine model.
Users can investigate the DISCON.IN file included in the WEIS-generated OpenFAST set when troubleshooting.
By setting the ``LoggingLevel > 0``, users can inspect the internal signals of ROSCO.


ROSCO controller
^^^^^^^^^^^^^^^^^^
The ROSCO control library uses signals like the

- generator speed
- nacelle tower top acceleration
- blade root moments
- nacelle wind vane and anemometer

to prescribe control actions for the turbine: primarily blade pitch and generator torque.

The generator torque controller is used to maintain the optimal tip speed ratio (TSR) below rated.
The optimal TSR is determined using the Cp surface from CC-Blade in WISDEM.
A wind speed estimate used to determine the optimal rotor speed. 
It is recommended to use ``VS_ControlMode: 2`` to achieve this type of torque control.
ROSCO can also set constant torque or constant power control above rated, using the ``VS_ConstPower`` input.

The pitch controller in ROSCO is a gain-schedule proportional-integral controller.
If a WISDEM turbine is used to tune the controller, the rotor speed set point is determined by the geometry inputs ``VS_maxspd`` and ``maxTS``, whichever results in a lower rotor speed.
The pitch control gains are automatically determined using the cc-blade-generated, Cp surfaces, and the natural frequency (``omega_pc``) and damping ratio (``zeta_pc``) of the desired closed-loop generator speed response.

ROSCO implements peak shaving using a look-up table from the wind speed estimate to the minimum pitch limit, which is determined using the Ct coefficients (generated in WISDEM's cc-blade).
ROSCO can also apply floating feedback control, where a nacelle acceleration signal is used to generate a pitch offset for damping platform motion.

Yaw control, control of structural elements (like TMDs or external forces), and mooring cable control are available in ROSCO and OpenFAST, but are not yet supported in WEIS.  
Soon, start-up and shutdown control will be available in both ROSCO and WEIS; they will be enabled by the DLC driver.
Shutdown control will include safety checks for high wind speed, generator overspeed, and large yaw misalignments.

