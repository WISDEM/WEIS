.. _section-weis_examples:

Capabilities and Examples
=========================

Overview of Capabilities
-------------------------

1. Simulate an OpenFAST model over various cases or IEC Design Load Cases (DLCs)
2. Generate OpenFAST models using WISDEM
3. Perform design optimization of generated OpenFAST Models
4. Perform design optimizations of generated RAFT frequency domain models
5. Optimize the ROSCO controller parameters
6. Perform parametric studies using the Design of Experiments (DOE) driver
7. Perform post-processing of WEIS outputs using Jupyter notebooks
8. Run OpenFAST and RAFT models using with potential flow models generated in pyHAMS

There are also a collection of legacy features that were previously used in WEIS but are no longer supported. 
These features are not recommended for use in production or research applications. 
If you choose to use them, you do so at your own risk and should be aware that they may not work as intended or may be removed in future releases; they may require additional development, dependencies, or configurations that are not documented, and they may not be compatible with the latest versions of WEIS or its dependencies.
These features are not documented in this section, but they are available in the examples directory.

1. Simulate Own OpenFAST Model
-------------------------------


To simulate a customized OpenFAST model, users must already have their own OpenFAST model. 
There are two primary approaches to automatically running simulations in WEIS:

Running a Set of Custom Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method allows users to define and vary OpenFAST inputs systematically:

- **Input Specification**:
  
  - The simulation iterates through ``case_inputs``, exploring different combinations, where a Cartesian product of defined ``groups`` generates a ``case_matrix`` of input changes.
  - The ``openfast_io`` module is used to read and parse the input files.
  - An OpenFAST dictionary "variable tree" (``fst_vt``) is created to manage and organize input variables.
    
- **Example Usage**:
  
  - The script ``run_openfast_cases.py`` simulates a set of steady-state cases.
  - Users define the initial conditions (corresponding to each wind speed) and wind speed in the same ``group``.

Running Design Load Cases (DLCs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach uses the WEIS driver along with a full set of inputs:

- **Analysis**:
  
  - In this case, analysis options are used to determine the location of the output folder.

- **Modeling Inputs**:
  
  - **OpenFAST Model**: Defines the location of the pre-existing turbine model.
  - **ROSCO Controller Input**: Specifies the control configuration.  See the ``ROSCO`` section and the ``tuning_yaml`` for details.
  - **Design Load Cases (DLCs)**: Configures the load cases. Users should see the :ref:`section-dlc_driver` for detailed DLC set up instructions.

- **Geometry Inputs**:
  
  - In these examples, where the turbine geometry is not used to generate the OpenFAST input, only the ``turbine_class``, ``hub_heigh``, and ``rotor_diameter`` are required.  The rest of the geometry is defined in the OpenFAST model.


2. Generate an OpenFAST Model from a WISDEM Component Model
-------------------------------------------------------------


The WEIS framework enables automatic generation of OpenFAST models from WISDEM component models:

- **Pre-processing**:
  
  - WISDEM pre-processes component model geometry part of the way toward an OpenFAST model.
  
- **Conversion**:
  
  - The WEIS function ``openmdao_openfast`` completes the conversion, producing an OpenFAST model represented as a ``fst_vt`` variable tree.

- **Simulation**:
  
  - The DLC generator creates the cases that are simulated, defined by the modeling options.

Available Starting Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several reference models are provided and can be used as starting points for new studies:

- **IEA-15MW RWT** with monopile and semisubmersible substructures
- **IEA-3.4MW RST** onshore turbine
- **OC3 Spar** with the NREL 5MW RWT
- **BAR-USC Turbine** modeled using OLAF (OpenFAST AeroDyn extension)

These geometry models are available in the ``examples/00_setup`` directory.

Analysis Options
^^^^^^^^^^^^^^^^^^

- Only the **output directory** is specified here.
- For these examples, optimization routines or parameter sweeps are **not** run by default.

Modeling Options
^^^^^^^^^^^^^^^^^^

- Used to define various flags within WISDEM.
- Allows overriding specific parameters in OpenFAST:
  
  - **Overrides are allowed**, except for those parameters directly related to DLCs.
  
- Defines inputs for the **ROSCO controller** and **DLC configurations**.

Geometry
^^^^^^^^^^^^^

- Geometry inputs are critical to determine the final structure and layout of the generated OpenFAST model.


3. Design with OpenFAST
---------------------------------

These examples demonstrate how to perform design optimizations using the OpenFAST model:

- **Turbine geometry** is altered by treating geometry parameters as design variables.
- **Automatically generated OpenFAST models** are used in each iteration.
- **Geometry and simulation outputs** are incorporated as constraints within the design optimization, as defined in the analysis options.


IEA-22MW RWT Semi-Submersible Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design Variables
^^^^^^^^^^^^^^^^

- Draft (lowest position of the platform)
- Column spacing (from the center of the center column to the center of the outer columns)
- Outer column diameter

Constraints
^^^^^^^^^^^

- ``draft_margin`` and ``heel_margin``: Ensures that the bottom of structural members does not leave the water and that the tops of members do not become submerged when the platform is tilted by the ``survival_heel``.
- ``*_ballast_capacity``: Ensures that chambers have enough volume to store both fixed and variable (water) ballast.
- ``Max_PtfmPitch``, ``Std_PtfmPitch``, and ``nacelle_acceleration`` are dynamic constraints derived from simulated OpenFAST outputs.

Merit Figure
^^^^^^^^^^^^

- **Structural mass**:
  
  - The merit figure for optimization is the structural mass of the platform, **excluding** the water ballast.

Optimization Method
^^^^^^^^^^^^^^^^^^^

- The optimization driver **``LN_COBYLA``** is used to iterate on the design variables and satisfy all constraints while minimizing structural mass.
- A comparison of solvers can be found on the **Optimization** page (see :ref:`section-optimization`).

Tower Design of the IEA-15MW RWT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design Variables
^^^^^^^^^^^^^^^^

- Tower diameter
- Tower thickness

Constraints
^^^^^^^^^^^

- Stress and buckling limits based on maximum loading scenarios
- Diameter-to-thickness ratio
- Frequency bounds ensure sufficient dynamic performance

Merit Figure
^^^^^^^^^^^^

- **Minimum tower mass**

4. Design in the Frequency Domain
---------------------------------


In these examples, design optimizations are performed using the lower-fidelity **RAFT** model.  
RAFT runs significantly faster than OpenFAST, enabling quicker optimization cycles while still capturing essential platform dynamics.


IEA-22MW RWT Semisubmersible Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The semisubmersible platform for the IEA-22MW RWT is optimized similarly to the OpenFAST-based case.
- Constraints used during optimization are generated from **RAFT** simulations instead of OpenFAST outputs.

IEA-15MW RWT Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Similar optimization techniques are applied to the IEA-15MW RWT semisubmersible.
- Geometry and performance constraints are again derived from **RAFT** results.

OC3 Spar Optimization
~~~~~~~~~~~~~~~~~~~~~~

- The **OC3 spar** floating platform undergoes optimization focused on the **fixed ballast volume**.
- During this process, the **Platform mass** includes the **water ballast** contribution, differing from some OpenFAST-based optimizations where water ballast was excluded from the structural mass.


5. Controller Optimization
-----------------------------

In these cases, the **turbine model is fixed** while optimization is focused on the **ROSCO controller parameters**.  
This allows tuning of the control systems to improve turbine performance without changing structural or aerodynamic designs.


ROSCO Controller Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design Variables
^^^^^^^^^^^^^^^^

Parameters of the ROSCO pitch controller:

  - Natural frequency and damping ratio of the speed regulator
  - Gain and filter frequency of floating feedback control for platform damping
  
Since these parameters are coupled, optimizing them together leads to better overall controller performance.

Constraint
^^^^^^^^^^

- **Maximum generator speed** limit to ensure safe turbine operation.

Merit Figure
^^^^^^^^^^^^

- **Tower base damage equivalent loads** (``DEL_TwrBsMyt``) are minimized to improve structural longevity; it is also a good proxy for the pitch actuation.

Model
^^^^^

- The optimization is performed on the **IEA-15MW RWT** with the VolturnUS-S semisubmersible, using the direct OpenFAST model (``from_openfast: True``, not generated from WISDEM).

Tuned Mass Damper (TMD) Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design Variables
^^^^^^^^^^^^^^^^

- **Substructure TMD parameters** are optimized via OpenFAST simulations:
  - Natural frequency
  - Damping ratio

Merit Figure
^^^^^^^^^^^^

- **Standard deviation of platform pitch** (``Std_PtfmPitch``) is minimized to measure platform stability.

Notes
^^^^^

- These optimizations are based on **OpenFAST simulations** for the **DLC 6.1** parked load case.
- TMD settings are provided through **StC** (Substructure Control) inputs in OpenFAST.


6. Parametric Analysis
---------------------------

In addition to optimization, **parametric analyses** can be performed by varying design variables using a **design of experiments (DOE)** approach in openmdao.  
This allows exploration of the design space without iterative optimization.

Design variables can be sampled:
- Randomly, using uniform or other probability distributions
- Using structured sampling techniques such as:
  - Full factorial design
  - Latin hypercube sampling

Constraints and merit figures in these cases are used only to add the associated OpenMDAO variables to the log (SQL) file for later postprocessing, not to drive optimization.


OpenFAST-Based Parametric Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Blade chord is varied according to a uniform random distribution.
- OpenFAST simulations are run for each sampled blade geometry.

RAFT-Based Parametric Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Semi-submersible platform parameters are varied, similar to the optimizations described previously.
- The RAFT model allows for faster evaluation across the design space.

A postprocessing script is provided to parse the SQL files generated during the parametric runs.  
This enables easy analysis of trends, constraint violations, and merit figure performance across the design space.


7. Postprocessing Notebooks
-----------------------------

A set of **Jupyter notebooks** is provided for easy **postprocessing** and **review** of simulation and optimization results.  
These tools streamline analysis and visualization of outputs generated during OpenFAST, RAFT, and WEIS optimization runs.
The WEIS Visualization tool is also available for working with WEIS outputs interactively (see :ref:`weis_viz_app`).

The postprocessing notebooks can be used to:

- **Plot OpenFAST outputs**
  - Visualize key time series and performance metrics from OpenFAST simulations.
  
- **Review summary outputs from DLC simulations**
  - Plot aggregate results across different Design Load Cases (DLCs).
  
- **Review optimization outputs from log (SQL) files**
  - Parse and plot optimization histories and trends.
  - Analyze constraint violations and merit figure progress.

- **Review WEIS output CSV files**
  - Access summarized simulation and optimization data stored in CSV format.
  - Useful for quick inspection and further custom analysis.

A more detailed description of the WEIS outputs can be found here: :ref:`section-weis_outputs`.


8. Potential Flow Modeling
-----------------------------

These examples demonstrate the use of **pyHAMS** within the **RAFT** framework to generate a **potential flow model** for hybrid hydrodynamic modeling.
The potential flow model can also be used in OpenFAST.
More information about potential flow modeling in WEIS can be found here: :ref:`section-BEM_modeling`.
