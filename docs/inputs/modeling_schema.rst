.. _modeling-options:

Modeling Options Inputs
==========================

Overview
-------------

The WEIS modeling options specify inputs for the various simulation models, the ROSCO controller, and other modules of WEIS that are not directly related to turbine components, which are defined in the :ref:`section-geometry_inputs`.
The modeling options includes

- Simulation settings
- Controller parameters
- Load case parameters

Detailed descriptions of the modeling options can be found below.
In the following, we will provide an overview of each modeling input section and some tips for setting up simulations using WEIS.

General
^^^^^^^^^^^^^^^
The ``General`` section of the modeling options primarily contains information related to the OpenFAST ``simulation_configuration``, including the

- Output file name base (``OF_run_fst``)
- Output folder location (``OF_run_dir``).  If not provided, WEIS will generate a folder called ``openfast_runs`` in the ``folder_output`` set in the analysis options. 
- A flag to use the OpenFAST executable (``use_exe: True``) or the dynamic library.
- A flag to allow failed OpenFAST runs during a WEIS run (``allow_fails: True``), and continue the current workflow.

Some post-processing settings are also contained in the ``General`` section, like whether to use a ``goodman_correction`` during fatigue calculations.


WISDEM
^^^^^^^^^^^^^^^
The ``WISDEM`` section of the modeling options in WEIS is inhereted from the modeling options of WISDEM.
In this section, users can toggle the ``flag`` value for the various SE (system engineering) components of WISDEM, like RotorSE.
Users can also specify elastic properties for inidividual components by toggle the ``user_elastic`` option. This way, WISDEM will pass through the elastic properties from the geomtry input, rather than generate them from component models.
Users should note:

- The spar cap and trailing edge pressure and suction layers (``spar_cap_ss``, ``spar_cap_ps``, ``te_ss``, and ``te_ps``) must be defined and correspond to blade layers in the geometry option.
- In the ``RotorSE`` section, users can determine the number of sampling points in the Cp surface through the ``n_pitch_perf_surfaces`` and ``n_tsr_perf_surfaces`` inputs.  The performance surfaces, generated using cc-blade, are used to automatically tune the ROSCO controller and provide aerodynamci modeling to RAFT.  If too few (less than 20) points are used in each dimension, the gains and limits of the ROSCO controller can be adversely affected.
- By default, ``symmetric_moorings`` in the ``FloatingSE`` section of WISDEM is ``True``.  In many cases, including lines with multiple segments, this should be set to ``False``.


OpenFAST
^^^^^^^^^^^^^^^
The ``OpenFAST`` section of the modeling options should reflect the OpenFAST iputs in a yaml format.
The ``simulation`` section corresponds to the ``.fst`` input of OpenFAST, and all the modules of OpenFAST (e.g., ElastoDyn) should have a corresponding subsection in the OpenFAST schema.

In this section, users can supply their own OpenFAST model using the ``from_openfast: True`` flag.
For example, the input::

   OpenFAST: 
      flag: True
      from_openfast: True
      regulation_trajectory: IEA-15-240-RWT-outputs.yaml
      openfast_file: IEA-15-240-RWT-UMaineSemi.fst
      openfast_dir: ../01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi

Will set up WEIS to run the OpenFAST model specified by the ``openfast_file`` in the ``openfast_dir``.
File paths in WEIS are relative to the file from which they originate. 

The ``regulation_trajectory`` is an optional input that can be used to specify initial conditions to simulations.

RAFT
^^^^^^^^^^^^^^^
The ``RAFT`` section of the modeling options is primarily used for RAFT simulation settings and also potential flow modeling using pyHAMS.
The ``potential_model_override`` input is set here and used in both RAFT and OpenFAST.
More details about potential flow modeling and hybrid models can be found here :ref:`bem-modeling`.
Selecting which members will be used in a potential flow solution is also specified in the ``RAFT`` modeling options.
For example::

   RAFT:
    flag: True
    min_freq: 0.1
    max_freq: 2.0
    potential_bem_members: [spar]

Will model the ``spar`` member in pyHAMS and use it to model the hydrodynamics using potential flow in both RAFT and OpenFAST.

WEIS now includes the capability to create more complicated intersected meshes for the BEM modeling. Users can set the ``intersection_mesh`` value to 1 if wanting to use this capability. Note this capabilty relies on using external packages, `pygmsh <https://pygmsh.readthedocs.io/en/latest/>`_ and `meshmagick <https://github.com/LHEEA/meshmagick>`_, and are not part of WEIS installation. Please make sure you install these separately beforehand. If users want to plot the intermediate meshes, toggle the ``plot_designs`` and ``save_designs`` options to True while using ``intersection_mesh``. This mesh plotting relies on using `trimesh <https://trimesh.org/>`_.


ROSCO 
^^^^^^^^^^^^^^^
The ``ROSCO`` modeling options in WEIS inheret the ``controller_parameters`` section of the `ROSCO toolbox schema <https://rosco.readthedocs.io/en/latest/source/toolbox_input.html#controller-params>`_.
Users can also specify a ``tuning_yaml`` to set the controller parameters of ROSCO.
Any options specified in the WEIS modeling options will override the ``tuning_yaml`` options.
If a user provides their own OpenFAST turbine model, the ``tuning_yaml`` is required and the ``turbine_parameters`` section will be used to generate a DISCON.IN input.
Otherwise, the ``turbine_parameters`` are determined from WISDEM and only the ``controller_parameters`` are used.

DLC Driver
^^^^^^^^^^^^^^^
Finally, the ``DLC_Driver`` section of the modeling options is where users specify metocean conditions and the list of design load case (DLCs) to be simulated.
A user can specify multiple cases of the same DLC.
A more detailed documentation of the ``DLC_Driver`` can be found here: :ref:`section-dlc_driver`.
The full set of DLCs that can be simulated can be seen in `the modeling options of this unit test <https://github.com/WISDEM/WEIS/blob/86a5df1b8792a3bf036642d1b2bd3557ace7f555/weis/dlc_driver/test/weis_inputs/modeling_options_all_dlcs.yaml#L66>`_.


Detailed modeling option descriptions
---------------------------------------

The following is automatically generated from the weis modeling schema:

.. raw:: html
   :file: ../_static/modeling_doc.html
