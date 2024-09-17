Design Load Cases in WEIS
-------------------------

Design load cases (DLCs) specify the conditions that a turbine must operate in safely thorughout its lifetime.
These load cases are defined in IEC standards.
We supplement the standards with information from the DTU design load basis (cite).

How to Set Up DLCs in WEIS
--------------------------

DLCs are configured in the modeling options (link). 
A full set of input information is contained within the modeling schema.
An example (a subset of a modeling input) is shown next::

  DLC_driver:
    metocean_conditions:
        wind_speed: [1.,3.,5.,7.,9.,11.,13.,15.,17.,19.,21.,23.,25.,27.,29.]
        wave_height_NSS: [0.84,0.84,0.87,0.99,1.15,1.34,1.58,1.82,2.08,2.34,2.66,2.98,3.28,3.77,3.94]
        wave_period_NSS: [8.3,8.3,8.3,7.7,7.1,6.3,6.1,6.2,6.2,6.7,7.1,7.1,7.7,7.7,7.7]
        wave_height_fatigue: [0.84,0.84,0.87,0.99,1.15,1.34,1.58,1.82,2.08,2.34,2.66,2.98,3.28,3.77,3.94]
        wave_period_fatigue: [8.3,8.3,8.3,7.7,7.1,6.3,6.1,6.2,6.2,6.7,7.1,7.1,7.7,7.7,7.7]
        wave_height_SSS: [9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7,9.7]
        wave_period_SSS: [13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6,13.6]
        wave_height1: 5.9
        wave_period1: 11.2
        wave_height50: 9.7
        wave_period50: 13.6
    DLCs: # Currently supported IEC 1.1, 1.3, 1.4, 1.5, 5.1, 6.1, 6.3, or define a Custom one
        - DLC: "1.1"
          ws_bin_size: 5
          n_seeds: 2
          #  analysis_time: 5.
          #  transient_time: 5.
          #  wind_speed: [3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.]
          turbulent_wind:
              HubHt: 110
              GridHeight: 160
              GridWidth: 160

The ``metocean_conditions`` are defined using tables of ``wind_speed``, ``wave_height``, and ``wave_period`` for normal sea states (``NSS``) and severe sea states (``SSS``), and sea states representative of 1- and 50-year return periods.
Individual DLCs use these conditions to determine specific sea conditions for each case, but they can also be overwritten in each case.

Users can specify the wind speed bin size (``ws_bin_size``) or the specific wind speeds (``wind_speed``).
The number of seeds (``n_seed``) and specifics about the turbsim inputs (link?, ``turbulent_wind``) can also be specified.
``transient_time`` is excluded from timeseries analysis; only ``analysis_time`` is used.


.. How WEIS works
.. --------------

.. WEIS is a stack of software that can be used for floating wind turbine design and analysis.  The turbine’s components (blade, tower, platform, mooring, and more) are parameterized in a geometry input. The system engineering tool `WISDEM <https://github.com/WISDEM/WISDEM>`_ determines component masses, costs, and engineering parameters that can be used to determine modeling inputs.  A variety of pre-processing tools are used to convert these system engineering models into simulation models.  

.. The modeling_options.yaml determines how the turbine is simulated.  We currently support `OpenFAST <https://github.com/OpenFAST/openfast>`_ and `RAFT <https://github.com/WISDEM/RAFT>`_.  

.. The analysis_options.yaml determine how to run simulations, either a single run for analysis, a sweep or design of experiments, or an optimization.  

.. A full description of the inputs can be found `here <https://github.com/WISDEM/WEIS/tree/master/weis/inputs>`_ (will point to rst when available).

.. .. image:: images/WEIS_Stack.png
..   :width: 500
..   :alt: Stack of software tools contained in WEIS

.. WEIS is `"glued" <https://github.com/WISDEM/WEIS/blob/master/weis/glue_code/glue_code.py>`_ together using `openmdao <https://openmdao.org/>`_ components and groups.

.. WEIS works best by running `examples <https://github.com/WISDEM/WEIS/tree/master/examples>`_:
..  * `01_aeroelasticse <https://github.com/WISDEM/WEIS/tree/master/examples/01_aeroelasticse>`_ can be used to set up batches of aeroelastic simulations in OpenFAST
..  * `02_control_opt <https://github.com/WISDEM/WEIS/tree/master/examples/02_control_opt>`_ can be used to run WEIS from an existing OpenFAST model, and optimize control parameters only
..  * `03_NREL5MW_OC3 <https://github.com/WISDEM/WEIS/tree/master/examples/03_NREL5MW_OC3_spar>`_ and `04_NREL5MW_OC4_semi <https://github.com/WISDEM/WEIS/tree/master/examples/04_NREL5MW_OC4_semi>`_ can be used to simulate the OC spar and semi, respectively, from WISDEM geometries.
..  * `05_IEA-3.4-130-RWT <https://github.com/WISDEM/WEIS/tree/master/examples/05_IEA-3.4-130-RWT>`_ runs the design load cases (DLCs) for the fixed-bottom IEA-3.4 turbine
..  * `06_IEA-15-240-RWT <https://github.com/WISDEM/WEIS/tree/master/examples/06_IEA-15-240-RWT>`_ contains several examples for running the IEA-15MW with the VolturnUS platform, including tower and structural controller optimization routines
..  * `15_RAFT_Studies <https://github.com/WISDEM/WEIS/tree/master/examples/15_RAFT_Studies>`_ contains an example for optimizing a the IEA-15MW with the VolturnUS platform in RAFT

.. More documentation specific to these examples can be found there, with more to follow.

.. This documentation only covers a summary of WEIS's functionality.  WEIS can be adapted to solve a wide variety of problems.  If you have questions or would like to discuss WEIS's functionality further, please email dzalkind (at) nrel (dot) gov. 

.. .. image:: images/WEIS_Flow.png
..   :width: 1000
..   :alt: Stack of software tools contained in WEIS
