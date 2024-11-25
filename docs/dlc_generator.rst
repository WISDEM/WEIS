=========================
Design Load Cases in WEIS
=========================

Design load cases (DLCs) specify the conditions that a turbine must operate in safely thorughout its lifetime.
These load cases are defined in IEC standards.
We supplement the standards with information from the DTU design load basis (cite).

--------------------------
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
A complete listing of the DLC options can be found here `DLC options`_ below.

-------------------
Reviewing DLC Cases
-------------------

The DLC Case Matrix
-------------------

WEIS creates a case matrix for each DLC; this case matrix, is located in the same directory as the openfast runs in both yaml and txt format.
The case matrix specifies which inputs are changed for each case.
An example case matrix is shown next::

   Case_ID  AeroDyn15      AeroDyn15       ElastoDyn  ElastoDyn  ElastoDyn  ElastoDyn  ElastoDyn  Fst    Fst    HydroDyn  HydroDyn  HydroDyn  InflowWind   TurbSim   
           WakeMod       tau1_const       BlPitch1   BlPitch2   BlPitch3    NacYaw    RotSpeed  TMax  TStart   WaveHs   WaveMod    WaveTp   HWindSpeed  RandSeed1  
    0         1      25.353075267567498   0.000535   0.000535   0.000535      0        5.6819   10.0   0.0      9.7        2        13.6       8.0      1501552846 
    1         1      25.353075267567498   0.000535   0.000535   0.000535      0        5.6819   10.0   0.0      9.7        2        13.6       8.0      488200390  
    2         1      9.536058651858337    9.189114   9.189114   9.189114      0       7.559987  10.0   0.0      9.7        2        13.6       15.0     1693606511 
    3         1      9.536058651858337    9.189114   9.189114   9.189114      0       7.559987  10.0   0.0      9.7        2        13.6       15.0     680233354  

This case matrix represents DLC 6.1 and shows the initial conditions (BlPitch*, RotSpeed) as well as the sea state (WaveHs, WaveTp) and wind condtions (HWindSpeed, RandSeed1) for each case.

Modeling Option Outputs
-----------------------

Additionally, the DLC options are outputted in the modeling options of WEIS.
These outputs can be used as inputs for future runs to exactly reproduce specific cases::

  DLC_driver:
    DLCs:
       -  DLC: '1.1'
          wind_speed: [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0]
          ws_bin_size: 2
          n_seeds: 1
          n_azimuth: 1
          wind_seed: [1501552846, 488200390, 1693606511, 680233354, 438466540, 1712329281, 1380152456, 1452245847, 2122694022, 839901364, 1802651553, 714712467]
          wave_seeds: [304524126, 1192975140, 1668389755, 489172031, 349375909, 208787680, 426140584, 37937012, 1601914564, 1619243266, 413287548, 918595805]
          wind_heading: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          turbine_status: operating
          wave_period: [8.3, 8.3, 7.7, 7.1, 6.3, 6.1, 6.2, 6.2, 6.7, 7.1, 7.1, 7.7]
          wave_height: [0.84, 0.87, 0.99, 1.15, 1.34, 1.58, 1.82, 2.08, 2.34, 2.66, 2.98, 3.28]
          wave_heading: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          wave_gamma: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          probabilities: [0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333]
          IEC_WindType: NTM
          analysis_time: 600.0
          transient_time: 120.0
          shutdown_time: 9999
          turbulent_wind: {flag: false, Echo: false, RandSeed1: 1, RandSeed2: RANLUX, WrBHHTP: false, WrFHHTP: false, WrADHH: false, WrADFF: true, WrBLFF: false, WrADTWR: false, WrFMTFF: false, WrACT: false, Clockwise: false, ScaleIEC: 0, NumGrid_Z: 25, NumGrid_Y: 25, TimeStep: 0.05, UsableTime: ALL, HubHt: 0, GridHeight: 0, GridWidth: 0, VFlowAng: 0.0, HFlowAng: 0.0, TurbModel: IECKAI, UserFile: unused, IECstandard: 1-ED3, ETMc: default, WindProfileType: PL, ProfileFile: unused, RefHt: 0, URef: -1, IECturbc: -1, ZJetMax: default, PLExp: -1, Z0: default, Latitude: default, RICH_NO: 0.05, UStar: default, ZI: default, PC_UW: default, PC_UV: default, PC_VW: default, SCMod1: default, SCMod2: default, SCMod3: default, InCDec1: default, InCDec2: default, InCDec3: default, CohExp: default, CTEventPath: unused, CTEventFile: RANDOM, Randomize: true, DistScl: 1.0, CTLy: 0.5, CTLz: 0.5, CTStartTime: 30}
          wake_mod: 1
          wave_model: 2
          label: '1.1'
          sea_state: normal
          PSF: 1.35
          yaw_misalign: [0]
          total_time: 720.0
          pitch_initial: [2.426047, 0.377375, 0.000535, 0.000535, 1.170321, 6.052129, 9.189114, 11.824437, 14.19975, 16.42107, 18.525951, 20.553121]
          rot_speed_initial: [5.000012, 5.000012, 5.000012, 6.390847, 7.559987, 7.559987, 7.559987, 7.559987, 7.559987, 7.559987, 7.559987, 7.559987]
          tau1_const: [69.824578732105, 40.971862185516514, 29.12612763770409, 22.53606690450444, 16.17186559734712, 11.569392828413536, 9.536058651858337, 8.207963142590629, 7.2407552404030975, 6.493729028184553, 5.894677553483227, 5.401388775677793]

--------------------------------
Expected DLC Outputs in OpenFAST
--------------------------------

Coming soon.

.. DLC 1.1
.. -------
.. For each:
.. Short description.
.. Defaults?
.. Timeseries

.. DLC 1.2
.. -------

.. DLC 1.3
.. -------

.. DLC 1.4
.. -------

.. DLC 1.5
.. -------

.. DLC 1.6
.. -------

.. DLC 2.1  (AG)
.. -------

.. DLC 2.2 (not yet)  This one has the joint probability
.. -------

.. DLC 2.3
.. -------

.. DLC 2.4
.. -------

.. DLC 3.1
.. -------

.. DLC 3.2
.. -------

.. DLC 3.3
.. -------

.. DLC 4.1
.. -------

.. DLC 4.2
.. -------

.. DLC 5.1
.. -------

.. DLC 6.1
.. -------

.. DLC 6.2
.. -------

.. DLC 6.3
.. -------

.. DLC 6.4
.. -------

.. DLC 7.1
.. -------

.. DLC 7.2
.. -------



.. ------------------------
.. User-defined DLC Example
.. ------------------------


------------------------------------
Setting Up DLCs (for developers)
------------------------------------

In the dlc_generator class (``/weis/dlc_driver/dlc_generator.py``), you can add new functions for additional DLCs.
Several examples are already there, like ``generate_2p3()``.  New functions should follow the ``generate_*`` naming convention.  Note that ``.`` is automatically mapped to ``p``.

The function should start with some helpful comments::

  # Power production normal turbulence model - normal sea state

The ``dlc_options`` dictionary contains inputs for that particular DLC in the modeling options.  Default options include some modeling options common across DLCs.::

  # Get default options
  dlc_options.update(self.default_options)   

Next, options specific to that DLC hard-coded in the function.  Error checking may be helpful here, too::
  
  # Handle DLC Specific options:
  dlc_options['label'] = '1.1'
  dlc_options['sea_state'] = 'normal'
  dlc_options['PSF'] = 1.35

  # Set yaw_misalign, else default
  if 'yaw_misalign' in dlc_options:
      dlc_options['yaw_misalign'] = dlc_options['yaw_misalign']
  else: # default
      dlc_options['yaw_misalign'] = [0]

Now, the special part happens, where we define groups of variables that are grouped and the cases are a cartesian product of the groups.
For example in this DLC 1.1 example::

  # DLC-specific: define groups
  # These options should be the same length and we will generate a matrix of all cases
  generic_case_inputs = []
  generic_case_inputs.append(['total_time','transient_time'])  # group 0, (usually constants) turbine variables, DT, aero_modeling
  generic_case_inputs.append(['wind_speed','wave_height','wave_period', 'wind_seed','wave_seed']) # group 1, initial conditions will be added here, define some method that maps wind speed to ICs and add those variables to this group
  generic_case_inputs.append(['yaw_misalign']) # group 2

The time and other constant options are in the first group.  This group usually has a length of one.
Wind speed, wave height, wave period, and the seeds are varied together in the second group.  
For example the wind speed may be 8, 10, and 12, and the corresponding wave height/period will vary with the wind speed. 
Initial conditions are automatically applied in this group via linear interpolation.  Search for the ``initial_condition_table`` dictionary.
The wind speed and other metocean conditions are added to the dlc_options automatically.  
The developer only needs to provide specific values in certain cases, like DLC 6.1.
If the user also wants to vary the yaw_misalign, those offsets will be applied on each wind speed.

Finally, the ``generate_cases`` method will do the rest of the work and (hopefully) check for errors along the way::

  self.generate_cases(generic_case_inputs,dlc_options)

Any options you want to vary across should be added to the ``dlc_options`` dictionary.  


.. _DLC options:

------------------------
DLC Option Input Listing
------------------------

The following inputs are a subset of the options available in the ``modeling_options`` file.

.. jsonschema:: inputs/modeling_schema.json#/definitions/DLC_driver
   :hide_key_if_empty: /**/default
