Usage and Workflow
==================

RAFT requires an input design YAML file to describe the design of a floating offshore wind turbine. RAFT currently does not do any 
design processes--it is only a modeler--but it can be coupled with other programs like WISDEM and OpenFAST to optimize design variables.
Using the input design YAML, RAFT will compute the 6x6 matrices of the frequency domain equations of motion (see Theory section).
The primary outputs of RAFT will be the response amplitude operators (RAOs) of the FOWT, which describe the response of the FOWT in reference
to the input loads.

Running RAFT
------------

RAFT can be run by following the process in the figure below to compute the 6x6 matrices in the equations of motion
(equations shown in the Theory page).

.. image:: /images/workflow.JPG
    :align: center

Model Setup
^^^^^^^^^^^

The input design yaml is loaded into a python design dictionary and used to create and initialize a Model object.

.. code-block::
    
    # load the input design yaml
    with open('VolturnUS-S_example.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

    # Create the RAFT model
    model = raft.Model(design) 

Unloaded Condition Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model can then be analyzed in its equilibrium unloaded position using the method: model.analyzeUnloaded()

.. code-block::

    # calculate the system's constant properties
    for fowt in self.fowtList:
        fowt.calcStatics()
        fowt.calcHydroConstants(dict(wave_spectrum='still', wave_heading=0))
        
    # get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
    self.C_moor0 = self.ms.getCoupledStiffness(lines_only=True)           # this method accounts for eqiuilibrium of free objects in the system
    self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)
        
    # calculate platform offsets and mooring system equilibrium state
    self.calcMooringAndOffsets()
    self.results['properties']['offset_unloaded'] = self.fowtList[0].Xi0

Design Load Case Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

The model can also be dynamically analyzed for each design load case as specified by the input design yaml
using the method: model.analyzeCases()

.. code-block::

    # calculate the system's constant properties
    for fowt in self.fowtList:
        fowt.calcStatics()
        fowt.calcBEM()
        
    # loop through each case
    for iCase in range(nCases):
    
        # form dictionary of case parameters
        case = dict(zip( self.design['cases']['keys'], self.design['cases']['data'][iCase]))   

        # get initial FOWT values assuming no offset
        for fowt in self.fowtList:
            fowt.Xi0 = np.zeros(6)      # zero platform offsets
            fowt.calcTurbineConstants(case, ptfm_pitch=0.0)
            fowt.calcHydroConstants(case)
        
        # calculate platform offsets and mooring system equilibrium state
        self.calcMooringAndOffsets()
        
        # update values based on offsets if applicable
        for fowt in self.fowtList:
            fowt.calcTurbineConstants(case, ptfm_pitch=fowt.Xi0[4])
            #fowt.calcHydroConstants(case)  (hydrodynamics don't account for offset, so far)
        
        # solve system dynamics
        self.solveDynamics(case)
        
        # process outputs that are specific to the floating unit       
        self.fowtList[0].saveTurbineOutputs(self.results['case_metrics'], case, iCase, fowt.Xi0, self.Xi[0:6,:])            

        # process mooring tension outputs
        nLine = int(len(self.T_moor)/2)
        T_moor_amps = np.zeros([2*nLine, self.nw], dtype=complex) 
        for iw in range(self.nw):
            T_moor_amps[:,iw] = np.matmul(self.J_moor, self.Xi[:,iw])   # FFT of mooring tensions
        

Prominent outputs will be saved in the model's "results" variable and can be used for further plotting and visualization purposes.





Inputs
------

The input design YAML can be broken up into multiple parts. The following contains the various sections of an example
input file for the IEA 15MW turbine with the VolturnUS-S steel semi-submersible platform.

Modeling Settings
^^^^^^^^^^^^^^^^^

.. code-block:: python

    settings:                   # global Settings
        min_freq     :  0.005   #  [Hz]       lowest frequency to consider, also the frequency bin width 
        max_freq     :  0.40    #  [Hz]       highest frequency to consider
        XiStart      :   0      # sets initial amplitude of each DOF for all frequencies
        nIter        :  10      # sets how many iterations to perform in Model.solveDynamics()

Site Characteristics
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    site:
        water_depth : 200        # [m]      uniform water depth
        rho_water   : 1025.0     # [kg/m^3] water density
        rho_air     : 1.225      # [kg/m^3] air density
        mu_air      : 1.81e-05   #          air dynamic viscosity
        shearExp    : 0.12       #          shear exponent

Design Load Cases
^^^^^^^^^^^^^^^^^

.. code-block:: python

    cases:
        keys : [wind_speed, wind_heading, turbulence, turbine_status, yaw_misalign, wave_spectrum, wave_period, wave_height, wave_heading  ]
        data :  #   m/s        deg    % or e.g. 2B_NTM    string            deg         string          (s)         (m)         (deg)
            -  [    12,         0,            0.01,       operating,          0,        JONSWAP,         13.1,        8.5,           0        ]

Turbine
^^^^^^^

.. code-block:: python

    turbine:
        
        mRNA          :     991000        #  [kg]       RNA mass 
        IxRNA         :          0        #  [kg-m2]    RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
        IrRNA         :          0        #  [kg-m2]    RNA moment of inertia about local y or z axes [kg-m^2]
        xCG_RNA       :          0        #  [m]        x location of RNA center of mass [m] (Actual is ~= -0.27 m)
        hHub          :        150.0      #  [m]        hub height above water line [m]
        Fthrust       :       1500.0E3    #  [N]        temporary thrust force to use
        
        I_drivetrain: 318628138.0   # full rotor + drivetrain inertia as felt on the high-speed shaft
        
        nBlades     : 3     # number of blades
        Zhub        : 150.0        # hub height [m]
        Rhub        : 3.97        # hub radius [m]
        precone     : 4.0     # [deg]
        shaft_tilt  : 6.0     # [deg]
        overhang    : 12.0313 # [m]
        
        blade: 
            precurveTip : -3.9999999999999964  # 
            presweepTip : 0.0  # 
            Rtip        : 120.96999999936446         # rotor radius

            #    r    chord   theta  precurve  presweep  
            geometry: 
              - [     8.004,      5.228,     15.474,      0.035,      0.000 ]
              - [    12.039,      5.321,     14.692,      0.084,      0.000 ]
              - [    16.073,      5.458,     13.330,      0.139,      0.000 ]
              - [    20.108,      5.602,     11.644,      0.192,      0.000 ]
              - [    24.142,      5.718,      9.927,      0.232,      0.000 ]
              - [    28.177,      5.767,      8.438,      0.250,      0.000 ]
              - [    32.211,      5.713,      7.301,      0.250,      0.000 ]
              - [    36.246,      5.536,      6.232,      0.246,      0.000 ]
              - [    40.280,      5.291,      5.230,      0.240,      0.000 ]
              - [    44.315,      5.035,      4.348,      0.233,      0.000 ]
              - [    48.349,      4.815,      3.606,      0.218,      0.000 ]
              - [    52.384,      4.623,      2.978,      0.178,      0.000 ]
              - [    56.418,      4.432,      2.423,      0.100,      0.000 ]
              - [    60.453,      4.245,      1.924,      0.000,      0.000 ]
              - [    64.487,      4.065,      1.467,     -0.112,      0.000 ]
              - [    68.522,      3.896,      1.056,     -0.244,      0.000 ]
              - [    72.556,      3.735,      0.692,     -0.415,      0.000 ]
              - [    76.591,      3.579,      0.355,     -0.620,      0.000 ]
              - [    80.625,      3.425,      0.019,     -0.846,      0.000 ]
              - [    84.660,      3.268,     -0.358,     -1.080,      0.000 ]
              - [    88.694,      3.112,     -0.834,     -1.330,      0.000 ]
              - [    92.729,      2.957,     -1.374,     -1.602,      0.000 ]
              - [    96.763,      2.800,     -1.848,     -1.895,      0.000 ]
              - [   100.798,      2.637,     -2.136,     -2.202,      0.000 ]
              - [   104.832,      2.464,     -2.172,     -2.523,      0.000 ]
              - [   108.867,      2.283,     -2.108,     -2.864,      0.000 ]
              - [   112.901,      2.096,     -1.953,     -3.224,      0.000 ]
              - [   116.936,      1.902,     -1.662,     -3.605,      0.000 ]
            #    station(rel)      airfoil name 
            airfoils: 
              - [   0.00000, circular ]
              - [   0.02000, circular ]
              - [   0.15000, SNL-FFA-W3-500 ]
              - [   0.24517, FFA-W3-360 ]
              - [   0.32884, FFA-W3-330blend ]
              - [   0.43918, FFA-W3-301 ]
              - [   0.53767, FFA-W3-270blend ]
              - [   0.63821, FFA-W3-241 ]
              - [   0.77174, FFA-W3-211 ]
              - [   1.00000, FFA-W3-211 ]


        airfoils: 
          - name               : circular  # 
            relative_thickness : 1.0  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00010,    0.35000,   -0.00010 ] 
              - [  179.9087,    0.00010,    0.35000,   -0.00010 ] 
          - name               : SNL-FFA-W3-500  # 
            relative_thickness : 0.5  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9660,    0.00000,    0.08440,    0.00000 ] 
              - ... 
          - name               : FFA-W3-211  # 
            relative_thickness : 0.211  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.02464,    0.00000 ] 
              - ...
          - name               : FFA-W3-241  # 
            relative_thickness : 0.241  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.01178,    0.00000 ] 
              - ...
          - name               : FFA-W3-270blend  # 
            relative_thickness : 0.27  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.01545,    0.00000 ] 
              - ...
          - name               : FFA-W3-301  # 
            relative_thickness : 0.301  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.02454,    0.00000 ] 
              - ...
          - name               : FFA-W3-330blend  # 
            relative_thickness : 0.33  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.03169,    0.00000 ] 
              - ...
          - name               : FFA-W3-360  # 
            relative_thickness : 0.36  # 
            data:  #  alpha    c_l    c_d     c_m   
              - [ -179.9087,    0.00000,    0.03715,    0.00000 ] 
              - ...

   
        pitch_control:
          GS_Angles: [0.06019804, 0.08713416, 0.10844806, 0.12685912, 0.14339822,       0.1586021 , 0.17279614, 0.18618935, 0.19892772, 0.21111989,             0.22285021, 0.23417256, 0.2451469 , 0.25580691, 0.26619545,           0.27632495, 0.28623134, 0.29593266, 0.30544521, 0.314779  ,       0.32395154, 0.33297489, 0.3418577 , 0.35060844, 0.35923641,       0.36774807, 0.37614942, 0.38444655, 0.39264363, 0.40074407]
          GS_Kp: [-0.9394215 , -0.80602855, -0.69555026, -0.60254912, -0.52318192,       -0.45465531, -0.39489024, -0.34230736, -0.29568537, -0.25406506,       -0.2166825 , -0.18292183, -0.15228099, -0.12434663, -0.09877533,       -0.0752794 , -0.05361604, -0.0335789 , -0.01499149,  0.00229803,  0.01842102,  0.03349169,  0.0476098 ,  0.0608629 ,  0.07332812,  0.0850737 ,  0.0961602 ,  0.10664158,  0.11656607,  0.12597691]
          GS_Ki: [-0.07416547, -0.06719673, -0.0614251 , -0.05656651, -0.0524202 ,       -0.04884022, -0.04571796, -0.04297091, -0.04053528, -0.03836094,       -0.03640799, -0.03464426, -0.03304352, -0.03158417, -0.03024826,       -0.02902079, -0.02788904, -0.02684226, -0.02587121, -0.02496797,       -0.02412567, -0.02333834, -0.02260078, -0.02190841, -0.0212572 ,       -0.02064359, -0.0200644 , -0.01951683, -0.01899836, -0.01850671]
          Fl_Kp: -9.35
        wt_ops:
            v: [3.0, 3.266896551724138, 3.533793103448276, 3.800689655172414, 4.067586206896552, 4.334482758620689, 4.601379310344828, 4.868275862068966, 5.135172413793104, 5.402068965517241, 5.6689655172413795, 5.935862068965518, 6.2027586206896554, 6.469655172413793, 6.736551724137931, 7.00344827586207, 7.270344827586207, 7.537241379310345, 7.804137931034483, 8.071034482758622, 8.337931034482759, 8.604827586206897, 8.871724137931036, 9.138620689655173, 9.405517241379311, 9.672413793103448, 9.939310344827586, 10.206206896551725, 10.473103448275863, 10.74, 11.231724137931035, 11.723448275862069, 12.215172413793104, 12.706896551724139, 13.198620689655172, 13.690344827586207, 14.182068965517242, 14.673793103448276, 15.16551724137931, 15.657241379310346, 16.14896551724138, 16.640689655172416, 17.13241379310345, 17.624137931034483, 18.11586206896552, 18.607586206896553, 19.099310344827586, 19.591034482758623, 20.082758620689653, 20.57448275862069, 21.066206896551726, 21.557931034482756, 22.049655172413793, 22.54137931034483, 23.03310344827586, 23.524827586206897, 24.016551724137933, 24.508275862068963, 25.0]
            pitch_op: [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 3.57152, 5.12896, 6.36736, 7.43866, 8.40197, 9.28843, 10.1161, 10.8974,  11.641, 12.3529,  13.038, 13.6997, 14.3409, 14.9642, 15.5713, 16.1639, 16.7435, 17.3109, 17.8673, 18.4136, 18.9506, 19.4788, 19.9989, 20.5112, 21.0164, 21.5147, 22.0067, 22.4925, 22.9724]
            omega_op: [2.1486, 2.3397, 2.5309,  2.722, 2.9132, 3.1043, 3.2955, 3.4866, 3.6778, 3.8689, 4.0601, 4.2512, 4.4424, 4.6335, 4.8247, 5.0159,  5.207, 5.3982, 5.5893, 5.7805, 5.9716, 6.1628, 6.3539, 6.5451, 6.7362, 6.9274, 7.1185, 7.3097, 7.5008, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56, 7.56]
        gear_ratio: 1
        torque_control:
            VS_KP: -38609162.66552
            VS_KI: -4588245.18720
        
        
        tower:
            dlsMax       :  5.0     # maximum node splitting section amount; can't be 0
        
            name      :  tower                     # [-]    an identifier (no longer has to be number)       
            type      :  1                         # [-]    
            rA        :  [ 0, 0,  15]              # [m]    end A coordinates
            rB        :  [ 0, 0, 144.582]          # [m]    and B coordinates
            shape     :  circ                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]   twist angle about the member's z-axis
            
            # --- outer shell including hydro---
            stations  :  [ 15,  28,  28.001,  41,  41.001,  54,  54.001,  67,  67.001,  80,  80.001,  93,  93.001,  106,  106.001,  119,  119.001,  132,  132.001,  144.582 ]    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  [ 10,  9.964,  9.964,  9.967,  9.967,  9.927,  9.927,  9.528,  9.528,  9.149,  9.149,  8.945,  8.945,  8.735,  8.735,  8.405,  8.405,  7.321,  7.321,  6.5 ]    # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  [ 0.082954,  0.082954,  0.083073,  0.083073,  0.082799,  0.082799,  0.0299,  0.0299,  0.027842,  0.027842,  0.025567,  0.025567,  0.022854,  0.022854,  0.02025,  0.02025,  0.018339,  0.018339,  0.021211,  0.021211 ]                     # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.0                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  0.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            # (neglecting axial coefficients for now)
            CdEnd     :  0.0                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
            CaEnd     :  0.0                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3]   material density

Platform
^^^^^^^^            

.. code-block:: python

    platform:

        potModMaster :   1      # [int] master switch for potMod variables; 0=keeps all member potMod vars the same, 1=turns all potMod vars to False (no HAMS), 2=turns all potMod vars to True (no strip)
        dlsMax       :  5.0     # maximum node splitting section amount for platform members; can't be 0

        members:   # list all members here
            
          - name      :  center_column             # [-]    an identifier (no longer has to be number)       
            type      :  2                         # [-]    
            rA        :  [ 0, 0, -20]              # [m]    end A coordinates
            rB        :  [ 0, 0,  15]              # [m]    and B coordinates
            shape     :  circ                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]  twist angle about the member's z-axis
            potMod    :  True                      # [bool] Whether to model the member with potential flow (BEM model) plus viscous drag or purely strip theory
            # --- outer shell including hydro---
            stations  :  [0, 1]                    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  10.0                      # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  0.05                      # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.8                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  1.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            CdEnd     :  0.6                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
            CaEnd     :  0.6                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3] 
            # --- handling of end caps or any internal structures if we need them ---
            cap_stations :  [ 0    ]               # [m]  location along member of any inner structures (in same scaling as set by 'stations')
            cap_t        :  [ 0.001  ]             # [m]  thickness of any internal structures
            cap_d_in     :  [ 0    ]               # [m]  inner diameter of internal structures (0 for full cap/bulkhead, >0 for a ring shape)

            
          - name      :  outer_column              # [-]    an identifier (no longer has to be number)       
            type      :  2                         # [-]    
            rA        :  [51.75, 0, -20]           # [m]    end A coordinates
            rB        :  [51.75, 0,  15]           # [m]    and B coordinates
            heading   :  [ 60, 180, 300]           # [deg]  heading rotation of column about z axis (for repeated members)
            shape     :  circ                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]  twist angle about the member's z-axis
            potMod    :  True                      # [bool] Whether to model the member with potential flow (BEM model) plus viscous drag or purely strip theory
            # --- outer shell including hydro---
            stations  :  [0, 1]                    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  12.5                      # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  0.05                      # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.8                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  1.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            CdEnd     :  0.6                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
            CaEnd     :  0.6                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3] 
            # --- ballast ---
            l_fill    :  1.4                       # [m]
            rho_fill  :  5000                      # [kg/m3]
            # --- handling of end caps or any internal structures if we need them ---
            cap_stations :  [ 0    ]               # [m]  location along member of any inner structures (in same scaling as set by 'stations')
            cap_t        :  [ 0.001  ]             # [m]  thickness of any internal structures
            cap_d_in     :  [ 0    ]               # [m]  inner diameter of internal structures (0 for full cap/bulkhead, >0 for a ring shape)

            
          - name      :  pontoon                   # [-]    an identifier (no longer has to be number)       
            type      :  2                         # [-]    
            rA        :  [  5  , 0, -16.5]         # [m]    end A coordinates
            rB        :  [ 45.5, 0, -16.5]         # [m]    and B coordinates
            heading   :  [ 60, 180, 300]           # [deg]  heading rotation of column about z axis (for repeated members)
            shape     :  rect                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]  twist angle about the member's z-axis
            potMod    :  False                     # [bool] Whether to model the member with potential flow (BEM model) plus viscous drag or purely strip theory
            # --- outer shell including hydro---
            stations  :  [0, 1]                    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  [12.5, 7.0]               # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  0.05                      # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.8                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  1.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            CdEnd     :  0.6                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
            CaEnd     :  0.6                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3]
            l_fill    :  43.0                      # [m]
            rho_fill  :  1025.0                    # [kg/m3]
            
            
          - name      :  upper_support             # [-]    an identifier (no longer has to be number)       
            type      :  2                         # [-]    
            rA        :  [  5  , 0, 14.545]        # [m]    end A coordinates
            rB        :  [ 45.5, 0, 14.545]        # [m]    and B coordinates
            heading   :  [ 60, 180, 300]           # [deg]  heading rotation of column about z axis (for repeated members)
            shape     :  circ                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]  twist angle about the member's z-axis
            potMod    :  False                     # [bool] Whether to model the member with potential flow (BEM model) plus viscous drag or purely strip theory
            # --- outer shell including hydro---
            stations  :  [0, 1]                    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  0.91                      # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  0.01                      # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.8                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  1.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            CdEnd     :  0.6                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
            CaEnd     :  0.6                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3] 
        

Mooring
^^^^^^^

.. code-block:: python

    mooring:
        water_depth: 200                                  # [m]       uniform water depth
        
        points:
            - name: line1_anchor
              type: fixed
              location: [-837, 0.0, -200.0]
              anchor_type: drag_embedment

            - name: line2_anchor
              type: fixed
              location: [418, 725, -200.0]
              anchor_type: drag_embedment

            - name: line3_anchor
              type: fixed
              location: [418, -725, -200.0]
              anchor_type: drag_embedment

            - name: line1_vessel
              type: vessel
              location: [-58,      0.0,     -14.0]

            - name: line2_vessel
              type: vessel
              location: [29,      50,     -14.0]

            - name: line3_vessel
              type: vessel
              location: [29,     -50,     -14.0]

        lines:
            - name: line1
              endA: line1_anchor
              endB: line1_vessel
              type: chain
              length: 850

            - name: line2
              endA: line2_anchor
              endB: line2_vessel
              type: chain
              length: 850

            - name: line3
              endA: line3_anchor
              endB: line3_vessel
              type: chain
              length: 850

        line_types:
            - name: chain
              diameter:         0.185
              mass_density:   685.0
              stiffness:     3270e6
              breaking_load:    1e8
              cost: 100.0
              transverse_added_mass: 1.0
              tangential_added_mass: 0.0
              transverse_drag: 1.6
              tangential_drag: 0.1

        anchor_types:
            - name: drag_embedment
              mass: 1e3
              cost: 1e4
              max_vertical_load: 0.0
              max_lateral_load: 1e5



Outputs
-------

The main output of RAFT is the model's RAOs. More information can be extracted through the model.calcOutputs() method, which is currently
being finalized.

.. image:: /images/output.JPG
    :align: center
   
   





