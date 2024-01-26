******************************
/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_modeling_schema.yaml
******************************
Schema that describes the modeling options for WEIS


/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_modeling_schema.



General
****************************************

:code:`verbosity` : Boolean
    Prints additional outputs to screen (and to a file log in the
    future)

    *Default* = False

:code:`solver_maxiter` : Integer
    Number of iterations for the top-level coupling solver

    *Default* = 5



openfast_configuration
########################################

:code:`OF_run_fst` : String
    Filename prefix for output files

    *Default* = none

:code:`OF_run_dir` : String
    Path to place FAST output files (e.g.
    /home/user/myturbines/output)

    *Default* = none

:code:`generate_af_coords` : Boolean
    Flag to write airfoil coordinates out or not

    *Default* = False

:code:`use_exe` : Boolean
    Use openfast executable instead of library

    *Default* = False

:code:`model_only` : Boolean
    Flag to only generate an OpenFAST model and stop

    *Default* = False

:code:`save_timeseries` : Boolean
    Save openfast output timeseries

    *Default* = True

:code:`keep_time` : Boolean
    Keep timeseries in openmdao_openfast for post-processing

    *Default* = True

:code:`save_iterations` : Boolean
    Save summary stats and other info for each openfast iteration.
    Could bump this up to a more global post-processing input.

    *Default* = True

:code:`FAST_exe` : String
    Path to FAST executable to override default WEIS value (e.g.
    /home/user/OpenFAST/bin/openfast)

    *Default* = none

:code:`FAST_lib` : String
    Path to FAST dynamic library to override default WEIS value (e.g.
    /home/user/OpenFAST/lib/libopenfast.so)

    *Default* = none

:code:`path2dll` : String
    Path to controller shared library (e.g.
    /home/user/myturbines/libdiscon.so)

    *Default* = none

:code:`allow_fails` : Boolean
    Allow WEIS to continue if OpenFAST fails?  All outputs will be
    filled with fail_value. Use with caution!

    *Default* = False

:code:`fail_value` : Float


    *Default* = -9999

:code:`goodman_correction` : Boolean
    Flag whether to apply the Goodman correction for mean stress value
    to the stress amplitude value in fatigue calculations

    *Default* = False



WISDEM
****************************************

Options for running WISDEM.  No further options are included in this file.  They are populated using the modeling schema in the WISDEM project in python.
:code:`n_dlc` : Integer
    Number of load cases

    *Default* = 1

    *Minimum* = 0



RotorSE
########################################

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False

:code:`n_aoa` : Integer
    Number of angles of attack in a common grid to define polars

    *Default* = 200

:code:`n_xy` : Integer
    Number of coordinate point used to define airfoils

    *Default* = 200

:code:`n_span` : Integer
    Number of spanwise stations in a common grid used to define blade
    properties

    *Default* = 30

:code:`n_pc` : Integer
    Number of wind speeds to compute the power curve

    *Default* = 20

:code:`n_pc_spline` : Integer
    Number of wind speeds to spline the power curve

    *Default* = 200

:code:`n_pitch_perf_surfaces` : Integer
    Number of pitch angles to determine the Cp-Ct-Cq-surfaces

    *Default* = 20

:code:`min_pitch_perf_surfaces` : Float
    Min pitch angle of the Cp-Ct-Cq-surfaces

    *Default* = -5.0

:code:`max_pitch_perf_surfaces` : Float
    Max pitch angle of the Cp-Ct-Cq-surfaces

    *Default* = 30.0

:code:`n_tsr_perf_surfaces` : Integer
    Number of tsr values to determine the Cp-Ct-Cq-surfaces

    *Default* = 20

:code:`min_tsr_perf_surfaces` : Float
    Min TSR of the Cp-Ct-Cq-surfaces

    *Default* = 2.0

:code:`max_tsr_perf_surfaces` : Float
    Max TSR of the Cp-Ct-Cq-surfaces

    *Default* = 12.0

:code:`n_U_perf_surfaces` : Integer
    Number of wind speeds to determine the Cp-Ct-Cq-surfaces

    *Default* = 1

:code:`regulation_reg_III` : Boolean
    Flag to derive the regulation trajectory in region III in terms of
    pitch and TSR

    *Default* = True

:code:`peak_thrust_shaving` : Boolean
    If True, apply peak thrust shaving within RotorSE.

    *Default* = False

:code:`thrust_shaving_coeff` : Float
    Scalar applied to the max torque within RotorSE for peak thrust
    shaving. Only used if `peak_thrust_shaving` is True.

    *Default* = 1.0

:code:`fix_pitch_regI12` : Boolean
    If True, pitch is fixed in region I1/2, i.e. when min rpm is
    enforced.

    *Default* = False

:code:`spar_cap_ss` : String
    Composite layer modeling the spar cap on the suction side in the
    geometry yaml. This entry is used to compute ultimate strains and
    it is linked to the design variable spar_cap_ss.

    *Default* = none

:code:`spar_cap_ps` : String
    Composite layer modeling the spar cap on the pressure side in the
    geometry yaml. This entry is used to compute ultimate strains and
    it is linked to the design variable spar_cap_ps.

    *Default* = none

:code:`te_ss` : String
    Composite layer modeling the trailing edge reinforcement on the
    suction side in the geometry yaml. This entry is used to compute
    ultimate strains and it is linked to the design variable te_ss.

    *Default* = none

:code:`te_ps` : String
    Composite layer modeling the trailing edge reinforcement on the
    pressure side in the geometry yaml. This entry is used to compute
    ultimate strains and it is linked to the design variable te_ps.

    *Default* = none

:code:`gamma_freq` : Float
    Partial safety factor on modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gust_std` : Float
    Number of standard deviations for strength of gust

    *Default* = 3.0

    *Minimum* = 0.0    *Maximum* = 15.0


:code:`root_fastener_s_f` : Float
    Safety factor for the max stress of blade root fasteners

    *Default* = 2.5

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`hubloss` : Boolean
    Include Prandtl hub loss model in CCBlade calls

    *Default* = True

:code:`tiploss` : Boolean
    Include Prandtl tip loss model in CCBlade calls

    *Default* = True

:code:`wakerotation` : Boolean
    Include effect of wake rotation (i.e., tangential induction factor
    is nonzero) in CCBlade calls

    *Default* = True

:code:`usecd` : Boolean
    Use drag coefficient in computing induction factors in CCBlade
    calls

    *Default* = True

:code:`n_sector` : Integer
    Number of sectors to divide rotor face into in computing thrust
    and power.

    *Default* = 4

    *Minimum* = 1    *Maximum* = 10


:code:`3d_af_correction` : Boolean
    Flag switching on and off the 3d DU-Selig airfoil correction
    implemented in Polar.py

    *Default* = True

:code:`inn_af` : Boolean
    Flag switching on and off the inverted neural network for airfoil
    design

    *Default* = False

:code:`inn_af_max_rthick` : Float
    Maximum airfoil thickness supported by the INN for airfoil design

    *Default* = 0.4

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`inn_af_min_rthick` : Float
    Minimum airfoil thickness supported by the INN for airfoil design

    *Default* = 0.15

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`rail_transport` : Boolean
    Flag switching on and off the rail transport module of RotorSE

    *Default* = False



DriveSE
########################################

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False

:code:`model_generator` : Boolean
    Whether or not to do detailed generator modeling using tools
    formerly in GeneratorSE

    *Default* = False

:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0




hub
========================================

:code:`hub_gamma` : Float
    Partial safety factor for hub sizing

    *Default* = 2.0

    *Minimum* = 1.0    *Maximum* = 7.0


:code:`spinner_gamma` : Float
    Partial safety factor for spinner sizing

    *Default* = 1.5

    *Minimum* = 1.0    *Maximum* = 5.0




TowerSE
########################################

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False

:code:`wind` : String from, ['PowerWind', 'LogisticWind']
    Wind scaling relationship with height

    *Default* = PowerWind

:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_b` : Float
    Partial safety factor for buckling

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_freq` : Float
    Partial safety factor on modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_fatigue` : Float
    Partial safety factor for fatigue failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`buckling_method` : String from, ['Eurocode', 'Euro-code', 'eurocode', 'euro-code', 'DNVGL', 'dnvgl', 'DNV-GL', 'dnv-gl']
    Buckling utilization calculation method- Eurocode 1994 or DNVGL
    RP-C202

    *Default* = dnvgl

:code:`buckling_length` : Float, m
    Buckling length factor in Eurocode safety check

    *Default* = 10.0

    *Minimum* = 1.0    *Maximum* = 100.0




frame3dd
========================================

Set of Frame3DD options used for tower analysis
:code:`shear` : Boolean
    Inclusion of shear area for symmetric sections

    *Default* = True

:code:`geom` : Boolean
    Inclusion of shear stiffening through axial loading

    *Default* = True

:code:`modal_method` : Float
    Eigenvalue solver 1=Subspace-Jacobi iteration, 2=Stodola (matrix
    iteration)

    *Default* = 1

:code:`tol` : Float
    Convergence tolerance for modal eigenvalue solution

    *Default* = 1e-09

    *Minimum* = 1e-12    *Maximum* = 0.1


:code:`n_refine` : Integer
    Number of Frame3DD element refinements for every specified section
    along tower/member

    *Default* = 3



FixedBottomSE
########################################

:code:`type` : String
    Can be `monopile` or `jacket`.

    *Default* = monopile

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False

:code:`wind` : String from, ['PowerWind', 'LogisticWind']
    Wind scaling relationship with height

    *Default* = PowerWind

:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_b` : Float
    Partial safety factor for buckling

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_freq` : Float
    Partial safety factor on modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_fatigue` : Float
    Partial safety factor for fatigue failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`buckling_method` : String from, ['Eurocode', 'Euro-code', 'eurocode', 'euro-code', 'DNVGL', 'dnvgl', 'DNV-GL', 'dnv-gl']
    Buckling utilization calculation method- Eurocode 1994 or DNVGL
    RP-C202

    *Default* = dnvgl

:code:`buckling_length` : Float, m
    Buckling length factor in Eurocode safety check

    *Default* = 10.0

    *Minimum* = 1.0    *Maximum* = 100.0




frame3dd
========================================

Set of Frame3DD options used for tower analysis
:code:`shear` : Boolean
    Inclusion of shear area for symmetric sections

    *Default* = True

:code:`geom` : Boolean
    Inclusion of shear stiffening through axial loading

    *Default* = True

:code:`modal_method` : Float
    Eigenvalue solver 1=Subspace-Jacobi iteration, 2=Stodola (matrix
    iteration)

    *Default* = 1

:code:`tol` : Float
    Convergence tolerance for modal eigenvalue solution

    *Default* = 1e-09

    *Minimum* = 1e-12    *Maximum* = 0.1


:code:`soil_springs` : Boolean
    If False, then a monopile is modeled with a perfectly clamped
    foundation.  If True, then spring-stiffness equivalents are
    computed from soil properties for all DOF.

    *Default* = False

:code:`gravity_foundation` : Boolean
    Model the monopile base as a gravity-based foundation with no pile
    embedment

    *Default* = False

:code:`n_refine` : Integer
    Number of Frame3DD element refinements for every specified section
    along tower/member

    *Default* = 3

:code:`n_legs` : Integer
    Number of legs for the jacket. Only used if `type`==`jacket`.

    *Default* = 4

:code:`n_bays` : Integer
    Number of bays for the jacket, or x-joints per tower leg pair.
    Only used if `type`==`jacket`.

    *Default* = 3

:code:`mud_brace` : Boolean
    If true, add a mud brace at the bottom of each jacket leg. Only
    used if `type`==`jacket`.

    *Default* = True

:code:`save_truss_figures` : Boolean
    If true, save .pngs of the jacket truss during analysis or
    optimization. Jacket only.

    *Default* = False



BOS
########################################

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False



FloatingSE
########################################

:code:`flag` : Boolean
    Whether or not to run this module

    *Default* = False

:code:`n_refine` : Integer
    Number of Frame3DD element refinements for every specified section
    along tower/member

    *Default* = 1



frame3dd
========================================

Set of Frame3DD options used for floating tower analysis
:code:`shear` : Boolean
    Inclusion of shear area for symmetric sections

    *Default* = False

:code:`geom` : Boolean
    Inclusion of shear stiffening through axial loading

    *Default* = False

:code:`modal_method` : Float
    Eigenvalue solver 1=Subspace-Jacobi iteration, 2=Stodola (matrix
    iteration)

    *Default* = 2

:code:`shift` : Float
    Numerical matrix diagonal adder for eigenvalue solve of
    unrestrained structure

    *Default* = 10.0

:code:`tol` : Float
    Convergence tolerance for modal eigenvalue solution

    *Default* = 1e-08

    *Minimum* = 1e-12    *Maximum* = 0.1


:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_b` : Float
    Partial safety factor for buckling

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_freq` : Float
    Partial safety factor on modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_fatigue` : Float
    Partial safety factor for fatigue failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`symmetric_moorings` : Boolean
    Whether or not to assume a symmetric mooring system

    *Default* = True

:code:`rank_and_file` : Boolean
    Use the rank-and-file method of identifying mode shapes that
    guarantees modeshape numbers in all directions, but will reuse the
    same modeshape for multiple directions

    *Default* = False



Loading
########################################

This is only used if not running the full WISDEM turbine Group and you need to input the mass properties, forces, and moments for a tower-only or nacelle-only analysis
:code:`mass` : Float, kilogram
    Mass at external boundary of the system.  For the tower, this
    would be the RNA mass.

    *Default* = 0.0

:code:`center_of_mass` : Array of Floats, meter
    Distance from system boundary to center of mass of the applied
    load.  For the tower, this would be the RNA center of mass in
    tower-top coordinates.

    *Default* = [0.0, 0.0, 0.0]

:code:`moment_of_inertia` : Array of Floats, kg*m^2
    Moment of inertia of external mass in coordinate system at the
    system boundary.  For the tower, this would be the RNA MoI in
    tower-top coordinates.

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



loads
========================================

:code:`force` : Array of Floats, Newton
    Force vector applied at system boundary

    *Default* = [0.0, 0.0, 0.0]

:code:`moment` : Array of Floats, N*m
    Force vector applied at system boundary

    *Default* = [0.0, 0.0, 0.0]

:code:`velocity` : Float, meter
    Applied wind reference velocity, if necessary

    *Default* = 0.0



Level1
****************************************

Options for WEIS fidelity level 1 = frequency domain (RAFT)
:code:`flag` : Boolean
    Whether or not to run WEIS fidelity level 1 = frequency domain
    (RAFT)

    *Default* = False

:code:`min_freq` : Float, Hz
    Minimum frequency to evaluate (frequencies will be
    min_freq:min_freq:max_freq)

    *Default* = 0.0159

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`max_freq` : Float, Hz
    Maximum frequency to evaluate (frequencies will be
    min_freq:min_freq:max_freq)

    *Default* = 0.3183

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`potential_bem_members` : Array of Strings
    List of submerged member names to model with potential flow
    boundary element methods.  Members not listed here will be modeled
    with strip theory

    *Default* = []

:code:`potential_model_override` : Integer
    User override for potential boundary element modeling. 0 = uses
    the potential_bem_members list for inviscid force and computes
    viscous drag with strip theory (members not listed use only strip
    theory), 1 = no potential BEM modeling for any member (just strip
    theory), 2 = potential BEM modeling for all members (no strip
    theory)

    *Default* = 0

:code:`xi_start` : Float
    Initial amplitude of each DOF for all frequencies

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`nIter` : Integer
    Number of iterations to solve dynamics

    *Default* = 15

    *Minimum* = 1    *Maximum* = 100


:code:`dls_max` : Integer
    Maximum node splitting section amount

    *Default* = 5

    *Minimum* = 1    *Maximum* = 100


:code:`min_freq_BEM` : Float, Hz
    lowest frequency and frequency interval to use in BEM analysis

    *Default* = 0.0159

    *Minimum* = 0.0    *Maximum* = 2.0


:code:`trim_ballast` : Integer
    Use RAFT to trim ballast so that average heave is near 0 (0 - no
    trim, 1 - adjust compartment fill values, 2 - adjust ballast
    density, recommended for now)

    *Default* = 0

:code:`heave_tol` : Float, m
    Heave tolerance for trim_ballast

    *Default* = 1

    *Minimum* = 0

:code:`save_designs` : Boolean
    Save RAFT design iterations in <outputs>/raft_designs

    *Default* = False

:code:`runPyHAMS` : Boolean
    Flag to run pyHAMS

    *Default* = True



Level3
****************************************

Options for WEIS fidelity level 3 = nonlinear time domain
:code:`flag` : Boolean
    Whether or not to run WEIS fidelity level 3 = nonlinear time
    domain (Linearize OpenFAST)

    *Default* = False



simulation
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`AbortLevel` : String from, ['WARNING', 'SEVERE', 'FATAL']
    Error level when simulation should abort (string) {'WARNING',
    'SEVERE', 'FATAL'}

    *Default* = FATAL

:code:`DT` : Float, s
    Integration time step (s)

    *Default* = 0.025

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`InterpOrder` : String from, ['1', '2', 'linear', 'Linear', 'LINEAR', 'quadratic', 'Quadratic', 'QUADRATIC']
    Interpolation order for input/output time history (-) {1=linear,
    2=quadratic}

    *Default* = 2

:code:`NumCrctn` : Integer
    Number of correction iterations (-) {0=explicit calculation, i.e.,
    no corrections}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 10


:code:`DT_UJac` : Float, s
    Time between calls to get Jacobians (s)

    *Default* = 99999.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`UJacSclFact` : Float
    Scaling factor used in Jacobians (-)

    *Default* = 1000000.0

    *Minimum* = 0.0    *Maximum* = 1000000000.0


:code:`CompElast` : Integer
    Compute structural dynamics (switch) {1=ElastoDyn; 2=ElastoDyn +
    BeamDyn for blades}

    *Default* = 1

:code:`CompInflow` : Integer
    Compute inflow wind velocities (switch) {0=still air;
    1=InflowWind; 2=external from OpenFOAM}

    *Default* = 1

:code:`CompAero` : Integer
    Compute aerodynamic loads (switch) {0=None; 1=AeroDyn v14;
    2=AeroDyn v15}

    *Default* = 2

:code:`CompServo` : Integer
    Compute control and electrical-drive dynamics (switch) {0=None;
    1=ServoDyn}

    *Default* = 1

:code:`CompHydro` : Integer
    Compute hydrodynamic loads (switch) {0=None; 1=HydroDyn}

    *Default* = 0

:code:`CompSub` : Integer
    Compute sub-structural dynamics (switch) {0=None; 1=SubDyn;
    2=External Platform MCKF}

    *Default* = 0

:code:`CompMooring` : Integer
    Compute mooring system (switch) {0=None; 1=MAP++; 2=FEAMooring;
    3=MoorDyn; 4=OrcaFlex}

    *Default* = 0

:code:`CompIce` : Integer
    Compute ice loads (switch) {0=None; 1=IceFloe; 2=IceDyn}

    *Default* = 0

:code:`MHK` : Integer
    MHK turbine type (switch) {0=Not an MHK turbine; 1=Fixed MHK
    turbine; 2=Floating MHK turbine}

    *Default* = 0

:code:`Gravity` : Float, m / s**2
    Gravitational acceleration (m/s^2)

    *Default* = 9.81

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AirDens` : Float, kg/m**3
    Air density (kg/m^3)

    *Default* = 1.225

:code:`WtrDens` : Float, kg/m**3
    Water density (kg/m^3)

    *Default* = 1025

:code:`KinVisc` : Float
    Kinematic viscosity of working fluid (m^2/s)

    *Default* = 1.464e-05

:code:`SpdSound` : Float
    Speed of sound in working fluid (m/s)

    *Default* = 335

:code:`Patm` : Float
    Atmospheric pressure (Pa) [used only for an MHK turbine cavitation
    check]

    *Default* = 103500

:code:`Pvap` : Float
    Vapour pressure of working fluid (Pa) [used only for an MHK
    turbine cavitation check]

    *Default* = 1700

:code:`WtrDpth` : Float
    Water depth (m)

    *Default* = 300

:code:`MSL2SWL` : Float
    Offset between still-water level and mean sea level (m) [positive
    upward]

    *Default* = 0

:code:`EDFile` : String
    Name of file containing ElastoDyn input parameters (quoted string)

    *Default* = none

:code:`BDBldFile(1)` : String
    Name of file containing BeamDyn input parameters for blade 1
    (quoted string)

    *Default* = none

:code:`BDBldFile(2)` : String
    Name of file containing BeamDyn input parameters for blade 2
    (quoted string)

    *Default* = none

:code:`BDBldFile(3)` : String
    Name of file containing BeamDyn input parameters for blade 3
    (quoted string)

    *Default* = none

:code:`InflowFile` : String
    Name of file containing inflow wind input parameters (quoted
    string)

    *Default* = none

:code:`AeroFile` : String
    Name of file containing aerodynamic input parameters (quoted
    string)

    *Default* = none

:code:`ServoFile` : String
    Name of file containing control and electrical-drive input
    parameters (quoted string)

    *Default* = none

:code:`HydroFile` : String
    Name of file containing hydrodynamic input parameters (quoted
    string)

    *Default* = none

:code:`SubFile` : String
    Name of file containing sub-structural input parameters (quoted
    string)

    *Default* = none

:code:`MooringFile` : String
    Name of file containing mooring system input parameters (quoted
    string)

    *Default* = none

:code:`IceFile` : String
    Name of file containing ice input parameters (quoted string)

    *Default* = none

:code:`SumPrint` : Boolean
    Print summary data to '<RootName>.sum' (flag)

    *Default* = False

:code:`SttsTime` : Float, s
    Amount of time between screen status messages (s)

    *Default* = 10.0

    *Minimum* = 0.01    *Maximum* = 1000.0


:code:`ChkptTime` : Float, s
    Amount of time between creating checkpoint files for potential
    restart (s)

    *Default* = 99999.0

    *Minimum* = 0.01    *Maximum* = 1000000.0


:code:`DT_Out` : Float
    Time step for tabular output (s) (or 'default')

    *Default* = 0

:code:`OutFileFmt` : Integer
    Format for tabular (time-marching) output file (switch) {1 text
    file [<RootName>.out], 2 binary file [<RootName>.outb], 3 both}

    *Default* = 2

:code:`TabDelim` : Boolean
    Use tab delimiters in text tabular output file? (flag) (currently
    unused)

    *Default* = True

:code:`OutFmt` : String
    Format used for text tabular output (except time).  Resulting
    field should be 10 characters. (quoted string (currently unused)

    *Default* = ES10.3E2

:code:`Linearize` : Boolean
    Linearization analysis (flag)

    *Default* = False

:code:`CalcSteady` : Boolean
    Calculate a steady-state periodic operating point before
    linearization? [unused if Linearize=False] (flag)

    *Default* = False

:code:`TrimCase` : String from, ['1', '2', '3', 'yaw', 'Yaw', 'YAW', 'torque', 'Torque', 'TORQUE', 'pitch', 'Pitch', 'PITCH']
    Controller parameter to be trimmed {1:yaw; 2:torque; 3:pitch}
    [used only if CalcSteady=True] (-)

    *Default* = 3

:code:`TrimTol` : Float
    Tolerance for the rotational speed convergence [used only if
    CalcSteady=True] (-)

    *Default* = 0.001

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`TrimGain` : Float, kg*m^2/rad/s
    Proportional gain for the rotational speed error (>0) [used only
    if CalcSteady=True] (rad/(rad/s) for yaw or pitch; Nm/(rad/s) for
    torque)

    *Default* = 0.01

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`Twr_Kdmp` : Float, kg/s
    Damping factor for the tower [used only if CalcSteady=True]
    (N/(m/s))

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`Bld_Kdmp` : Float, kg/s
    Damping factor for the blades [used only if CalcSteady=True]
    (N/(m/s))

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`NLinTimes` : Integer
    Number of times to linearize (-) [>=1] [unused if Linearize=False]

    *Default* = 2

    *Minimum* = 0    *Maximum* = 10


:code:`LinTimes` : Array of Floats
    List of times at which to linearize (s) [1 to NLinTimes] [used
    only when Linearize=True and CalcSteady=False]

    *Default* = [30.0, 60.0]

    *Minimum* = 0.0

    *Maximum* = 10000.0

:code:`LinInputs` : String from, ['0', '1', '2', 'none', 'None', 'NONE', 'standard', 'Standard', 'STANDARD', 'all', 'All', 'ALL']
    Inputs included in linearization (switch) {0=none; 1=standard;
    2=all module inputs (debug)} [unused if Linearize=False]

    *Default* = 1

:code:`LinOutputs` : String from, ['0', '1', '2', 'none', 'None', 'NONE', 'standard', 'Standard', 'STANDARD', 'all', 'All', 'ALL']
    Outputs included in linearization (switch) {0=none; 1=from
    OutList(s); 2=all module outputs (debug)} [unused if
    Linearize=False]

    *Default* = 1

:code:`LinOutJac` : Boolean
    Include full Jacobians in linearization output (for debug) (flag)
    [unused if Linearize=False; used only if LinInputs=LinOutputs=2]

    *Default* = False

:code:`LinOutMod` : Boolean
    Write module-level linearization output files in addition to
    output for full system? (flag) [unused if Linearize=False]

    *Default* = False

:code:`WrVTK` : Integer
    VTK visualization data output (switch) {0=none; 1=initialization
    data only; 2=animation}

    *Default* = 0

:code:`VTK_type` : Integer
    Type of VTK visualization data (switch) {1=surfaces; 2=basic
    meshes (lines/points); 3=all meshes (debug)} [unused if WrVTK=0]

    *Default* = 2

:code:`VTK_fields` : Boolean
    Write mesh fields to VTK data files? (flag) {true/false} [unused
    if WrVTK=0]

    *Default* = False

:code:`VTK_fps` : Float
    Frame rate for VTK output (frames per second){will use closest
    integer multiple of DT} [used only if WrVTK=2]

    *Default* = 10.0

    *Minimum* = 0.0



InflowWind
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`WindType` : Integer
    Switch for wind file type (1=steady; 2=uniform; 3=binary TurbSim
    FF; 4=binary Bladed-style FF; 5=HAWC format; 6=User defined;
    7=native Bladed FF)

    *Default* = 1

:code:`PropagationDir` : Float, deg
    Direction of wind propagation (meteoroligical rotation from
    aligned with X (positive rotates towards -Y) -- degrees)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 360.0


:code:`VFlowAng` : Float, deg
    Upflow angle (degrees) (not used for native Bladed format
    WindType=7)

    *Default* = 0.0

    *Minimum* = -90.0    *Maximum* = 90.0


:code:`VelInterpCubic` : Boolean
    Use cubic interpolation for velocity in time (false=linear,
    true=cubic) [Used with WindType=2,3,4,5,7]

    *Default* = False

:code:`NWindVel` : Integer
    Number of points to output the wind velocity (0 to 9)

    *Default* = 1

    *Minimum* = 0    *Maximum* = 9


:code:`HWindSpeed` : Float, m / s
    Horizontal windspeed, for WindType = 1

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`RefHt` : Float, m
    Reference height for horizontal wind speed (m)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`PLExp` : Float
    Power law exponent (-)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`Filename_Uni` : String
    Filename of time series data for uniform wind field [used only for
    WindType = 2]

    *Default* = none

:code:`RefHt_Uni` : Float, m
    Reference height for horizontal wind speed (m)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`RefLength` : Float
    Reference length for linear horizontal and vertical sheer (-)
    [used only for WindType = 2]

    *Default* = 1.0

    *Minimum* = 1e-06    *Maximum* = 1000.0


:code:`FileName_BTS` : String
    Name of the Full field wind file to use (.bts) [used only for
    WindType = 3]

    *Default* = none

:code:`FilenameRoot` : String
    Rootname of the full-field wind file to use (.wnd, .sum) [used
    only for WindType = 4]

    *Default* = none

:code:`TowerFile` : Boolean
    Have tower file (.twr) (flag) [used only for WindType = 4]

    *Default* = False

:code:`FileName_u` : String
    Name of the file containing the u-component fluctuating wind
    (.bin) [Only used with WindType = 5]

    *Default* = none

:code:`FileName_v` : String
    Name of the file containing the v-component fluctuating wind
    (.bin) [Only used with WindType = 5]

    *Default* = none

:code:`FileName_w` : String
    Name of the file containing the w-component fluctuating wind
    (.bin) [Only used with WindType = 5]

    *Default* = none

:code:`nx` : Integer
    Number of grids in the x direction (in the 3 files above) (-)

    *Default* = 2

    *Minimum* = 2    *Maximum* = 1000


:code:`ny` : Integer
    Number of grids in the y direction (in the 3 files above) (-)

    *Default* = 2

    *Minimum* = 2    *Maximum* = 1000


:code:`nz` : Integer
    Number of grids in the z direction (in the 3 files above) (-)

    *Default* = 2

    *Minimum* = 2    *Maximum* = 1000


:code:`dx` : Float, meter
    Distance (in meters) between points in the x direction    (m)

    *Default* = 10

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`dy` : Float, meter
    Distance (in meters) between points in the y direction    (m)

    *Default* = 10

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`dz` : Float, meter
    Distance (in meters) between points in the z direction    (m)

    *Default* = 10

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`RefHt_Hawc` : Float, m
    Reference height for horizontal wind speed (m)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`ScaleMethod` : Integer
    Turbulence scaling method   [0 = none, 1 = direct scaling, 2 =
    calculate scaling factor based on a desired standard deviation]

    *Default* = 0

:code:`SFx` : Float
    Turbulence scaling factor for the x direction (-)
    [ScaleMethod=1]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SFy` : Float
    Turbulence scaling factor for the y direction (-)
    [ScaleMethod=1]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SFz` : Float
    Turbulence scaling factor for the z direction (-)
    [ScaleMethod=1]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SigmaFx` : Float, m /s
    Turbulence standard deviation to calculate scaling from in x
    direction (m/s)    [ScaleMethod=2]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SigmaFy` : Float, m /s
    Turbulence standard deviation to calculate scaling from in y
    direction (m/s)    [ScaleMethod=2]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SigmaFz` : Float, m /s
    Turbulence standard deviation to calculate scaling from in z
    direction (m/s)    [ScaleMethod=2]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`URef` : Float, m / s
    Mean u-component wind speed at the reference height (m/s) [HAWC-
    format files]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`WindProfile` : Integer
    Wind profile type (0=constant;1=logarithmic,2=power law)

    *Default* = 0

:code:`PLExp_Hawc` : Float
    Power law exponent (-) (used for PL wind profile type only)[HAWC-
    format files]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`Z0` : Float, m
    Surface roughness length (m) (used for LG wind profile type
    only)[HAWC-format files]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`XOffset` : Float, m
    Initial offset in +x direction (shift of wind box)

    *Default* = 0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`SumPrint` : Boolean
    Print summary data to '<RootName>.sum' (flag)

    *Default* = False

:code:`SensorType` : Integer
    Switch for lidar configuration (0 = None, 1 = Single Point
    Beam(s), 2 = Continuous, 3 = Pulsed)

    *Default* = 0

:code:`NumPulseGate` : Integer
    Number of lidar measurement gates (used when SensorType = 3)

    *Default* = 0

:code:`PulseSpacing` : Float
    Distance between range gates (m) (used when SensorType = 3)

    *Default* = 0

:code:`NumBeam` : Integer
    Number of lidar measurement beams (0-5)(used when SensorType = 1)

    *Default* = 0

:code:`FocalDistanceX` : Float
    Focal distance coordinates of the lidar beam in the x direction
    (relative to hub height) (only first coordinate used for
    SensorType 2 and 3) (m)

    *Default* = 0

:code:`FocalDistanceY` : Float
    Focal distance coordinates of the lidar beam in the y direction
    (relative to hub height) (only first coordinate used for
    SensorType 2 and 3) (m)

    *Default* = 0.0

:code:`FocalDistanceZ` : Float
    Focal distance coordinates of the lidar beam in the z direction
    (relative to hub height) (only first coordinate used for
    SensorType 2 and 3) (m)

    *Default* = 0.0

:code:`RotorApexOffsetPos` : Array of Floats
    Offset of the lidar from hub height (m)

    *Default* = [0.0, 0.0, 0.0]

:code:`URefLid` : Float
    Reference average wind speed for the lidar [m/s]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`MeasurementInterval` : Float
    Time between each measurement [s]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`LidRadialVel` : Boolean
    TRUE => return radial component, FALSE => return 'x' direction
    estimate

    *Default* = False

:code:`ConsiderHubMotion` : Integer
    Flag whether to consider the hub motion's impact on Lidar
    measurements

    *Default* = 1



AeroDyn
########################################

:code:`flag` : Boolean
    Whether or not to run AeroDyn

    *Default* = False

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`DTAero` : Float, s
    Time interval for aerodynamic calculations. Set it to 0. for
    default (same as main fst)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`WakeMod` : Integer
    Type of wake/induction model (switch) {0=none, 1=BEMT, 3=OLAF}

    *Default* = 1

:code:`AFAeroMod` : Integer
    Type of blade airfoil aerodynamics model (switch) {1=steady model,
    2=Beddoes-Leishman unsteady model} [must be 1 when linearizing]

    *Default* = 2

:code:`TwrPotent` : Integer
    Type tower influence on wind based on potential flow around the
    tower (switch) {0=none, 1=baseline potential flow, 2=potential
    flow with Bak correction}

    *Default* = 1

:code:`TwrShadow` : Integer
    Calculate tower influence on wind based on downstream tower shadow
    (switch) {0=none, 1=Powles model, 2=Eames model}

    *Default* = 1

:code:`TwrAero` : Boolean
    Calculate tower aerodynamic loads? (flag)

    *Default* = True

:code:`FrozenWake` : Boolean
    Assume frozen wake during linearization? (flag) [used only when
    WakeMod=1 and when linearizing]

    *Default* = False

:code:`CavitCheck` : Boolean
    Perform cavitation check? (flag) TRUE will turn off unsteady
    aerodynamics

    *Default* = False

:code:`Buoyancy` : Boolean
    Include buoyancy effects? (flag)

    *Default* = False

:code:`CompAA` : Boolean
    Flag to compute AeroAcoustics calculation [only used when
    WakeMod=1 or 2]

    *Default* = False

:code:`AA_InputFile` : String
    Aeroacoustics input file

    *Default* = AeroAcousticsInput.dat

:code:`SkewMod` : Integer
    Type of skewed-wake correction model (switch) {1=uncoupled,
    2=Pitt/Peters, 3=coupled} [used only when WakeMod=1]

    *Default* = 2

:code:`SkewModFactor` : Float
    Constant used in Pitt/Peters skewed wake model {or 'default' is
    15/32*pi} (-) [used only when SkewMod=2; unused when WakeMod=0]

    *Default* = 1.4726215563702154

:code:`TipLoss` : Boolean
    Use the Prandtl tip-loss model? (flag) [used only when WakeMod=1]

    *Default* = True

:code:`HubLoss` : Boolean
    Use the Prandtl hub-loss model? (flag) [used only when WakeMod=1]

    *Default* = True

:code:`TanInd` : Boolean
    Include tangential induction in BEMT calculations? (flag) [used
    only when WakeMod=1]

    *Default* = True

:code:`AIDrag` : Boolean
    Include the drag term in the axial-induction calculation? (flag)
    [used only when WakeMod=1]

    *Default* = True

:code:`TIDrag` : Boolean
    Include the drag term in the tangential-induction calculation?
    (flag) [used only when WakeMod=1 and TanInd=TRUE]

    *Default* = True

:code:`IndToler` : Float
    Convergence tolerance for BEMT nonlinear solve residual equation
    {or 0.0 for default} (-) [used only when WakeMod=1]

    *Default* = 0.0

:code:`MaxIter` : Integer
    Maximum number of iteration steps (-) [used only when WakeMod=1]

    *Default* = 500

:code:`DBEMT_Mod` : Integer
    Type of dynamic BEMT (DBEMT) model {1=constant tau1, 2=time-
    dependent tau1, 3=constant tau1 with continuous formulation} (-)
    [used only when WakeMod=2]

    *Default* = 2

:code:`tau1_const` : Float, s
    Time constant for DBEMT (s) [used only when WakeMod=2 and
    DBEMT_Mod=1]

    *Default* = 2.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`OLAFInputFileName` : String
    Input file for OLAF [used only when WakeMod=3]

    *Default* = unused



OLAF
========================================

:code:`IntMethod` : Integer
    Integration method 1 RK4, 5 Forward Euler 1st order, default 5
    switch

    *Default* = 5

:code:`DTfvw` : Float, s
    Time interval for wake propagation. {default dtaero} (s)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`CircSolvMethod` : Integer
    Circulation solving method {1 Cl-Based, 2 No-Flow Through, 3
    Prescribed, default 1 }(switch)

    *Default* = 1

:code:`CircSolvConvCrit` : Float
    Convergence criteria {default 0.001} [only if CircSolvMethod=1]
    (-)

    *Default* = 0.001

:code:`CircSolvRelaxation` : Float
    Relaxation factor {default 0.1} [only if CircSolvMethod=1] (-)

    *Default* = 0.1

:code:`CircSolvMaxIter` : Integer
    Maximum number of iterations for circulation solving {default 30}
    (-)

    *Default* = 30

:code:`PrescribedCircFile` : String
    File containing prescribed circulation [only if CircSolvMethod=3]
    (quoted string)

    *Default* = NA

:code:`nNWPanels` : Integer
    Number of near-wake panels [integer] (-)

    *Default* = 120

    *Minimum* = 0

:code:`nNWPanelsFree` : Integer
    Number of free near-wake panels (-) {default nNWPanels}

    *Default* = 120

    *Minimum* = 0

:code:`nFWPanels` : Integer
    Number of far-wake panels (-) {default 0}

    *Default* = 0

    *Minimum* = 0

:code:`nFWPanelsFree` : Integer
    Number of free far-wake panels (-) {default nFWPanels}

    *Default* = 0

    *Minimum* = 0

:code:`FWShedVorticity` : Boolean
    Include shed vorticity in the far wake {default false}

    *Default* = False

:code:`DiffusionMethod` : Integer
    Diffusion method to account for viscous effects {0 None, 1 Core
    Spreading, 'default' 0}

    *Default* = 0

:code:`RegDeterMethod` : Integer
    Method to determine the regularization parameters {0  Manual, 1
    Optimized, 2 chord, 3 span default 0 }

    *Default* = 0

:code:`RegFunction` : Integer
    Viscous diffusion function {0 None, 1 Rankine, 2 LambOseen, 3
    Vatistas, 4 Denominator, 'default' 3} (switch)

    *Default* = 3

:code:`WakeRegMethod` : Integer
    Wake regularization method {1 Constant, 2 Stretching, 3 Age,
    default 1} (switch)

    *Default* = 1

:code:`WakeRegFactor` : Float
    Wake regularization factor (m)

    *Default* = 0.25

:code:`WingRegFactor` : Float
    Wing regularization factor (m)

    *Default* = 0.25

:code:`CoreSpreadEddyVisc` : Float
    Eddy viscosity in core spreading methods, typical values 1-1000

    *Default* = 100

:code:`TwrShadowOnWake` : Boolean
    Include tower flow disturbance effects on wake convection
    {default:false} [only if TwrPotent or TwrShadow]

    *Default* = False

:code:`ShearModel` : Integer
    Shear Model {0 No treatment, 1 Mirrored vorticity, default 0}

    *Default* = 0

:code:`VelocityMethod` : Integer
    Method to determine the velocity {1Biot-Savart Segment, 2Particle
    tree, default 1}

    *Default* = 1

:code:`TreeBranchFactor` : Float
    Branch radius fraction above which a multipole calculation is used
    {default 2.0} [only if VelocityMethod=2]

    *Default* = 2.0

    *Minimum* = 0.0

:code:`PartPerSegment` : Integer
    Number of particles per segment [only if VelocityMethod=2]

    *Default* = 1

    *Minimum* = 0

:code:`WrVTk` : Integer
    Outputs Visualization Toolkit (VTK) (independent of .fst option)
    {0 NoVTK, 1 Write VTK at each time step} (flag)

    *Default* = 0

:code:`nVTKBlades` : Integer
    Number of blades for which VTK files are exported {0 No VTK per
    blade, n VTK for blade 1 to n} (-)

    *Default* = 3

:code:`VTKCoord` : Integer
    Coordinate system used for VTK export. {1 Global, 2 Hub, 3 Both,
    'default' 1}

    *Default* = 1

:code:`VTK_fps` : Float
    Frame rate for VTK output (frames per second) {"all" for all glue
    code timesteps, "default" for all OLAF timesteps} [used only if
    WrVTK=1]

    *Default* = 1

:code:`nGridOut` : Integer
    (GB DEBUG 7/8) Number of grid points for VTK output

    *Default* = 0

:code:`UAMod` : Integer
    Unsteady Aero Model Switch (switch) {1=Baseline model (Original),
    2=Gonzalez's variant (changes in Cn,Cc,Cm), 3=Minemma/Pierce
    variant (changes in Cc and Cm)} [used only when AFAeroMod=2]

    *Default* = 3

:code:`FLookup` : Boolean
    Flag to indicate whether a lookup for f' will be calculated (TRUE)
    or whether best-fit exponential equations will be used (FALSE); if
    FALSE S1-S4 must be provided in airfoil input files (flag) [used
    only when AFAeroMod=2]

    *Default* = True

:code:`AFTabMod` : Integer
    Interpolation method for multiple airfoil tables {1=1D
    interpolation on AoA (first table only); 2=2D interpolation on AoA
    and Re; 3=2D interpolation on AoA and UserProp} (-)

    *Default* = 1

:code:`InCol_Alfa` : Integer
    The column in the airfoil tables that contains the angle of attack
    (-)

    *Default* = 1

:code:`InCol_Cl` : Integer
    The column in the airfoil tables that contains the lift
    coefficient (-)

    *Default* = 2

:code:`InCol_Cd` : Integer
    The column in the airfoil tables that contains the drag
    coefficient (-)

    *Default* = 3

:code:`InCol_Cm` : Integer
    The column in the airfoil tables that contains the pitching-moment
    coefficient; use zero if there is no Cm column (-)

    *Default* = 4

:code:`InCol_Cpmin` : Integer
    The column in the airfoil tables that contains the Cpmin
    coefficient; use zero if there is no Cpmin column (-)

    *Default* = 0

:code:`UseBlCm` : Boolean
    Include aerodynamic pitching moment in calculations?  (flag)

    *Default* = True

:code:`VolHub` : Float
    Hub volume (m^3)

    *Default* = 0

    *Minimum* = 0.0

:code:`HubCenBx` : Float
    Hub center of buoyancy x direction offset (m)

    *Default* = 0

    *Minimum* = -100.0    *Maximum* = 100.0


:code:`VolNac` : Float
    Nacelle volume (m^3)

    *Default* = 0

    *Minimum* = 0.0

:code:`NacCenB` : Array of Floats
    Position of nacelle center of buoyancy from yaw bearing in nacelle
    coordinates (m)

    *Default* = [0.0, 0.0, 0.0]

    *Minimum* = -100.0

    *Maximum* = 100.0

:code:`TFinAero` : Boolean
    Calculate tail fin aerodynamics model (flag)

    *Default* = False

:code:`TFinFile` : String
    Input file for tail fin aerodynamics [used only when
    TFinAero=True]

    *Default* = unused

:code:`Patm` : Float
    Atmospheric pressure (Pa) [used only when CavitCheck=True]

    *Default* = 103500.0

    *Minimum* = 0.0

:code:`Pvap` : Float
    Vapour pressure of fluid (Pa) [used only when CavitCheck=True]

    *Default* = 1700.0

    *Minimum* = 0.0

:code:`FluidDepth` : Float
    Water depth above mid-hub height (m) [used only when
    CavitCheck=True]

    *Default* = 0.5

    *Minimum* = 0.0

:code:`TwrTI` : Float
    Turbulence intensity used in the Eames tower shadow model. Values
    of TwrTI between 0.05 and 0.4 are recommended.

    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`TwrCb` : Float
    Turbulence buoyancy coefficient

    *Default* = 0.0

:code:`SumPrint` : Boolean
    Print summary data to '<RootName>.sum' (flag)

    *Default* = False



ElastoDyn
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`Method` : String from, ['1', '2', '3', 'RK4', 'AB4', 'ABM4']


    *Default* = 3

:code:`DT` : Float, s
    Integration time step, 0.0 for default (s)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`FlapDOF1` : Boolean
    First flapwise blade mode DOF (flag)

    *Default* = True

:code:`FlapDOF2` : Boolean
    Second flapwise blade mode DOF (flag)

    *Default* = True

:code:`EdgeDOF` : Boolean
    First edgewise blade mode DOF (flag)

    *Default* = True

:code:`TeetDOF` : Boolean
    Rotor-teeter DOF (flag) [unused for 3 blades]

    *Default* = False

:code:`DrTrDOF` : Boolean
    Drivetrain rotational-flexibility DOF (flag)

    *Default* = True

:code:`GenDOF` : Boolean
    Generator DOF (flag)

    *Default* = True

:code:`YawDOF` : Boolean
    Yaw DOF (flag)

    *Default* = True

:code:`TwFADOF1` : Boolean
    First fore-aft tower bending-mode DOF (flag)

    *Default* = True

:code:`TwFADOF2` : Boolean
    Second fore-aft tower bending-mode DOF (flag)

    *Default* = True

:code:`TwSSDOF1` : Boolean
    First side-to-side tower bending-mode DOF (flag)

    *Default* = True

:code:`TwSSDOF2` : Boolean
    Second side-to-side tower bending-mode DOF (flag)

    *Default* = True

:code:`PtfmSgDOF` : Boolean
    Platform horizontal surge translation DOF (flag)

    *Default* = True

:code:`PtfmSwDOF` : Boolean
    Platform horizontal sway translation DOF (flag)

    *Default* = True

:code:`PtfmHvDOF` : Boolean
    Platform vertical heave translation DOF (flag)

    *Default* = True

:code:`PtfmRDOF` : Boolean
    Platform roll tilt rotation DOF (flag)

    *Default* = True

:code:`PtfmPDOF` : Boolean
    Platform pitch tilt rotation DOF (flag)

    *Default* = True

:code:`PtfmYDOF` : Boolean
    Platform yaw rotation DOF (flag)

    *Default* = True

:code:`OoPDefl` : Float, m
    Initial out-of-plane blade-tip displacement (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`IPDefl` : Float, m
    Initial in-plane blade-tip deflection (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`BlPitch1` : Float, rad
    Blade 1 initial pitch (radians)

    *Default* = 0.017453292519943295

    *Minimum* = -1.5707963267948966    *Maximum* = 1.5707963267948966


:code:`BlPitch2` : Float, rad
    Blade 2 initial pitch (radians)

    *Default* = 0.017453292519943295

    *Minimum* = -1.5707963267948966    *Maximum* = 1.5707963267948966


:code:`BlPitch3` : Float, rad
    Blade 3 initial pitch (radians) [unused for 2 blades]

    *Default* = 0.017453292519943295

    *Minimum* = -1.5707963267948966    *Maximum* = 1.5707963267948966


:code:`TeetDefl` : Float, rad
    Initial or fixed teeter angle (radians) [unused for 3 blades]

    *Default* = 0.0

    *Minimum* = -1.5707963267948966    *Maximum* = 1.5707963267948966


:code:`Azimuth` : Float, rad
    Initial azimuth angle for blade 1 (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`RotSpeed` : Float, rpm
    Initial or fixed rotor speed (rpm)

    *Default* = 5.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`NacYaw` : Float, rad
    Initial or fixed nacelle-yaw angle (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`TTDspFA` : Float, m
    Initial fore-aft tower-top displacement (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 50.0


:code:`TTDspSS` : Float, m
    Initial side-to-side tower-top displacement (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 50.0


:code:`PtfmSurge` : Float, m
    Initial or fixed horizontal surge translational displacement of
    platform (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`PtfmSway` : Float, m
    Initial or fixed horizontal sway translational displacement of
    platform (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`PtfmHeave` : Float, m
    Initial or fixed vertical heave translational displacement of
    platform (meters)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`PtfmRoll` : Float, rad
    Initial or fixed roll tilt rotational displacement of platform
    (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`PtfmPitch` : Float, rad
    Initial or fixed pitch tilt rotational displacement of platform
    (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`PtfmYaw` : Float, rad
    Initial or fixed yaw rotational displacement of platform (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`UndSling` : Float, m
    Undersling length [distance from teeter pin to the rotor apex]
    (meters) [unused for 3 blades]

    *Default* = 0.0

    *Minimum* = -10.0    *Maximum* = 10.0


:code:`Delta3` : Float, deg
    Delta-3 angle for teetering rotors (degrees) [unused for 3 blades]

    *Default* = 0.0

    *Minimum* = -30.0    *Maximum* = 30.0


:code:`AzimB1Up` : Float, rad
    Azimuth value to use for I/O when blade 1 points up (radians)

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`ShftGagL` : Float, m
    Distance from rotor apex [3 blades] or teeter pin [2 blades] to
    shaft strain gages [positive for upwind rotors] (meters)

    *Default* = 0.0

    *Minimum* = -10.0    *Maximum* = 10.0


:code:`NcIMUxn` : Float, m
    Downwind distance from the tower-top to the nacelle IMU (meters)

    *Default* = 0.0

    *Minimum* = -10.0    *Maximum* = 10.0


:code:`NcIMUyn` : Float, m
    Lateral distance from the tower-top to the nacelle IMU (meters)

    *Default* = 0.0

    *Minimum* = -10.0    *Maximum* = 10.0


:code:`NcIMUzn` : Float, m
    Vertical distance from the tower-top to the nacelle IMU (meters)

    *Default* = 0.0

    *Minimum* = -10.0    *Maximum* = 10.0


:code:`BldNodes` : Integer
    Number of blade nodes (per blade) used for analysis (-)

    *Default* = 50

    *Minimum* = 10    *Maximum* = 200


:code:`TeetMod` : Integer
    Rotor-teeter spring/damper model {0: none, 1: standard, 2: user-
    defined from routine UserTeet} (switch) [unused for 3 blades]

    *Default* = 0

:code:`TeetDmpP` : Float, rad
    Rotor-teeter damper position (radians) [used only for 2 blades and
    when TeetMod=1]

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`TeetDmp` : Float, kg*m^2/rad/s
    Rotor-teeter damping constant (N-m/(rad/s)) [used only for 2
    blades and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`TeetCDmp` : Float, kg*m^2/s^2
    Rotor-teeter rate-independent Coulomb-damping moment (N-m) [used
    only for 2 blades and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`TeetSStP` : Float, rad
    Rotor-teeter soft-stop position (radians) [used only for 2 blades
    and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`TeetHStP` : Float, rad
    Rotor-teeter hard-stop position (radians) [used only for 2 blades
    and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = -6.283185307179586    *Maximum* = 6.283185307179586


:code:`TeetSSSp` : Float, kg*m^2/rad/s^2
    Rotor-teeter soft-stop linear-spring constant (N-m/rad) [used only
    for 2 blades and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`TeetHSSp` : Float, kg*m^2/rad/s^2
    Rotor-teeter hard-stop linear-spring constant (N-m/rad) [used only
    for 2 blades and when TeetMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`Furling` : Boolean
    Read in additional model properties for furling turbine (flag)
    [must currently be FALSE)

    *Default* = False

:code:`FurlFile` : String
    Name of file containing furling properties (quoted string) [unused
    when Furling=False]

    *Default* = none

:code:`TwrNodes` : Integer
    Number of tower nodes used for analysis (-)

    *Default* = 20

    *Minimum* = 10    *Maximum* = 200


:code:`SumPrint` : Boolean
    Print summary data to '<RootName>.sum' (flag)

    *Default* = False

:code:`OutFile` : Integer
    Switch to determine where output will be placed 1 in module output
    file only; 2 in glue code output file only; 3 both (currently
    unused)

    *Default* = 1

:code:`TabDelim` : Boolean
    Use tab delimiters in text tabular output file? (flag) (currently
    unused)

    *Default* = True

:code:`OutFmt` : String
    Format used for text tabular output (except time).  Resulting
    field should be 10 characters. (quoted string (currently unused)

    *Default* = ES10.3E2

:code:`DecFact` : Integer
    Decimation factor for tabular output 1 output every time step} (-)
    (currently unused)

    *Default* = 1

:code:`TStart` : Float, s
    Time to begin tabular output (s) (currently unused)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0




ElastoDynBlade
########################################

:code:`BldFlDmp1` : Float
    Blade flap mode 1 structural damping in percent of critical (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`BldFlDmp2` : Float
    Blade flap mode 2 structural damping in percent of critical (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`BldEdDmp1` : Float
    Blade edge mode 1 structural damping in percent of critical (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`FlStTunr1` : Float
    Blade flapwise modal stiffness tuner, 1st mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`FlStTunr2` : Float
    Blade flapwise modal stiffness tuner, 2nd mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjBlMs` : Float
    Factor to adjust blade mass density (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjFlSt` : Float
    Factor to adjust blade flap stiffness (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjEdSt` : Float
    Factor to adjust blade edge stiffness (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0




ElastoDynTower
########################################

:code:`TwrFADmp1` : Float
    Tower 1st fore-aft mode structural damping ratio (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`TwrFADmp2` : Float
    Tower 2nd fore-aft mode structural damping ratio (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`TwrSSDmp1` : Float
    Tower 1st side-to-side mode structural damping ratio (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`TwrSSDmp2` : Float
    Tower 2nd side-to-side mode structural damping ratio (%)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`FlStTunr1` : Float
    Blade flapwise modal stiffness tuner, 1st mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`FAStTunr1` : Float
    Tower fore-aft modal stiffness tuner, 1st mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`FAStTunr2` : Float
    Tower fore-aft modal stiffness tuner, 2nd mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SSStTunr1` : Float
    Tower side-to-side stiffness tuner, 1st mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SSStTunr2` : Float
    Tower side-to-side stiffness tuner, 2nd mode (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjTwMa` : Float
    Factor to adjust tower mass density (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjFASt` : Float
    Factor to adjust tower fore-aft stiffness (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`AdjSSSt` : Float
    Factor to adjust tower side-to-side stiffness (-)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0




BeamDyn
########################################

:code:`QuasiStaticInit` : Boolean
    Use quasistatic pre-conditioning with centripetal accelerations in
    initialization (flag) [dynamic solve only]

    *Default* = True

:code:`rhoinf` : Float
    Numerical damping parameter for generalized-alpha integrator

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000000000.0


:code:`quadrature` : String from, ['1', '2', 'gaussian', 'Gaussian', 'GAUSSIAN', 'trapezoidal', 'Trapezoidal', 'TRAPEZOIDAL']
    Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)

    *Default* = 2

:code:`refine` : Integer
    Refinement factor for trapezoidal quadrature (-). DEFAULT = 1
    [used only when quadrature=2]

    *Default* = 1

    *Minimum* = 1    *Maximum* = 10


:code:`n_fact` : Integer
    Factorization frequency (-). DEFAULT = 5

    *Default* = 5

    *Minimum* = 1    *Maximum* = 50


:code:`DTBeam` : Float, s
    Time step size (s). Use 0.0 for Default

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`load_retries` : Integer
    Number of factored load retries before quitting the simulation.
    Use 0 for Default

    *Default* = 0

    *Minimum* = 0    *Maximum* = 50


:code:`NRMax` : Integer
    Max number of iterations in Newton-Ralphson algorithm (-). DEFAULT
    = 10

    *Default* = 10

    *Minimum* = 1    *Maximum* = 100


:code:`stop_tol` : Float
    Tolerance for stopping criterion (-)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1e+16


:code:`tngt_stf_fd` : Boolean
    Flag to use finite differenced tangent stiffness matrix (-)

    *Default* = False

:code:`tngt_stf_comp` : Boolean
    Flag to compare analytical finite differenced tangent stiffness
    matrix  (-)

    *Default* = False

:code:`tngt_stf_pert` : Float
    perturbation size for finite differencing (-).  Use 0.0 for
    DEFAULT

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`tngt_stf_difftol` : Float
    Maximum allowable relative difference between analytical and fd
    tangent stiffness (-)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`RotStates` : Boolean
    Orient states in the rotating frame during linearization? (flag)
    [used only when linearizing]

    *Default* = True

:code:`order_elem` : Integer
    Order of interpolation (basis) function (-)

    *Default* = 10

    *Minimum* = 0    *Maximum* = 50


:code:`UsePitchAct` : Boolean
    Whether a pitch actuator should be used (flag)

    *Default* = False

:code:`PitchJ` : Float, kg*m^2
    Pitch actuator inertia (kg-m^2) [used only when UsePitchAct is
    true]

    *Default* = 200.0

    *Minimum* = 0.0    *Maximum* = 1000000000000.0


:code:`PitchK` : Float, kg*m^2/s^2
    Pitch actuator stiffness (kg-m^2/s^2) [used only when UsePitchAct
    is true]

    *Default* = 20000000.0

    *Minimum* = 0.0    *Maximum* = 1000000000000.0


:code:`PitchC` : Float, kg*m^2/s
    Pitch actuator damping (kg-m^2/s) [used only when UsePitchAct is
    true]

    *Default* = 500000.0

    *Minimum* = 0.0    *Maximum* = 1000000000000.0




HydroDyn
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`WaveMod` : Integer
    Incident wave kinematics model {0- none/still water, 1- regular
    (periodic), 1P#- regular with user-specified phase, 2-
    JONSWAP/Pierson-Moskowitz spectrum (irregular), 3- White noise
    spectrum (irregular), 4- user-defined spectrum from routine
    UserWaveSpctrm (irregular), 5- Externally generated wave-elevation
    time series, 6- Externally generated full wave-kinematics time
    series [option 6 is invalid for PotMod/=0]} (switch)

    *Default* = 2

:code:`WaveStMod` : Integer
    Model for stretching incident wave kinematics to instantaneous
    free surface {0 = none=no stretching, 1 = vertical stretching, 2 =
    extrapolation stretching, 3 = Wheeler stretching} (switch) [unused
    when WaveMod=0 or when PotMod/=0]

    *Default* = 0

:code:`WaveTMax` : Float, s
    Analysis time for incident wave calculations (sec) [unused when
    WaveMod=0; determines WaveDOmega=2Pi/WaveTMax in the IFFT]

    *Default* = 3600

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`WaveDT` : Float, s
    Time step for incident wave calculations     (sec) [unused when
    WaveMod=0; 0.1<=WaveDT<=1.0 recommended; determines
    WaveOmegaMax=Pi/WaveDT in the IFFT]

    *Default* = 0.25

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`WavePkShp` : Float
    Peak-shape parameter of incident wave spectrum (-) or DEFAULT
    (string) [used only when WaveMod=2; use 1.0 for Pierson-Moskowitz]

    *Default* = 1.0

    *Minimum* = 1    *Maximum* = 7


:code:`WvLowCOff` : Float, rad/s
    Low cut-off frequency or lower frequency limit of the wave
    spectrum beyond which the wave spectrum is zeroed (rad/s) [unused
    when WaveMod=0, 1, or 6]

    *Default* = 0.111527

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`WvHiCOff` : Float, rad/s
    High cut-off frequency or upper frequency limit of the wave
    spectrum beyond which the wave spectrum is zeroed (rad/s) [unused
    when WaveMod=0, 1, or 6]

    *Default* = 0.783827

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`WaveDir` : Float, rad
    Incident wave propagation heading direction [unused when WaveMod=0
    or 6]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 6.283185307179586


:code:`WaveDirMod` : Integer
    Directional spreading function {0 = none, 1 = COS2S} [only used
    when WaveMod=2,3, or 4]

    *Default* = 0

:code:`WaveDirSpread` : Float
    Wave direction spreading coefficient ( > 0 ) [only used when
    WaveMod=2,3, or 4 and WaveDirMod=1]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`WaveNDir` : Integer
    Number of wave directions [only used when WaveMod=2,3, or 4 and
    WaveDirMod=1; odd number only]

    *Default* = 1

:code:`WaveDirRange` : Float, deg
    Range of wave directions (full range = WaveDir +/-
    1/2*WaveDirRange) (degrees) [only used when WaveMod=2,3,or 4 and
    WaveDirMod=1]

    *Default* = 90

    *Minimum* = 0.0    *Maximum* = 360


:code:`WaveSeed1` : Integer
    First random seed of incident waves [-2147483648 to 2147483647]
    [unused when WaveMod=0, 5, or 6]

    *Default* = -561580799

    *Minimum* = -2147483648    *Maximum* = 2147483647


:code:`WaveNDAmp` : Boolean
    Flag for normally distributed amplitudes [only used when
    WaveMod=2, 3, or 4]

    *Default* = True

:code:`WvKinFile` : String
    Root name of externally generated wave data file(s) (quoted
    string) [used only when WaveMod=5 or 6]

    *Default* = 

:code:`NWaveElev` : Integer
    Number of points where the incident wave elevations can be
    computed (-) [maximum of 9 output locations]

    *Default* = 1

    *Minimum* = 0    *Maximum* = 9


:code:`WaveElevxi` : Array of Strings
    List of xi-coordinates for points where the incident wave
    elevations can be output (meters) [NWaveElev points, separated by
    commas or white space; usused if NWaveElev = 0]

    *Default* = ['0.0']

:code:`WaveElevyi` : Array of Strings
    List of yi-coordinates for points where the incident wave
    elevations can be output (meters) [NWaveElev points, separated by
    commas or white space; usused if NWaveElev = 0]

    *Default* = ['0.0']

:code:`WvDiffQTF` : Boolean
    Full difference-frequency 2nd-order wave kinematics (flag)

    *Default* = False

:code:`WvSumQTF` : Boolean
    Full summation-frequency  2nd-order wave kinematics (flag)

    *Default* = False

:code:`WvLowCOffD` : Float, rad/s
    Low frequency cutoff used in the difference-frequencies (rad/s)
    [Only used with a difference-frequency method]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`WvHiCOffD` : Float, rad/s
    High frequency cutoff used in the difference-frequencies (rad/s)
    [Only used with a difference-frequency method]

    *Default* = 0.737863

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`WvLowCOffS` : Float, rad/s
    Low frequency cutoff used in the summation-frequencies  (rad/s)
    [Only used with a summation-frequency method]

    *Default* = 0.314159

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`WvHiCOffS` : Float, rad/s
    High frequency cutoff used in the summation-frequencies  (rad/s)
    [Only used with a summation-frequency method]

    *Default* = 3.2

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`CurrMod` : Integer
    Current profile model {0 = none=no current, 1 = standard, 2 =
    user-defined from routine UserCurrent} (switch)

    *Default* = 0

:code:`CurrSSV0` : Float, m/s
    Sub-surface current velocity at still water level  (m/s) [used
    only when CurrMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`CurrSSDir` : Float, rad
    Sub-surface current heading direction (radians) or 0.0 for default
    [used only when CurrMod=1]

    *Default* = 0    *Maximum* = 6.283185307179586


:code:`CurrNSRef` : Float, m
    Near-surface current reference depth (meters) [used only when
    CurrMod=1]

    *Default* = 20.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`CurrNSV0` : Float, m/s
    Near-surface current velocity at still water level (m/s) [used
    only when CurrMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`CurrNSDir` : Float, rad
    Near-surface current heading direction (degrees) [used only when
    CurrMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 6.283185307179586


:code:`CurrDIV` : Float, m/s
    Depth-independent current velocity (m/s) [used only when
    CurrMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`CurrDIDir` : Float, rad
    Depth-independent current heading direction (radians) [used only
    when CurrMod=1]

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 6.283185307179586


:code:`PotMod` : Integer
    Potential-flow model {0 = none=no potential flow, 1 = frequency-
    to-time-domain transforms based on Capytaine/NEMOH/WAMIT output, 2
    = fluid-impulse theory (FIT)} (switch)

    *Default* = 0

:code:`PotFile` : String
    Will be automatically filled in with HAMS output unless a value
    here overrides it; WAMIT output files containing the linear,
    nondimensionalized, hydrostatic restoring matrix (.hst),
    frequency-dependent hydrodynamic added mass matrix and damping
    matrix (.1), and frequency- and direction-dependent wave
    excitation force vector per unit wave amplitude (.3) (quoted
    string) [MAKE SURE THE FREQUENCIES INHERENT IN THESE WAMIT FILES
    SPAN THE PHYSICALLY-SIGNIFICANT RANGE OF FREQUENCIES FOR THE GIVEN
    PLATFORM; THEY MUST CONTAIN THE ZERO- AND INFINITE-FREQUENCY
    LIMITS]

    *Default* = unused

:code:`WAMITULEN` : Float, m
    Characteristic body length scale used to redimensionalize
    Capytaine/NEMOH/WAMIT output (meters) [only used when PotMod=1]

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`PtfmMass_Init` : Float, kg
    Mass of initial platform design. When PtfmMass_Init > 0, PtfmVol0
    will scale with the platform mass; this is a temporary solution to
    enable spar simulations where the heave is very sensitive to
    platform mass.

    *Default* = 0.0

    *Minimum* = 0.0

:code:`PtfmCOBxt` : Float, m
    The xt offset of the center of buoyancy (COB) from the platform
    reference point (meters) [only used when PotMod=1]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`PtfmCOByt` : Float, m
    The yt offset of the center of buoyancy (COB) from the platform
    reference point (meters) [only used when PotMod=1]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`ExctnMod` : Integer
    Wave Excitation model {0 = None, 1 = DFT, 2 = state-space}
    (switch) [only used when PotMod=1; STATE-SPACE REQUIRES *.ssexctn
    INPUT FILE]

    *Default* = 0

:code:`RdtnMod` : Integer
    Radiation memory-effect model {0 = no memory-effect calculation, 1
    = convolution, 2 = state-space} (switch) [only used when PotMod=1;
    STATE-SPACE REQUIRES *.ss INPUT FILE]

    *Default* = 0

:code:`RdtnTMax` : Float, s
    Analysis time for wave radiation kernel calculations (sec) [only
    used when PotMod=1; determines RdtnDOmega=Pi/RdtnTMax in the
    cosine transform; MAKE SURE THIS IS LONG ENOUGH FOR THE RADIATION
    IMPULSE RESPONSE FUNCTIONS TO DECAY TO NEAR-ZERO FOR THE GIVEN
    PLATFORM!]

    *Default* = 60.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`RdtnDT` : Float, s
    Time step for wave radiation kernel calculations, use 0.0 for
    default (sec) [only used when PotMod=1; DT<=RdtnDT<=0.1
    recommended; determines RdtnOmegaMax=Pi/RdtnDT in the cosine
    transform]

    *Default* = 0.0125

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`MnDrift` : Integer
    Mean-drift 2nd-order forces computed {0 = None; [7, 8, 9, 10, 11,
    or 12] = WAMIT file to use} [Only one of MnDrift, NewmanApp, or
    DiffQTF can be non-zero]

    *Default* = 0

:code:`NewmanApp` : Integer
    Mean- and slow-drift 2nd-order forces computed with Newman's
    approximation {0 = None; [7, 8, 9, 10, 11, or 12] = WAMIT file to
    use} [Only one of MnDrift, NewmanApp, or DiffQTF can be non-zero.
    Used only when WaveDirMod=0]

    *Default* = 0

:code:`DiffQTF` : Integer
    Full difference-frequency 2nd-order forces computed with full QTF
    {0 = None; [10, 11, or 12] = WAMIT file to use} [Only one of
    MnDrift, NewmanApp, or DiffQTF can be non-zero]

    *Default* = 0

:code:`SumQTF` : Integer
    Full summation -frequency 2nd-order forces computed with full QTF
    {0 = None; [10, 11, or 12] = WAMIT file to use}

    *Default* = 0

:code:`AddF0` : Array of Floats
    Additional preload (N, N-m)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin1` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin2` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin3` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin4` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin5` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddCLin6` : Array of Floats
    Additional linear stiffness by row (N/m, N/rad, N-m/m, N-m/rad)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin1` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin2` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin3` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin4` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin5` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBLin6` : Array of Floats
    Additional linear damping by row (N/(m/s), N/(rad/s), N-m/(m/s),
    N-m/(rad/s))

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad1` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad2` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad3` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad4` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad5` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`AddBQuad6` : Array of Floats
    Additional quadratic drag by row (N/(m/s)^2, N/(rad/s)^2,
    N-m(m/s)^2, N-m/(rad/s)^2)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`NMOutputs` : Integer
    Number of member outputs (-) [must be < 10]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 9


:code:`NJOutputs` : Integer
    Number of joint outputs [Must be < 10]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 9


:code:`JOutLst` : Array of Integers
    List of JointIDs which are to be output (-)[unused if NJOutputs=0]

    *Default* = [0]

:code:`HDSum` : Boolean
    Output a summary file [flag]

    *Default* = True

:code:`OutAll` : Boolean
    Output all user-specified member and joint loads (only at each
    member end, not interior locations) [flag]

    *Default* = False

:code:`OutSwtch` : Integer
    Output requested channels to [1=Hydrodyn.out, 2=GlueCode.out,
    3=both files]

    *Default* = 2

:code:`OutFmt` : String
    Output format for numerical results (quoted string) [not checked
    for validity]

    *Default* = ES11.4e2

:code:`OutSFmt` : String
    Output format for header strings (quoted string) [not checked for
    validity]

    *Default* = A11

:code:`NBody` : Integer
    Number of WAMIT bodies to be used (-) [>=1; only used when
    PotMod=1. If NBodyMod=1, the WAMIT data contains a vector of size
    6*NBody x 1 and matrices of size 6*NBody x 6*NBody; if NBodyMod>1,
    there are NBody sets of WAMIT data each with a vector of size 6 x
    1 and matrices of size 6 x 6]

    *Default* = 1

    *Minimum* = 1    *Maximum* = 9


:code:`NBodyMod` : Integer
    Body coupling model {1- include coupling terms between each body
    and NBody in HydroDyn equals NBODY in WAMIT, 2- neglect coupling
    terms between each body and NBODY=1 with XBODY=0 in WAMIT, 3-
    Neglect coupling terms between each body and NBODY=1 with XBODY=/0
    in WAMIT} (switch) [only used when PotMod=1]

    *Default* = 1

    *Minimum* = 1    *Maximum* = 3


:code:`SimplCd` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplCa` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplCp` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplCdMG` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplCaMG` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplCpMG` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCd` : Float
    Simple strip theory model coefficient, default of 0.0

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCa` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCp` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCdMG` : Float
    Simple strip theory model coefficient, default of 0.0

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCaMG` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`SimplAxCpMG` : Float
    Simple strip theory model coefficient, default of 1.0

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0




SubDyn
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`SDdeltaT` : Float, s
    Local Integration Step. If 0.0, the glue-code integration step
    will be used.

    *Default* = -999.0    *Maximum* = 100.0


:code:`IntMethod` : Integer
    Integration Method [1/2/3/4 = RK4/AB4/ABM4/AM2].

    *Default* = 3

:code:`SttcSolve` : Boolean
    Solve dynamics about static equilibrium point

    *Default* = True

:code:`GuyanLoadCorrection` : Boolean
    Include extra moment from lever arm at interface and rotate FEM
    for floating.

    *Default* = False

:code:`FEMMod` : Integer
    FEM switch = element model in the FEM. [1= Euler-Bernoulli(E-B);
    2=Tapered E-B (unavailable);  3= 2-node Timoshenko;  4= 2-node
    tapered Timoshenko (unavailable)]

    *Default* = 3

:code:`NDiv` : Integer
    Number of sub-elements per member

    *Default* = 1

    *Minimum* = 1    *Maximum* = 100


:code:`CBMod` : Boolean
    If True perform C-B reduction, else full FEM dofs will be
    retained. If True, select Nmodes to retain in C-B reduced system.

    *Default* = True

:code:`Nmodes` : Integer
    Number of internal modes to retain (ignored if CBMod=False). If
    Nmodes=0 --> Guyan Reduction.

    *Default* = 0

    *Minimum* = 0    *Maximum* = 50


:code:`JDampings` : Array of Floats
    Damping Ratios for each retained mode (% of critical) If Nmodes>0,
    list Nmodes structural damping ratios for each retained mode (% of
    critical), or a single damping ratio to be applied to all retained
    modes. (last entered value will be used for all remaining modes).

    *Default* = [1.0]

:code:`GuyanDampMod` : Integer
    Guyan damping {0=none, 1=Rayleigh Damping, 2=user specified 6x6
    matrix}

    *Default* = 0

:code:`RayleighDamp` : Array of Floats
    Mass and stiffness proportional damping  coefficients (Rayleigh
    Damping) [only if GuyanDampMod=1]

    *Default* = [0.0, 0.0]

:code:`GuyanDampSize` : Integer
    Guyan damping matrix (6x6) [only if GuyanDampMod=2]

    *Default* = 6

    *Minimum* = 0    *Maximum* = 6


:code:`GuyanDamp1` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`GuyanDamp2` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`GuyanDamp3` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`GuyanDamp4` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`GuyanDamp5` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`GuyanDamp6` : Array of Floats
    Guyan damping matrix by row (6x6)

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

:code:`SumPrint` : Boolean
    Output a Summary File (flag) that contains matrices K,M  and C-B
    reduced M_BB, M-BM, K_BB, K_MM(OMG^2), PHI_R, PHI_L. It can also
    contain COSMs if requested.

    *Default* = False

:code:`OutCOSM` : Boolean
    Output cosine matrices with the selected output member forces
    (flag)

    *Default* = False

:code:`OutAll` : Boolean
    Output all members' end forces (flag)

    *Default* = False

:code:`OutSwtch` : Integer
    Output requested channels to 1=<rootname>.SD.out;
    2=<rootname>.out (generated by FAST);  3=both files.

    *Default* = 2

:code:`TabDelim` : Boolean
    Generate a tab-delimited output in the <rootname>.SD.out file

    *Default* = True

:code:`OutDec` : Integer
    Decimation of output in the <rootname>.SD.out file

    *Default* = 1

    *Minimum* = 0

:code:`OutFmt` : String
    Output format for numerical results in the <rootname>.SD.out file
    (quoted string) [not checked for validity]

    *Default* = ES11.4e2

:code:`OutSFmt` : String
    Output format for header strings in the <rootname>.SD.out file
    (quoted string) [not checked for validity]

    *Default* = A11

:code:`NMOutputs` : Integer
    Number of members whose
    forces/displacements/velocities/accelerations will be output (-)
    [Must be <= 9].

    *Default* = 0

    *Minimum* = 0    *Maximum* = 9




MoorDyn
########################################

:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`dtM` : Float, s
    Time step to use in mooring integration (s)

    *Default* = 0.001

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`kbot` : Float, kg/(m^2*s^2)
    Bottom stiffness (Pa/m)

    *Default* = 3000000.0

    *Minimum* = 0.0    *Maximum* = 1000000000.0


:code:`cbot` : Float, kg/(m^2*s)
    Bottom damping (Pa/m)

    *Default* = 300000.0

    *Minimum* = 0.0    *Maximum* = 1000000000.0


:code:`dtIC` : Float, s
    Time interval for analyzing convergence during IC gen (s)

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`TmaxIC` : Float, s
    Max time for ic gen (s)

    *Default* = 60.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`CdScaleIC` : Float
    Factor by which to scale drag coefficients during dynamic
    relaxation (-)

    *Default* = 4.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`threshIC` : Float
    Threshold for IC convergence (-)

    *Default* = 0.001

    *Minimum* = 0.0    *Maximum* = 1.0




ServoDyn
########################################

ServoDyn modelling options in OpenFAST
:code:`Echo` : Boolean
    Echo input data to '<RootName>.ech' (flag)

    *Default* = False

:code:`DT` : String
    Communication interval for controllers (s) (or 'default')

    *Default* = default

:code:`PCMode` : Integer
    Pitch control mode {0 = none, 4 = user-defined from
    Simulink/Labview, 5 = user-defined from Bladed-style DLL}

    *Default* = 5

:code:`TPCOn` : Float, s
    Time to enable active pitch control (s) [unused when PCMode=0]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`TPitManS1` : Float, s
    Time to start override pitch maneuver for blade 1 and end standard
    pitch control (s)

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TPitManS2` : Float, s
    Time to start override pitch maneuver for blade 2 and end standard
    pitch control (s)

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TPitManS3` : Float, s
    Time to start override pitch maneuver for blade 3 and end standard
    pitch control (s)

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`PitManRat(1)` : Float, deg / s
    Pitch rate at which override pitch maneuver heads toward final
    pitch angle for blade 1 (deg/s). It cannot be 0

    *Default* = 1.0

    *Minimum* = 1e-06    *Maximum* = 30.0


:code:`PitManRat(2)` : Float, deg / s
    Pitch rate at which override pitch maneuver heads toward final
    pitch angle for blade 2 (deg/s). It cannot be 0

    *Default* = 1.0

    *Minimum* = 1e-06    *Maximum* = 30.0


:code:`PitManRat(3)` : Float, deg / s
    Pitch rate at which override pitch maneuver heads toward final
    pitch angle for blade 3 (deg/s). It cannot be 0

    *Default* = 1.0

    *Minimum* = 1e-06    *Maximum* = 30.0


:code:`BlPitchF(1)` : Float, deg
    Blade 1 final pitch for pitch maneuvers (degrees)

    *Default* = 90.0

    *Minimum* = -180    *Maximum* = 180


:code:`BlPitchF(2)` : Float, deg
    Blade 2 final pitch for pitch maneuvers (degrees)

    *Default* = 90.0

    *Minimum* = -180    *Maximum* = 180


:code:`BlPitchF(3)` : Float, deg
    Blade 3 final pitch for pitch maneuvers (degrees)

    *Default* = 90.0

    *Minimum* = -180    *Maximum* = 180


:code:`VSContrl` : Integer
    Variable-speed control mode {0 = none, 4 = user-defined from
    Simulink/Labview, 5 = user-defined from Bladed-style DLL}

    *Default* = 5

:code:`GenModel` : Integer
    Generator model {1 = simple, 2 = Thevenin, 3 = user-defined from
    routine UserGen}

    *Default* = 1

:code:`GenTiStr` : Boolean
    Method to start the generator {True - timed using TimGenOn, False
    - generator speed using SpdGenOn} (flag)

    *Default* = True

:code:`GenTiStp` : Boolean
    Method to stop the generator {True - timed using TimGenOf, False -
    when generator power = 0} (flag)

    *Default* = True

:code:`SpdGenOn` : Float, rpm
    Generator speed to turn on the generator for a startup (HSS speed)
    (rpm) [used only when GenTiStr=False]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TimGenOn` : Float, s
    Time to turn on the generator for a startup (s) [used only when
    GenTiStr=True]

    *Default* = 0.0

    *Minimum* = 0.0

:code:`TimGenOf` : Float, s
    Time to turn off the generator (s) [used only when GenTiStp=True]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`VS_RtGnSp` : Float, rpm
    Rated generator speed for simple variable-speed generator control
    (HSS side) (rpm) [used only when VSContrl=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`VS_RtTq` : Float, N * m
    Rated generator torque/constant generator torque in Region 3 for
    simple variable-speed generator control (HSS side) (N-m) [used
    only when VSContrl=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`VS_Rgn2K` : Float, N * m / rpm**2
    Generator torque constant in Region 2 for simple variable-speed
    generator control (HSS side) (N-m/rpm^2) [used only when
    VSContrl=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`VS_SlPc` : Float
    Rated generator slip percentage in Region 2 1/2 for simple
    variable-speed generator control (%) [used only when VSContrl=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`SIG_SlPc` : Float
    Rated generator slip percentage (%) [used only when VSContrl=0 and
    GenModel=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`SIG_SySp` : Float, rpm
    Synchronous (zero-torque) generator speed (rpm) [used only when
    VSContrl=0 and GenModel=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`SIG_RtTq` : Float, N * m
    Rated torque (N-m) [used only when VSContrl=0 and GenModel=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`SIG_PORt` : Float
    Pull-out ratio (Tpullout/Trated) (-) [used only when VSContrl=0
    and GenModel=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_Freq` : Float, Hz
    Line frequency [50 or 60] (Hz) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_NPol` : Integer
    Number of poles [even integer > 0] (-) [used only when VSContrl=0
    and GenModel=2]

    *Default* = 0

    *Minimum* = 0

:code:`TEC_SRes` : Float, ohms
    Stator resistance (ohms) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_RRes` : Float, ohms
    Rotor resistance (ohms) [used only when VSContrl=0 and GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_VLL` : Float, volts
    Line-to-line RMS voltage (volts) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_SLR` : Float, ohms
    Stator leakage reactance (ohms) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_RLR` : Float, ohms
    Rotor leakage reactance (ohms) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`TEC_MR` : Float, ohms
    Magnetizing reactance (ohms) [used only when VSContrl=0 and
    GenModel=2]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`HSSBrMode` : Integer
    HSS brake model {0 = none, 1 = simple, 4 = user-defined from
    Simulink/Labview, 5 = user-defined from Bladed-style DLL (not in
    ROSCO, yet)}

    *Default* = 0

:code:`THSSBrDp` : Float, s
    Time to initiate deployment of the HSS brake (s)

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`HSSBrDT` : Float, s
    Time for HSS-brake to reach full deployment once initiated (sec)
    [used only when HSSBrMode=1]

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`HSSBrTqF` : Float, N * m
    Fully deployed HSS-brake torque (N-m)

    *Default* = 99999.0

    *Minimum* = 0.0

:code:`YCMode` : Integer
    Yaw control mode {0 - none, 3 - user-defined from routine
    UserYawCont, 4 - user-defined from Simulink/Labview, 5 - user-
    defined from Bladed-style DLL} (switch)

    *Default* = 0

:code:`TYCOn` : Float, s
    Time to enable active yaw control (s) [unused when YCMode=0]

    *Default* = 99999.0

:code:`YawNeut` : Float, deg
    Neutral yaw position--yaw spring force is zero at this yaw
    (degrees)

    *Default* = 0.0

:code:`YawSpr` : Float, N * m / rad
    Nacelle-yaw spring constant (N-m/rad)

    *Default* = 0.0

:code:`YawDamp` : Float, N * m / rad / s
    Nacelle-yaw damping constant (N-m/(rad/s))

    *Default* = 0.0

:code:`TYawManS` : Float, s
    Time to start override yaw maneuver and end standard yaw control
    (s)

    *Default* = 99999.0

:code:`YawManRat` : Float, deg / s
    Yaw maneuver rate (in absolute value) (deg/s). It cannot be zero

    *Default* = 0.25

    *Minimum* = 1e-06

:code:`NacYawF` : Float, deg
    Final yaw angle for override yaw maneuvers (degrees)

    *Default* = 0.0

:code:`AfCmode` : Integer
    Airfoil control mode {0- none, 1- cosine wave cycle, 4- user-
    defined from Simulink/Labview, 5- user-defined from Bladed-style
    DLL}

    *Default* = 0

:code:`AfC_Mean` : Float, deg
    Mean level for sinusoidal cycling or steady value (-) [used only
    with AfCmode==1]

    *Default* = 0.0

:code:`AfC_Amp` : Float, deg
    Amplitude for for cosine cycling of flap signal (AfC =
    AfC_Amp*cos(Azimuth+phase)+AfC_mean) (-) [used only with
    AfCmode==1]

    *Default* = 0.0

:code:`AfC_Phase` : Float, deg
    AfC_phase - Phase relative to the blade azimuth (0 is vertical)
    for for cosine cycling of flap signal (deg) [used only with
    AfCmode==1]

    *Default* = 0.0

:code:`CCmode` : Integer
    Cable control mode {0- none, 4- user-defined from
    Simulink/Labview, 5- user-defineAfC_phased from Bladed-style DLL}

    *Default* = 0

:code:`CompNTMD` : Boolean
    Compute nacelle tuned mass damper {true/false}

    *Default* = False

:code:`NTMDfile` : String
    Name of the file for nacelle tuned mass damper (quoted string)
    [unused when CompNTMD is false]

    *Default* = none

:code:`CompTTMD` : Boolean
    Compute tower tuned mass damper {true/false}

    *Default* = False

:code:`TTMDfile` : String
    Name of the file for tower tuned mass damper (quoted string)
    [unused when CompTTMD is false]

    *Default* = none

:code:`DLL_ProcName` : String
    Name of procedure in DLL to be called (-) [case sensitive; used
    only with DLL Interface]

    *Default* = DISCON

:code:`DLL_DT` : String
    Communication interval for dynamic library (s) (or 'default')
    [used only with Bladed Interface]

    *Default* = default

:code:`DLL_Ramp` : Boolean
    Whether a linear ramp should be used between DLL_DT time steps
    [introduces time shift when true] (flag) [used only with Bladed
    Interface]

    *Default* = False

:code:`BPCutoff` : Float, Hz
    Cuttoff frequency for low-pass filter on blade pitch from DLL (Hz)
    [used only with Bladed Interface]

    *Default* = 99999.0

:code:`NacYaw_North` : Float, deg
    Reference yaw angle of the nacelle when the upwind end points due
    North (deg) [used only with Bladed Interface]

    *Default* = 0.0

:code:`Ptch_Cntrl` : Integer
    Record 28 Use individual pitch control {0 - collective pitch; 1 -
    individual pitch control} (switch) [used only with Bladed
    Interface]

    *Default* = 0

:code:`Ptch_SetPnt` : Float, deg
    Record  5 Below-rated pitch angle set-point (deg) [used only with
    Bladed Interface]

    *Default* = 0.0

:code:`Ptch_Min` : Float, deg
    Record  6 - Minimum pitch angle (deg) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`Ptch_Max` : Float, deg
    Record  7 Maximum pitch angle (deg) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`PtchRate_Min` : Float, deg / s
    Record  8 Minimum pitch rate (most negative value allowed) (deg/s)
    [used only with Bladed Interface]

    *Default* = 0.0

:code:`PtchRate_Max` : Float, deg / s
    Record  9 Maximum pitch rate  (deg/s) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`Gain_OM` : Float, N * m / (rad / s)**2
    Record 16 Optimal mode gain (Nm/(rad/s)^2) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`GenSpd_MinOM` : Float, rpm
    Record 17 Minimum generator speed (rpm) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`GenSpd_MaxOM` : Float, rpm
    Record 18 Optimal mode maximum speed (rpm) [used only with Bladed
    Interface]

    *Default* = 0.0

:code:`GenSpd_Dem` : Float, rpm
    Record 19 Demanded generator speed above rated (rpm) [used only
    with Bladed Interface]

    *Default* = 0.0

:code:`GenTrq_Dem` : Float, N * m
    Record 22 Demanded generator torque above rated (Nm) [used only
    with Bladed Interface]

    *Default* = 0.0

:code:`GenPwr_Dem` : Float, W
    Record 13 Demanded power (W) [used only with Bladed Interface]

    *Default* = 0.0

:code:`DLL_NumTrq` : Integer
    Record 26 No. of points in torque-speed look-up table {0 = none
    and use the optimal mode parameters; nonzero = ignore the optimal
    mode PARAMETERs by setting Record 16 to 0.0} (-) [used only with
    Bladed Interface]

    *Default* = 0

:code:`SumPrint` : Boolean
    Print summary data to '<RootName>.sum' (flag)

    *Default* = False

:code:`OutFile` : Integer
    Switch to determine where output will be placed 1 in module output
    file only; 2 in glue code output file only; 3 both (currently
    unused)

    *Default* = 1

:code:`TabDelim` : Boolean
    Use tab delimiters in text tabular output file? (flag) (currently
    unused)

    *Default* = True

:code:`OutFmt` : String
    Format used for text tabular output (except time).  Resulting
    field should be 10 characters. (quoted string (currently unused)

    *Default* = ES10.3E2

:code:`TStart` : Float, s
    Time to begin tabular output (s) (currently unused)

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0




outlist
########################################

:code:`from_openfast` : Boolean
    Whether we derive OpenFAST model from an existing model and ignore
    WISDEM

    *Default* = False

:code:`openfast_file` : String
    Main (.fst) OpenFAST input file name. No directory.

    *Default* = unused

:code:`openfast_dir` : String
    OpenFAST input directory, containing .fst file.  Absolute path or
    relative to modeling input

    *Default* = unused



xfoil
########################################

:code:`path` : String
    File path to xfoil executable (e.g. /home/user/Xfoil/bin/xfoil)

    *Default* = 

:code:`run_parallel` : Boolean
    Whether or not to run xfoil in parallel (requires mpi setup)

    *Default* = False



Level2
****************************************

Options for WEIS fidelity level 2 = linearized time domain (OpenFAST)
:code:`flag` : Boolean
    Whether or not to run WEIS fidelity level 2 = linearized OpenFAST

    *Default* = False



simulation
########################################

:code:`flag` : Boolean
    Whether or not to run a level 2 time domain simulation

    *Default* = False

:code:`TMax` : Float, s
    Total run time (s)

    *Default* = 720.0

    *Minimum* = 0.0    *Maximum* = 100000.0




linearization
########################################

:code:`TMax` : Float, s
    Total run time (s)

    *Default* = 720.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`DT` : Float, s
    Integration time step (s)

    *Default* = 0.025

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`wind_speeds` : Array of Floats
    List of wind speeds at which to linearize (m/s)

    *Default* = [14.0, 16.0, 18.0]

    *Minimum* = 0.0

    *Maximum* = 200.0

:code:`rated_offset` : Float, m/s
    Amount to increase rated wind speed from cc-blade to openfast with
    DOFs enabled.  In general, the more DOFs, the greater this value.

    *Default* = 1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`DOFs` : Array of Strings
    List of degrees-of-freedom to linearize about

    *Default* = ['GenDOF', 'TwFADOF1']

:code:`TrimTol` : Float
    Tolerance for the rotational speed convergence [used only if
    CalcSteady=True] (-)

    *Default* = 1e-05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`TrimGain` : Float, rad/(rad/s)
    Proportional gain for the rotational speed error (>0) [used only
    if CalcSteady=True] (rad/(rad/s) for yaw or pitch; Nm/(rad/s) for
    torque)

    *Default* = 0.0001

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`Twr_Kdmp` : Float, kg/s
    Damping factor for the tower [used only if CalcSteady=True]
    (N/(m/s))

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`Bld_Kdmp` : Float, kg/s
    Damping factor for the blades [used only if CalcSteady=True]
    (N/(m/s))

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`NLinTimes` : Integer
    Number of times to linearize (-) [>=1] [unused if Linearize=False]

    *Default* = 12

    *Minimum* = 0    *Maximum* = 120


:code:`LinTimes` : Array of Floats
    List of times at which to linearize (s) [1 to NLinTimes] [used
    only when Linearize=True and CalcSteady=False]

    *Default* = [30.0, 60.0]

    *Minimum* = 0.0

    *Maximum* = 10000.0



DTQP
########################################

:code:`flag` : Boolean
    Whether or not to run a DTQP optimization at level 2

    *Default* = False

:code:`nt` : Float
    Number of timesteps in DTQP timeseries optimization

    *Default* = 1000

:code:`maxiters` : Float
    Maximum number of DTQP optimization iterations

    *Default* = 150000

:code:`tolerance` : Float
    Tolerance of DTQP optimization

    *Default* = 0.0001

:code:`function` : String from, ['osqp', 'ipopt']
    Solver used for DTQP optimization

    *Default* = osqp



DLC_driver
****************************************



DLCs
########################################

:code:`DLC` : String from, ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '5.1', '6.1', '6.2', '6.3', '6.4', '6.5', '12.1', 'Custom']
    IEC design load case to run. The DLCs currently supported are 1.1,
    1.2, 1.3, 1.4, 1.5, 1.6, 5.1, 6.1, 6.3, and 6.4

    *Default* = 1.1

:code:`wind_speed` : Array of Floats, m/s
    Wind speeds for this DLC. If these are defined, ws_bin_size is
    neglected.

    *Default* = []

    *Minimum* = 0.0

    *Maximum* = 200.0

:code:`ws_bin_size` : Float, m/s
    Size of the wind speed bin between cut in and cout out wind
    speeds. It usually can be set to 2 m/s. This entry is neglected if
    the wind speeds are specified by the user.

    *Default* = 2

    *Minimum* = 0.01    *Maximum* = 20.0


:code:`n_seeds` : Integer
    Number of turbulent wind seeds drawn from the numpy random integer
    generator. This entry is neglected if the entry wind_seed is
    defined.  If DLC 1.4, number of waves seeds.

    *Default* = 1

    *Minimum* = 1    *Maximum* = 100


:code:`n_azimuth` : Integer
    Number of azimuth initial conditions to use (primarily during DLC
    5.1)

    *Default* = 1

    *Minimum* = 1    *Maximum* = 100


:code:`wind_seed` : Array of Integers
    Array of turbulent wind seeds for TurbSim. If these are defined,
    n_seeds is neglected.

    *Default* = []

:code:`wave_seeds` : Array of Integers
    Wave random number generator seeds for HydroDyn

    *Default* = []

:code:`wind_heading` : Array of Floats, deg
    Wind direction from north. This array must currently have either
    length=1, i.e. one constant value, or the same length of the array
    wind_speed.

    *Default* = [0.0]

    *Minimum* = -180.0

    *Maximum* = 180.0

:code:`yaw_misalign` : Array of Floats, deg
    Alignment of the nacelle with respect to north. This array must
    currently have either length=1, i.e. one constant value, or the
    same length of the array wind_speed. Default depends on DLC,
    specified in dlc_generator.

    *Minimum* = -180.0

    *Maximum* = 180.0

:code:`turbine_status` : String from, ['operating', 'parked-idling', 'parked-still']
    Status of the turbine, it can be either operating, parked-idling,
    or parked-still. Each DLC come with its default turbine status
    specified by the standards.

    *Default* = operating

:code:`wave_period` : Array of Floats, s
    Period between waves. If this array is populated by the user, then
    the field metocean_conditions is neglected. If wave_period is not
    defined, metocean_conditions will be used, either in the values
    provided by the user or with its default values (the first option
    is highly recommended).

    *Default* = []

    *Minimum* = 0.0

    *Maximum* = 1000.0

:code:`wave_height` : Array of Floats, m
    Height of the waves. If this array is populated by the user, then
    the field metocean_conditions is neglected. If wave_height is not
    defined, metocean_conditions will be used, either in the values
    provided by the user or with its default values (the first option
    is highly recommended).

    *Default* = []

    *Minimum* = 0.0

    *Maximum* = 100.0

:code:`wave_heading` : Array of Floats, deg
    Heading of the waves with respect to north. This array must
    currently have either length=1, i.e. one constant value, or the
    same length of the array wind_speed

    *Default* = [0.0]

    *Minimum* = -180.0

    *Maximum* = 180.0

:code:`wave_gamma` : Array of Floats
    Peak-shape parameter of incident wave spectrum. If 0, the default
    from IEC61400-3 / HydroDyn is used. This array must currently have
    either length=1, i.e. one constant value, or the same length of
    the array wind_speed

    *Default* = [0.0]

    *Minimum* = 0.0

    *Maximum* = 10.0

:code:`probabilities` : Array of Floats
    Probability of occurrance for each case. This entry is relevant
    only for DLC 1.2 and 6.4. This array must currently have either
    length=1, i.e. one constant value, or the same length of the array
    wind_speed.

    *Default* = [1.0]

    *Minimum* = 0.0

    *Maximum* = 1.0

:code:`IEC_WindType` : String from, ['NTM', '1ETM', '2ETM', '3ETM', '1EWM1', '2EWM1', '3EWM1', '1EWM50', '2EWM50', '3EWM50', 'ECD', 'EDC', 'EOG']
    IEC turbulence type ('NTM'=normal, 'xETM'=extreme turbulence,
    'xEWM1'=extreme 1-year wind, 'xEWM50'=extreme 50-year wind, where
    x=wind turbine class 1, 2, or 3), 'ECD'=extreme coherent gust with
    direction change, 'EDC'=extreme direction change, 'EOG'=extreme
    operating gust. Normally the user does not need to define this
    entry.

    *Default* = NTM

:code:`analysis_time` : Float, s
    This is the length of the simulation where outputs will be
    recorded. Its default is 600 seconds (10 minutes) for most
    simulations, except for the coherent cases where a shorter time
    window of 200 s is used.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`transient_time` : Float, s
    This is the length of the simulation where outputs will be
    discarded. Its default is 120 seconds (2 minutes) for all
    simulations. The total simulation time is the sum of analysis_time
    and transient_time

    *Default* = 120.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`shutdown_time` : Float, s
    Time when shutdown occurs in DLC 5.1

    *Default* = 9999

    *Minimum* = 0.0    *Maximum* = 100000.0


:code:`wind_file` : String
    File path of custom wind file



turbulent_wind
========================================

These are all inputs to TurbSim. These inputs usually do not need to be set unless you are trying to customize a DLC
:code:`flag` : Boolean
    Flag switching between steady wind and turbulent wind grid from
    TurbSim.

    *Default* = False

:code:`Echo` : Boolean
    Echo input data to <RootName>.ech (flag)

    *Default* = False

:code:`RandSeed1` : Integer
    First random seed  (-2147483648 to 2147483647)

    *Default* = 1

:code:`WrBHHTP` : Boolean
    Output hub-height turbulence parameters in binary form?
    (Generates RootName.bin)

    *Default* = False

:code:`WrFHHTP` : Boolean
    Output hub-height turbulence parameters in formatted form?
    (Generates RootName.dat)

    *Default* = False

:code:`WrADHH` : Boolean
    Output hub-height time-series data in AeroDyn form?  (Generates
    RootName.hh)

    *Default* = False

:code:`WrADFF` : Boolean
    Output full-field time-series data in TurbSim/AeroDyn form?
    (Generates RootName.bts)

    *Default* = True

:code:`WrBLFF` : Boolean
    Output full-field time-series data in BLADED/AeroDyn form?
    (Generates RootName.wnd)

    *Default* = False

:code:`WrADTWR` : Boolean
    Output tower time-series data? (Generates RootName.twr)

    *Default* = False

:code:`WrFMTFF` : Boolean
    Output full-field time-series data in formatted (readable) form?
    (Generates RootName.u, RootName.v, RootName.w)

    *Default* = False

:code:`WrACT` : Boolean
    Output coherent turbulence time steps in AeroDyn form? (Generates
    RootName.cts)

    *Default* = False

:code:`Clockwise` : Boolean
    Clockwise rotation looking downwind? (used only for full-field
    binary files - not necessary for AeroDyn)

    *Default* = False

:code:`ScaleIEC` : Integer
    Scale IEC turbulence models to exact target standard deviation?
    [0=no additional scaling; 1=use hub scale uniformly; 2=use
    individual scales]

    *Default* = 0

:code:`NumGrid_Z` : Integer
    Vertical grid-point matrix dimension

    *Default* = 25

    *Minimum* = 5    *Maximum* = 100


:code:`NumGrid_Y` : Integer
    Horizontal grid-point matrix dimension

    *Default* = 25

    *Minimum* = 5    *Maximum* = 100


:code:`TimeStep` : Float, s
    Time step [seconds]

    *Default* = 0.05

    *Minimum* = 0.0001    *Maximum* = 1.0


:code:`UsableTime` : String
    Usable length of output time series [seconds] (program will add
    GridWidth/MeanHHWS seconds unless UsableTime is 'ALL')

    *Default* = ALL

:code:`HubHt` : Float, m
    Hub height [m] (should be > 0.5*GridHeight)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 500.0


:code:`GridHeight` : Float, m
    Grid height [m]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 500.0


:code:`GridWidth` : Float, m
    Grid width [m] (should be >= 2*(RotorRadius+ShaftLength))

    *Default* = 0

    *Minimum* = 0    *Maximum* = 500.0


:code:`VFlowAng` : Float, deg
    Vertical mean flow (uptilt) angle [degrees]

    *Default* = 0.0

    *Minimum* = -90.0    *Maximum* = 90.0


:code:`HFlowAng` : Float, deg
    Horizontal mean flow (skew) angle [degrees]

    *Default* = 0.0

    *Minimum* = -90.0    *Maximum* = 90.0


:code:`TurbModel` : String from, ['IECKAI', 'IECVKM', 'GP_LLJ', 'NWTCUP', 'SMOOTH', 'WF_UPW', 'WF_07D', 'WF_14D', 'TIDAL', 'API', 'USRINP', 'TIMESR', 'NONE']
    Turbulence model

    *Default* = IECKAI

:code:`UserFile` : String
    Name of the file that contains inputs for user-defined spectra or
    time series inputs (used only for "USRINP" and "TIMESR" models)

    *Default* = unused

:code:`IECstandard` : String from, ['1-ED3', '1-ED2']
    Number of IEC 61400-x standard (x=1,2, or 3 with optional 61400-1
    edition number (i.e. "1-Ed2") )

    *Default* = 1-ED3

:code:`ETMc` : String
    IEC Extreme Turbulence Model

    *Default* = default

:code:`WindProfileType` : String from, ['LOG', 'PL', 'JET', 'H2L', 'API', 'USR', 'TS', 'IEC', 'LOG', 'default']
    Velocity profile type ('LOG';'PL'=power law;'JET';'H2L'=Log law
    for TIDAL model;'API';'USR';'TS';'IEC'=PL on rotor disk, LOG
    elsewhere; or 'default')

    *Default* = PL

:code:`ProfileFile` : String
    Name of the file that contains input profiles for
    WindProfileType='USR' and/or TurbModel='USRVKM' [-]

    *Default* = unused

:code:`RefHt` : Float, m
    Height of the reference velocity (URef) [m]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 100000.0


:code:`URef` : Float, m/s
    Mean (total) velocity at the reference height [m/s] (or 'default'
    for JET velocity profile) [must be 1-hr mean for API model;
    otherwise is the mean over AnalysisTime seconds]

    *Default* = -1

:code:`IECturbc` : Float, (-)
    Turbulence intensity (fraction) for custom DLCs, if default (-1),
    the class letter will be used

    *Default* = -1

:code:`ZJetMax` : String
    Jet height [m] (used only for JET velocity profile, valid 70-490
    m)

    *Default* = default

:code:`PLExp` : Float
    Power law exponent [-] (or 'default'), if default (-1), the
    environment option shear_exp will be used for all DLCs

    *Default* = -1

:code:`Z0` : String
    Surface roughness length [m] (or 'default')

    *Default* = default

:code:`Latitude` : String
    Site latitude [degrees] (or 'default')

    *Default* = default

:code:`RICH_NO` : Float
    Gradient Richardson number [-]

    *Default* = 0.05

:code:`UStar` : String
    Friction or shear velocity [m/s] (or 'default')

    *Default* = default

:code:`ZI` : String
    Mixing layer depth [m] (or 'default')

    *Default* = default

:code:`PC_UW` : String
    Hub mean uw Reynolds stress [m^2/s^2] (or 'default' or 'none')

    *Default* = default

:code:`PC_UV` : String
    Hub mean uv Reynolds stress [m^2/s^2] (or 'default' or 'none')

    *Default* = default

:code:`PC_VW` : String
    Hub mean vw Reynolds stress [m^2/s^2] (or 'default' or 'none')

    *Default* = default

:code:`SCMod1` : String
    u-component coherence model ('GENERAL', 'IEC', 'API', 'NONE', or
    'default')

    *Default* = default

:code:`SCMod2` : String
    v-component coherence model ('GENERAL', 'IEC', 'NONE', or
    'default')

    *Default* = default

:code:`SCMod3` : String
    w-component coherence model ('GENERAL', 'IEC', 'NONE', or
    'default')

    *Default* = default

:code:`InCDec1` : String
    u-component coherence parameters for general or IEC models [-,
    m^-1] (e.g. '10.0  0.3e-3' in quotes) (or 'default')

    *Default* = default

:code:`InCDec2` : String
    v-component coherence parameters for general or IEC models [-,
    m^-1] (e.g. '10.0  0.3e-3' in quotes) (or 'default')

    *Default* = default

:code:`InCDec3` : String
    w-component coherence parameters for general or IEC models [-,
    m^-1] (e.g. '10.0  0.3e-3' in quotes) (or 'default')

    *Default* = default

:code:`CohExp` : String
    Coherence exponent for general model [-] (or 'default')

    *Default* = default

:code:`CTEventPath` : String
    Name of the path where event data files are located

    *Default* = unused

:code:`CTEventFile` : String from, ['LES', 'DNS', 'RANDOM']
    Type of event files

    *Default* = RANDOM

:code:`Randomize` : Boolean
    Randomize the disturbance scale and locations? (true/false)

    *Default* = True

:code:`DistScl` : Float
    Disturbance scale [-] (ratio of event dataset height to rotor
    disk). (Ignored when Randomize = true.)

    *Default* = 1.0

    *Minimum* = 0    *Maximum* = 1.0


:code:`CTLy` : Float
    Fractional location of tower centerline from right [-] (looking
    downwind) to left side of the dataset. (Ignored when Randomize =
    true.)

    *Default* = 0.5

    *Minimum* = 0    *Maximum* = 1.0


:code:`CTLz` : Float
    Fractional location of hub height from the bottom of the dataset.
    [-] (Ignored when Randomize = true.)

    *Default* = 0.5

    *Minimum* = 0    *Maximum* = 1.0


:code:`CTStartTime` : Float, s
    Minimum start time for coherent structures in RootName.cts

    *Default* = 30

    *Minimum* = 0    *Maximum* = 1000.0


:code:`fix_wind_seeds` : Boolean
    Fix the seed of the random integer generator controlling the seed
    of TurbSim. When set to False, the seeds change everytime the DLC
    generator class is called. It is recommended to keep it to True
    when the optimization is on, or different wind seeds will be
    generated for every function call, complicating the smoothness of
    the solution space. Even when set to True, the wind seeds are
    different across wind speeds and DLCs.

    *Default* = True

:code:`fix_wave_seeds` : Boolean
    Fix the seed of the random integer generator controlling the wave
    seed of HydroDyn. When set to False, the seeds change everytime
    the DLC generator class is called. It is recommended to keep it to
    True when the optimization is on, or different wave seeds will be
    generated for every function call, complicating the smoothness of
    the solution space. Even when set to True, the wave seeds are
    different across wind speeds and DLCs.

    *Default* = True



metocean_conditions
########################################

Here the metocean conditions can be specified in terms of wind speeds, significant wave height (Hs), and wave period (Tp) for normal sea state (NSS), fatigue calculations, and severe sea state (SSS). Currently WEIS neglects the joint probability density function crossing wind/wave directionality, wave peak shape parameter gamma
:code:`wind_speed` : Array of Floats, m/s
    Array of wind speeds to tabulate Hs and Tp

    *Default* = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]

    *Minimum* = 0.0

    *Maximum* = 50.0

:code:`wave_height_NSS` : Array of Floats, m
    Array of Hs for NSS conditional to wind speed

    *Default* = [1.1, 1.18, 1.32, 1.54, 1.84, 2.19, 2.6, 3.06, 3.62, 4.03, 4.52]

    *Minimum* = 0.0

    *Maximum* = 100.0

:code:`wave_period_NSS` : Array of Floats, s
    Array of Tp for NSS conditional to wind speed

    *Default* = [8.52, 8.31, 8.01, 7.65, 7.44, 7.46, 7.64, 8.05, 8.52, 8.99, 9.45]

    *Minimum* = 0.0

    *Maximum* = 1000.0

:code:`wave_height_fatigue` : Array of Floats, m
    Array of Hs for fatigue computations conditional to wind speed

    *Default* = [1.1, 1.18, 1.32, 1.54, 1.84, 2.19, 2.6, 3.06, 3.62, 4.03, 4.52]

    *Minimum* = 0.0

    *Maximum* = 100.0

:code:`wave_period_fatigue` : Array of Floats, s
    Array of Tp for fatigue computations conditional to wind speed

    *Default* = [8.52, 8.31, 8.01, 7.65, 7.44, 7.46, 7.64, 8.05, 8.52, 8.99, 9.45]

    *Minimum* = 0.0

    *Maximum* = 1000.0

:code:`wave_height_SSS` : Array of Floats, m
    Array of Hs for SSS conditional to wind speed

    *Default* = [1.1, 1.18, 1.32, 1.54, 1.84, 2.19, 2.6, 3.06, 3.62, 4.03, 4.52]

    *Minimum* = 0.0

    *Maximum* = 100.0

:code:`wave_period_SSS` : Array of Floats, s
    Array of Tp for SSS conditional to wind speed

    *Default* = [8.52, 8.31, 8.01, 7.65, 7.44, 7.46, 7.64, 8.05, 8.52, 8.99, 9.45]

    *Minimum* = 0.0

    *Maximum* = 1000.0

:code:`wave_height50` : Float, m
    Wave height with 50-year occurrence, used in DLC 6.1

    *Default* = 15.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`wave_period50` : Float, s
    Wave period with 50-year occurrence, used in DLC 6.1

    *Default* = 15.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`wave_height1` : Float, m
    Wave height with 1-year occurrence, used in DLC 6.3, 7.1, and 8.2

    *Default* = 15.0

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`wave_period1` : Float, s
    Wave period with 1-year occurrence, used in DLC 6.3, 7.1, and 8.2

    *Default* = 15.0

    *Minimum* = 0.0    *Maximum* = 1000.0




ROSCO
****************************************

Options for WEIS fidelity level 3 = nonlinear time domain. Inherited from ROSCO/rosco/toolbox/inputs/toolbox_shema.yaml
:code:`LoggingLevel` : Float
    0- write no debug files, 1- write standard output .dbg-file, 2-
    write standard output .dbg-file and complete avrSWAP-array
    .dbg2-file

    *Default* = 1

    *Minimum* = 0    *Maximum* = 3


:code:`F_LPFType` : Float
    1- first-order low-pass filter, 2- second-order low-pass filter,
    [rad/s] (currently filters generator speed and pitch control
    signals)

    *Default* = 1

    *Minimum* = 1    *Maximum* = 2


:code:`F_NotchType` : Float
    Notch on the measured generator speed and/or tower fore-aft motion
    (for floating) {0- disable, 1- generator speed, 2- tower-top fore-
    aft motion, 3- generator speed and tower-top fore-aft motion}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 3


:code:`IPC_ControlMode` : Float
    Turn Individual Pitch Control (IPC) for fatigue load reductions
    (pitch contribution) (0- off, 1- 1P reductions, 2- 1P+2P
    reduction)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`VS_ControlMode` : Float
    Generator torque control mode in above rated conditions (0- no
    torque control, 1- k*omega^2 with PI transitions, 2- WSE TSR
    Tracking, 3- Power-based TSR Tracking)

    *Default* = 2

    *Minimum* = 0    *Maximum* = 3


:code:`VS_ConstPower` : Float
    Do constant power torque control, where above rated torque varies,
    0 for constant torque

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`PC_ControlMode` : Float
    Blade pitch control mode (0- No pitch, fix to fine pitch, 1-
    active PI blade pitch control)

    *Default* = 1

    *Minimum* = 0    *Maximum* = 1


:code:`Y_ControlMode` : Float
    Yaw control mode (0- no yaw control, 1- yaw rate control, 2- yaw-
    by-IPC)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`SS_Mode` : Float
    Setpoint Smoother mode (0- no setpoint smoothing, 1- introduce
    setpoint smoothing)

    *Default* = 1

    *Minimum* = 0    *Maximum* = 2


:code:`WE_Mode` : Float
    Wind speed estimator mode (0- One-second low pass filtered hub
    height wind speed, 1- Immersion and Invariance Estimator (Ortega
    et al.)

    *Default* = 2

    *Minimum* = 0    *Maximum* = 2


:code:`PS_Mode` : Float
    Pitch saturation mode (0- no pitch saturation, 1- peak shaving, 2-
    Cp-maximizing pitch saturation, 3- peak shaving and Cp-maximizing
    pitch saturation)

    *Default* = 3

    *Minimum* = 0    *Maximum* = 3


:code:`SD_Mode` : Float
    Shutdown mode (0- no shutdown procedure, 1- pitch to max pitch at
    shutdown)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`TD_Mode` : Float
    Tower damper mode (0- no tower damper, 1- feed back translational
    nacelle accelleration to pitch angle

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`TRA_Mode` : Float
    Tower resonance avoidance mode (0- no tower resonsnace avoidance,
    1- use torque control setpoints to avoid a specific frequency

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`Fl_Mode` : Float
    Floating specific feedback mode (0- no nacelle velocity feedback,
    1 - nacelle velocity feedback, 2 - nacelle pitching acceleration
    feedback)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`Flp_Mode` : Float
    Flap control mode (0- no flap control, 1- steady state flap angle,
    2- Proportional flap control)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`PwC_Mode` : Float
    Active Power Control Mode (0- no active power control 1- constant
    active power control, 2- open loop power vs time, 3- open loop
    power vs. wind speed)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`ZMQ_Mode` : Float
    ZMQ Mode (0 - ZMQ Inteface, 1 - ZMQ for yaw control)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`ZMQ_UpdatePeriod` : Float
    Call ZeroMQ every [x] seconds, [s]

    *Default* = 2

    *Minimum* = 0

:code:`PA_Mode` : Float
    Pitch actuator mode {0 - not used, 1 - first order filter, 2 -
    second order filter}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`PF_Mode` : Float
    Pitch fault mode {0 - not used, 1 - constant offset on one or more
    blades}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`OL_Mode` : Float
    Open loop control mode {0- no open loop control, 1- open loop
    control}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`AWC_Mode` : Float
    Active wake control mode {0 - not used, 1 - SNL method, 2 - NREL
    method}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`Ext_Mode` : Float
    External control mode [0 - not used, 1 - call external dynamic
    library]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1


:code:`CC_Mode` : Float
    Cable control mode [0- unused, 1- User defined, 2- Open loop
    control]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`StC_Mode` : Float
    Structural control mode [0- unused, 1- User defined, 2- Open loop
    control]

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`U_pc` : Array of Floats
    List of wind speeds to schedule pitch control zeta and omega

    *Default* = [12]

    *Minimum* = 0

:code:`interp_type` : String from, ['sigma', 'linear', 'quadratic', 'cubic']
    Type of interpolation between above rated tuning values (only used
    for multiple pitch controller tuning values)

    *Default* = sigma

:code:`zeta_vs` : Float
    Torque controller desired damping ratio [-]

    *Default* = 1.0

    *Minimum* = 0

:code:`omega_vs` : Float, rad/s
    Torque controller desired natural frequency [rad/s]

    *Default* = 0.2

    *Minimum* = 0

:code:`max_pitch` : Float, rad
    Maximum pitch angle [rad], {default = 90 degrees}

    *Default* = 1.57

:code:`min_pitch` : Float, rad
    Minimum pitch angle [rad], {default = 0 degrees}

    *Default* = 0

:code:`vs_minspd` : Float, rad/s
    Minimum rotor speed [rad/s], {default = 0 rad/s}

    *Default* = 0

:code:`ss_vsgain` : Float
    Torque controller setpoint smoother gain bias percentage [%, <= 1
    ], {default = 100%}

    *Default* = 1.0

:code:`ss_pcgain` : Float, rad
    Pitch controller setpoint smoother gain bias percentage  [%, <= 1
    ], {default = 0.1%}

    *Default* = 0.001

:code:`ps_percent` : Float, rad
    Percent peak shaving  [%, <= 1 ], {default = 80%}

    *Default* = 0.8    *Maximum* = 1


:code:`sd_maxpit` : Float, rad
    Maximum blade pitch angle to initiate shutdown [rad], {default =
    40 deg.}

    *Default* = 0.6981

:code:`flp_maxpit` : Float, rad
    Maximum (and minimum) flap pitch angle [rad]

    *Default* = 0.1745

:code:`twr_freq` : Float, rad/s
    Tower natural frequency, for floating only

    *Minimum* = 0

:code:`ptfm_freq` : Float, rad/s
    Platform natural frequency, for floating only

    *Minimum* = 0

:code:`WS_GS_n` : Float
    Number of wind speed breakpoints

    *Default* = 60

    *Minimum* = 0

:code:`PC_GS_n` : Float
    Number of pitch angle gain scheduling breakpoints

    *Default* = 30

    *Minimum* = 0

:code:`tune_Fl` : Boolean
    Whether to automatically tune Kp_float

    *Default* = True

:code:`zeta_flp` : Float
    Flap controller desired damping ratio [-]

    *Minimum* = 0

:code:`omega_flp` : Float, rad/s
    Flap controller desired natural frequency [rad/s]

    *Minimum* = 0

:code:`flp_kp_norm` : Float
    Flap controller normalization term for DC gain (kappa)

    *Minimum* = 0

:code:`flp_tau` : Float, s
    Flap controller time constant for integral gain

    *Minimum* = 0

:code:`max_torque_factor` : Float
    Maximum torque = rated torque * max_torque_factor

    *Default* = 1.1

    *Minimum* = 0

:code:`IPC_Kp1p` : Float, s
    Proportional gain for IPC, 1P [s]

    *Default* = 0.0

    *Minimum* = 0

:code:`IPC_Kp2p` : Float
    Proportional gain for IPC, 2P [-]

    *Default* = 0.0

    *Minimum* = 0

:code:`IPC_Ki1p` : Float, s
    Integral gain for IPC, 1P [s]

    *Default* = 0.0

    *Minimum* = 0

:code:`IPC_Ki2p` : Float
    integral gain for IPC, 2P [-]

    *Default* = 0.0

    *Minimum* = 0

:code:`IPC_Vramp` : Array of Floats
    wind speeds for IPC cut-in sigma function [m/s]

    *Default* = [0.0, 0.0]

    *Minimum* = 0.0

:code:`rgn2k_factor` : Float
    Factor on VS_Rgn2K to increase/decrease optimal torque control
    gain, default is 1.  Sometimes environmental conditions or
    differences in BEM solvers necessitate this change.

    *Default* = 1

    *Minimum* = 0



filter_params
########################################

:code:`f_lpf_cornerfreq` : Float, rad/s
    Corner frequency (-3dB point) in the first order low pass filter
    of the generator speed [rad/s]

    *Minimum* = 0

:code:`f_lpf_damping` : Float, rad/s
    Damping ratio in the first order low pass filter of the generator
    speed [-]

    *Minimum* = 0

:code:`f_we_cornerfreq` : Float, rad/s
    Corner frequency (-3dB point) in the first order low pass filter
    for the wind speed estimate [rad/s]

    *Default* = 0.20944

    *Minimum* = 0

:code:`f_fl_highpassfreq` : Float, rad/s
    Natural frequency of first-order high-pass filter for nacelle
    fore-aft motion [rad/s]

    *Default* = 0.01042

    *Minimum* = 0

:code:`f_ss_cornerfreq` : Float, rad/s
    First order low-pass filter cornering frequency for setpoint
    smoother [rad/s]

    *Default* = 0.6283

    *Minimum* = 0

:code:`f_yawerr` : Float, rad/s
    Low pass filter corner frequency for yaw controller [rad/

    *Default* = 0.17952

    *Minimum* = 0

:code:`f_sd_cornerfreq` : Float, rad
    Cutoff Frequency for first order low-pass filter for blade pitch
    angle [rad/s], {default = 0.41888 ~ time constant of 15s}

    *Default* = 0.41888



open_loop
########################################

:code:`flag` : Boolean
    Flag to use open loop control

    *Default* = False

:code:`filename` : String
    Filename of open loop input that ROSCO reads

    *Default* = unused

:code:`Ind_Breakpoint` : Float
    Index (column, 1-indexed) of breakpoint (time) in open loop index

    *Default* = 1

    *Minimum* = 0

:code:`Ind_BldPitch` : Array of Floats
    Indices (columns, 1-indexed) of pitch (1,2,3) inputs in open loop
    input

    *Default* = [0, 0, 0]

    *Minimum* = 0

:code:`Ind_GenTq` : Float
    Index (column, 1-indexed) of generator torque in open loop input

    *Default* = 0

    *Minimum* = 0

:code:`Ind_YawRate` : Float
    Index (column, 1-indexed) of nacelle yaw in open loop input

    *Default* = 0

    *Minimum* = 0

:code:`Ind_Azimuth` : Float
    The column in OL_Filename that contains the desired azimuth
    position in rad (used if OL_Mode = 2)

    *Default* = 0

:code:`Ind_CableControl` : Array of Floats
    The column in OL_Filename that contains the cable control inputs
    in m

:code:`Ind_StructControl` : Array of Floats
    The column in OL_Filename that contains the structural control
    inputs in various units

:code:`PA_CornerFreq` : Float, rad/s
    Pitch actuator natural frequency [rad/s]

    *Default* = 3.14

    *Minimum* = 0

:code:`PA_Damping` : Float
    Pitch actuator damping ratio [-]

    *Default* = 0.707

    *Minimum* = 0



DISCON
########################################

These are pass-through parameters for the DISCON.IN file.  Use with caution. Do not set defaults in schema.
:code:`LoggingLevel` : Float
    (0- write no debug files, 1- write standard output .dbg-file, 2-
    write standard output .dbg-file and complete avrSWAP-array
    .dbg2-file)

:code:`Echo` : Float
    0 - no Echo, 1 - Echo input data to <RootName>.echo

    *Default* = 0

:code:`DT_Out` : Float
    Time step to output .dbg* files, or 0 to match sampling period of
    OpenFAST

    *Default* = 0

:code:`Ext_Interface` : Float
    0 - use standard bladed interface, 1 - Use the extened DLL
    interface introduced in OpenFAST 3.5.0.

    *Default* = 1

    *Minimum* = 0    *Maximum* = 1


:code:`F_LPFType` : Float
    1- first-order low-pass filter, 2- second-order low-pass filter
    (currently filters generator speed and pitch control signals

:code:`VS_ControlMode` : Float
    Generator torque control mode in above rated conditions (0- no
    torque control, 1- k*omega^2 with PI transitions, 2- WSE TSR
    Tracking, 3- Power-based TSR Tracking)

    *Minimum* = 0    *Maximum* = 3


:code:`VS_ConstPower` : Float
    Do constant power torque control, where above rated torque varies

    *Minimum* = 0    *Maximum* = 1


:code:`F_NotchType` : Float
    Notch on the measured generator speed and/or tower fore-aft motion
    (for floating) (0- disable, 1- generator speed, 2- tower-top fore-
    aft motion, 3- generator speed and tower-top fore-aft motion)

:code:`IPC_ControlMode` : Float
    Turn Individual Pitch Control (IPC) for fatigue load reductions
    (pitch contribution) (0- off, 1- 1P reductions, 2- 1P+2P
    reductions)

:code:`PC_ControlMode` : Float
    Blade pitch control mode (0- No pitch, fix to fine pitch, 1-
    active PI blade pitch control)

:code:`Y_ControlMode` : Float
    Yaw control mode (0- no yaw control, 1- yaw rate control, 2- yaw-
    by-IPC)

:code:`SS_Mode` : Float
    Setpoint Smoother mode (0- no setpoint smoothing, 1- introduce
    setpoint smoothing)

:code:`WE_Mode` : Float
    Wind speed estimator mode (0- One-second low pass filtered hub
    height wind speed, 1- Immersion and Invariance Estimator, 2-
    Extended Kalman Filter)

:code:`PS_Mode` : Float
    Pitch saturation mode (0- no pitch saturation, 1- implement pitch
    saturation)

:code:`SD_Mode` : Float
    Shutdown mode (0- no shutdown procedure, 1- pitch to max pitch at
    shutdown)

:code:`Fl_Mode` : Float
    Floating specific feedback mode (0- no nacelle velocity feedback,
    1- feed back translational velocity, 2- feed back rotational
    veloicty)

:code:`Flp_Mode` : Float
    Flap control mode (0- no flap control, 1- steady state flap angle,
    2- Proportional flap control)

:code:`OL_Mode` : Float
    Open loop control mode (0 - no open-loop control, 1 - direct open
    loop control, 2 - rotor position control)

:code:`F_LPFCornerFreq` : Float, rad/s
    Corner frequency (-3dB point) in the low-pass filters,

:code:`F_LPFDamping` : Float
    Damping coefficient (used only when F_FilterType = 2 [-]

:code:`F_NumNotchFilts` : Float
    Number of notch filters placed on sensors

:code:`F_GenSpdNotch_N` : Float
    Number of notch filters on generator speed

:code:`F_TwrTopNotch_N` : Float
    Number of notch filters on tower top acceleration signal

:code:`F_SSCornerFreq` : Float, rad/s.
    Corner frequency (-3dB point) in the first order low pass filter
    for the setpoint smoother,

:code:`F_WECornerFreq` : Float, rad/s.
    Corner frequency (-3dB point) in the first order low pass filter
    for the wind speed estimate

:code:`F_FlCornerFreq` : Array of Floats
    Natural frequency and damping in the second order low pass filter
    of the tower-top fore-aft motion for floating feedback control

:code:`F_FlHighPassFreq` : Float, rad/s
    Natural frequency of first-order high-pass filter for nacelle
    fore-aft motion

:code:`F_FlpCornerFreq` : Array of Floats
    Corner frequency and damping in the second order low pass filter
    of the blade root bending moment for flap control

:code:`PC_GS_n` : Float
    Amount of gain-scheduling table entries

:code:`PC_GS_angles` : Array of Floats
    Gain-schedule table- pitch angles

:code:`PC_GS_KP` : Array of Floats
    Gain-schedule table- pitch controller kp gains

:code:`PC_GS_KI` : Array of Floats
    Gain-schedule table- pitch controller ki gains

:code:`PC_GS_KD` : Array of Floats
    Gain-schedule table- pitch controller kd gains

:code:`PC_GS_TF` : Array of Floats
    Gain-schedule table- pitch controller tf gains (derivative filter)

:code:`PC_MaxPit` : Float, rad
    Maximum physical pitch limit,

:code:`PC_MinPit` : Float, rad
    Minimum physical pitch limit,

:code:`PC_MaxRat` : Float, rad/s.
    Maximum pitch rate (in absolute value) in pitch controller

:code:`PC_MinRat` : Float, rad/s.
    Minimum pitch rate (in absolute value) in pitch controller

:code:`PC_RefSpd` : Float, rad/s.
    Desired (reference) HSS speed for pitch controller

:code:`PC_FinePit` : Float, rad
    Record 5- Below-rated pitch angle set-point

:code:`PC_Switch` : Float, rad
    Angle above lowest minimum pitch angle for switch

:code:`IPC_IntSat` : Float, rad
    Integrator saturation (maximum signal amplitude contribution to
    pitch from IPC)

:code:`IPC_SatMode` : Integer
    IPC Saturation method (0 - no saturation, 1 - saturate by
    PC_MinPit, 2 - saturate by PS_BldPitchMin)

:code:`IPC_KP` : Array of Floats
    Proportional gain for the individual pitch controller- first
    parameter for 1P reductions, second for 2P reductions, [-]

:code:`IPC_KI` : Array of Floats
    Integral gain for the individual pitch controller- first parameter
    for 1P reductions, second for 2P reductions, [-]

:code:`IPC_aziOffset` : Array of Floats
    Phase offset added to the azimuth angle for the individual pitch
    controller

:code:`IPC_CornerFreqAct` : Float, rad/s
    Corner frequency of the first-order actuators model, to induce a
    phase lag in the IPC signal (0- Disable)

:code:`VS_GenEff` : Float, percent
    Generator efficiency mechanical power -> electrical power, should
    match the efficiency defined in the generator properties

:code:`VS_ArSatTq` : Float, Nm
    Above rated generator torque PI control saturation

:code:`VS_MaxRat` : Float, Nm/s
    Maximum torque rate (in absolute value) in torque controller

:code:`VS_MaxTq` : Float, Nm
    Maximum generator torque in Region 3 (HSS side)

:code:`VS_MinTq` : Float, Nm
    Minimum generator torque (HSS side)

:code:`VS_MinOMSpd` : Float, rad/s
    Minimum generator speed

:code:`VS_Rgn2K` : Float, Nm/(rad/s)^2
    Generator torque constant in Region 2 (HSS side). Only used in
    VS_ControlMode = 1,3

:code:`VS_RtPwr` : Float, W
    Wind turbine rated power

:code:`VS_RtTq` : Float, Nm
    Rated torque

:code:`VS_RefSpd` : Float, rad/s
    Rated generator speed

:code:`VS_n` : Float
    Number of generator PI torque controller gains

:code:`VS_KP` : Float
    Proportional gain for generator PI torque controller. (Only used
    in the transitional 2.5 region if VS_ControlMode =/ 2)

:code:`VS_KI` : Float, s
    Integral gain for generator PI torque controller  (Only used in
    the transitional 2.5 region if VS_ControlMode =/ 2)

:code:`VS_TSRopt` : Float, rad
    Power-maximizing region 2 tip-speed-ratio. Only used in
    VS_ControlMode = 2.

:code:`VS_PwrFiltF` : Float, rad
    Low pass filter on power used to determine generator speed set
    point.  Only used in VS_ControlMode = 3.

    *Default* = 0.314

:code:`SS_VSGain` : Float
    Variable speed torque controller setpoint smoother gain

:code:`SS_PCGain` : Float
    Collective pitch controller setpoint smoother gain

:code:`PRC_Mode` : Float
    Power reference tracking mode, 0- use standard rotor speed set
    points, 1- use PRC rotor speed setpoints

:code:`PRC_WindSpeeds` : Array of Floats
    Array of wind speeds used in rotor speed vs. wind speed lookup
    table [m/s]

:code:`PRC_GenSpeeds` : Array of Floats
    Array of generator speeds corresponding to PRC_WindSpeeds [rad/s]

:code:`PRC_LPF_Freq` : Float
    Frequency of the low pass filter on the wind speed estimate used
    to set PRC_GenSpeeds [rad/s]

    *Default* = 0.078539

:code:`PRC_n` : Float
    Number of elements in PRC_WindSpeeds and PRC_GenSpeeds array

:code:`TRA_ExclSpeed` : Float
    Rotor speed for exclusion [LSS, rad/s]

    *Default* = 0.0

    *Minimum* = 0

:code:`TRA_ExclBand` : Float
    Size of the rotor frequency exclusion band [LSS, rad/s]. Torque
    controller reference will be TRA_ExclSpeed +/- TRA_ExlBand/2

    *Default* = 0.0

    *Minimum* = 0

:code:`TRA_RateLimit` : Float
    Rate limit of change in rotor speed reference [LSS, rad/s].
    Suggested to be VS_RefSpd/400.

    *Default* = 0.0

    *Minimum* = 0

:code:`WE_BladeRadius` : Float, m
    Blade length (distance from hub center to blade tip)

:code:`WE_CP_n` : Float
    Amount of parameters in the Cp array

:code:`WE_CP` : Array of Floats
    Parameters that define the parameterized CP(lambda) function

:code:`WE_Gamma` : Float, m/rad
    Adaption gain of the wind speed estimator algorithm

:code:`WE_GearboxRatio` : Float
    Gearbox ratio, >=1

:code:`WE_Jtot` : Float, kg m^2
    Total drivetrain inertia, including blades, hub and casted
    generator inertia to LSS

:code:`WE_RhoAir` : Float, kg m^-3
    Air density

:code:`PerfFileName` : String
    File containing rotor performance tables (Cp,Ct,Cq) (absolute path
    or relative to this file)

:code:`PerfTableSize` : Float
    Size of rotor performance tables, first number refers to number of
    blade pitch angles, second number referse to number of tip-speed
    ratios

:code:`WE_FOPoles_N` : Float
    Number of first-order system poles used in EKF

:code:`WE_FOPoles_v` : Array of Floats
    Wind speeds corresponding to first-order system poles

:code:`WE_FOPoles` : Array of Floats
    First order system poles

:code:`Y_ErrThresh` : Float, rad^2 s
    Yaw error threshold. Turbine begins to yaw when it passes this

:code:`Y_IPC_IntSat` : Float, rad
    Integrator saturation (maximum signal amplitude contribution to
    pitch from yaw-by-IPC)

:code:`Y_IPC_n` : Float
    Number of controller gains (yaw-by-IPC)

:code:`Y_IPC_KP` : Float
    Yaw-by-IPC proportional controller gain Kp

:code:`Y_IPC_KI` : Float
    Yaw-by-IPC integral controller gain Ki

:code:`Y_IPC_omegaLP` : Float, rad/s.
    Low-pass filter corner frequency for the Yaw-by-IPC controller to
    filtering the yaw alignment error

:code:`Y_IPC_zetaLP` : Float
    Low-pass filter damping factor for the Yaw-by-IPC controller to
    filtering the yaw alignment error.

:code:`Y_MErrSet` : Float, rad
    Yaw alignment error, set point

:code:`Y_omegaLPFast` : Float, rad/s
    Corner frequency fast low pass filter, 1.0

:code:`Y_omegaLPSlow` : Float, rad/s
    Corner frequency slow low pass filter, 1/60

:code:`Y_Rate` : Float, rad/s
    Yaw rate

:code:`FA_KI` : Float, rad s/m
    Integral gain for the fore-aft tower damper controller, -1 = off /
    >0 = on

:code:`FA_HPFCornerFreq` : Float, rad/s
    Corner frequency (-3dB point) in the high-pass filter on the fore-
    aft acceleration signal

:code:`FA_IntSat` : Float, rad
    Integrator saturation (maximum signal amplitude contribution to
    pitch from FA damper)

:code:`PS_BldPitchMin_N` : Float
    Number of values in minimum blade pitch lookup table (should equal
    number of values in PS_WindSpeeds and PS_BldPitchMin)

:code:`PS_WindSpeeds` : Array of Floats
    Wind speeds corresponding to minimum blade pitch angles

:code:`PS_BldPitchMin` : Array of Floats
    Minimum blade pitch angles

:code:`SD_MaxPit` : Float, rad
    Maximum blade pitch angle to initiate shutdown

:code:`SD_CornerFreq` : Float, rad/s
    Cutoff Frequency for first order low-pass filter for blade pitch
    angle

:code:`Fl_n` : Float, s
    Number of Fl_Kp gains in gain scheduling, optional with default of
    1

:code:`Fl_Kp` : Array of Floats
    Nacelle velocity proportional feedback gain

:code:`Fl_U` : Array of Floats
    Wind speeds for scheduling Fl_Kp, optional if Fl_Kp is single
    value [m/s]

:code:`Flp_Angle` : Float, rad
    Initial or steady state flap angle

:code:`Flp_Kp` : Float, s
    Blade root bending moment proportional gain for flap control

:code:`Flp_Ki` : Float
    Flap displacement integral gain for flap control

:code:`Flp_MaxPit` : Float, rad
    Maximum (and minimum) flap pitch angle

:code:`OL_Filename` : String
    Input file with open loop timeseries (absolute path or relative to
    this file)

:code:`Ind_Breakpoint` : Float
    The column in OL_Filename that contains the breakpoint (time if
    OL_Mode > 0)

:code:`Ind_BldPitch` : Float
    The column in OL_Filename that contains the blade pitch input in
    rad

:code:`Ind_GenTq` : Float
    The column in OL_Filename that contains the generator torque in Nm

:code:`Ind_YawRate` : Float
    The column in OL_Filename that contains the generator torque in Nm

:code:`Ind_Azimuth` : Float
    The column in OL_Filename that contains the desired azimuth
    position in rad (used if OL_Mode = 2)

:code:`RP_Gains` : Array of Floats
    PID gains and Tf of derivative for rotor position control (used if
    OL_Mode = 2)

    *Default* = [0, 0, 0, 0]

:code:`Ind_CableControl` : Array of Floats
    The column in OL_Filename that contains the cable control inputs
    in m

:code:`Ind_StructControl` : Array of Floats
    The column in OL_Filename that contains the structural control
    inputs in various units

:code:`DLL_FileName` : String
    Name/location of the dynamic library {.dll [Windows] or .so
    [Linux]} in the Bladed-DLL format

    *Default* = unused

:code:`DLL_InFile` : String
    Name of input file sent to the DLL

    *Default* = unused

:code:`DLL_ProcName` : String
    Name of procedure in DLL to be called

    *Default* = DISCON

:code:`PF_Offsets` : Array of Floats
    Pitch angle offsets for each blade (array with length of 3)

    *Default* = [0, 0, 0]

:code:`CC_Group_N` : Float
    Number of cable control groups

    *Default* = 0

:code:`CC_GroupIndex` : Array of Floats
    First index for cable control group, should correspond to deltaL

    *Default* = [0]

:code:`CC_ActTau` : Float
    Time constant for line actuator [s]

    *Default* = 20

:code:`StC_Group_N` : Float
    Number of cable control groups

    *Default* = 0

:code:`StC_GroupIndex` : Array of Floats
    First index for structural control group, options specified in
    ServoDyn summary output

    *Default* = [0]

:code:`AWC_Mode` : Float
    Active wake control mode {0 - not used, 1 - complex number method,
    2 - Coleman transformation method}

    *Default* = 0

    *Minimum* = 0    *Maximum* = 2


:code:`AWC_NumModes` : Float, rad
    Number of AWC modes

    *Default* = 1

:code:`AWC_n` : Array of Floats
    AWC azimuthal number (only used in complex number method)

    *Default* = [1]

:code:`AWC_harmonic` : Array of Integers
    AWC Coleman transform harmonic (only used in Coleman transform
    method)

    *Default* = [1]

:code:`AWC_freq` : Array of Floats
    AWC frequency [Hz]

    *Default* = [0.05]

:code:`AWC_amp` : Array of Floats
    AWC amplitude [deg]

    *Default* = [1.0]

:code:`AWC_clockangle` : Array of Floats
    AWC clock angle [deg]

    *Default* = [0]

:code:`ZMQ_CommAddress` : String
    Communication address for ZMQ server, (e.g.
    "tcp://localhost:5555")

    *Default* = tcp://localhost:5555

:code:`ZMQ_UpdatePeriod` : Float
    Update period at zmq interface to send measurements and wait for
    setpoint [sec.]

    *Default* = 1.0

:code:`ZMQ_ID` : Float
    Integer identifier of turbine

    *Default* = 0

:code:`tuning_yaml` : String
    yaml file to tune the ROSCO controller, only used for control-only
    optimizations using an OpenFAST model.  Absolute path or relative
    to modeling input.

    *Default* = none



linmodel_tuning
########################################

Inputs used for tuning ROSCO using linear (level 2) models
:code:`type` : String from, ['none', 'robust', 'simulation']
    Type of level 2 based tuning - robust gain scheduling (robust) or
    simulation based optimization (simulation)

    *Default* = none

:code:`linfile_path` : String
    Path to OpenFAST linearization (.lin) files, if they exist

    *Default* = none

:code:`lintune_outpath` : String
    Path for outputs from linear model based tuning

    *Default* = lintune_outfiles

:code:`load_parallel` : Boolean
    Load linearization files in parallel (True/False)

    *Default* = False



OL2CL
****************************************

:code:`flag` : Boolean
    Whether or not to run open loop to closed loop optimization

    *Default* = False

:code:`trajectory_dir` : String
    Directory where open loop control trajectories are located

    *Default* = unused

:code:`save_error` : Boolean
    Save error timeseries?

    *Default* = True

