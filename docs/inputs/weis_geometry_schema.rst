******************************
/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_geometry_schema.yaml
******************************
Ontology definition for wind turbines as defined in WP1 of IEA Wind Task 37 - Phase II


/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_geometry_schema.

:code:`comments` : String
    Description of the model

:code:`name` : String
    Name of the turbine



assembly
****************************************

:code:`turbine_class` : String from, ['I', 'II', 'III', 'IV', 'i', 'ii', 'iii', 'iv', 1, 2, 3, 4]
    IEC wind class of the wind turbine. The options are "I", "II",
    "III", and 'IV'

    *Default* = I

:code:`turbulence_class` : String from, ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']
    IEC turbulence class of the wind turbine. The options are "A",
    "B", and "C"

    *Default* = B

:code:`drivetrain` : String from, ['Geared', 'geared', 'Direct_drive', 'Direct_Drive', 'Direct', 'direct_drive', 'direct', 'pm_direct_drive', 'Constant_eff']
    String characterizing the drivetrain configuration

    *Default* = geared

:code:`rotor_orientation` : String from, ['Upwind', 'upwind', 'UPWIND', 'downwind', 'Downwind', 'DOWNWIND']
    Orientation of the horizontal-axis rotor. The options are "Upwind"
    and "Downwind"

    *Default* = Upwind

:code:`number_of_blades` : Integer
    Number of blades of the rotor

    *Default* = 3

    *Minimum* = 0    *Maximum* = 10


:code:`rotor_diameter` : Float, m
    Diameter of the rotor, defined as two times the projected blade
    length plus the hub diameter

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1000


:code:`hub_height` : Float, m
    Height of the hub center over the ground (land-based) or the mean
    sea level (offshore)

    *Default* = 0

    *Minimum* = 0    *Maximum* = 1000


:code:`rated_power` : Float, W
    Nameplate power of the turbine, i.e. the rated electrical output
    of the generator.

    *Minimum* = 0

:code:`lifetime` : Float, years
    Turbine design lifetime in years.

    *Default* = 25.0

    *Minimum* = 0



components
****************************************



blade
########################################



outer_shape_bem
========================================



airfoil_position
----------------------------------------



chord
----------------------------------------



twist
----------------------------------------



pitch_axis
----------------------------------------



rthick
----------------------------------------



L/D
----------------------------------------



c_d
----------------------------------------



stall_margin
----------------------------------------



elastic_properties_mb
========================================



internal_structure_2d_fem
========================================



root
----------------------------------------

:code:`d_f` : Float, m
    Diameter of the fastener, default is M30, so 0.03 meters

    *Default* = 0.03

    *Minimum* = 0.01    *Maximum* = 0.2


:code:`sigma_max` : Float, Pa
    Max stress on bolt

    *Default* = 675000000.0

    *Minimum* = 100000.0    *Maximum* = 10000000000.0




webs
----------------------------------------

:code:`name` : String
    structural component identifier



layers
----------------------------------------

:code:`name` : String
    structural component identifier

:code:`material` : String
    material identifier

:code:`web` : String
    web to which the layer is associated to, only to be defined for
    web layers



thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

thickness of the laminate


n_plies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

number of plies of the laminate


fiber_orientation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

orientation of the fibers


width
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dimensional width of the component along the arc


joint
----------------------------------------

This is a spanwise joint along the blade, usually adopted to ease transportation constraints. WISDEM currently supports a single joint.
:code:`position` : Float
    Spanwise position of the segmentation joint.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`mass` : Float, kg
    Mass of the joint.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`cost` : Float, USD
    Cost of the joint.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`bolt` : String from, ['M18', 'M24', 'M30', 'M36', 'M42', 'M48', 'M52']
    Bolt size for the blade bolted joint

    *Default* = M30

:code:`nonmaterial_cost` : Float, USD
    Cost of the joint not from materials.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`reinforcement_layer_ss` : String
    Layer identifier for the joint reinforcement on the suction side

    *Default* = joint_reinf_ss

:code:`reinforcement_layer_ps` : String
    Layer identifier for the joint reinforcement on the pressure side

    *Default* = joint_reinf_ps



hub
########################################

:code:`diameter` : Float, meter
    Diameter of the hub measured at the blade root positions.

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`cone_angle` : Float, rad
    Rotor precone angle, defined positive for both upwind and downwind
    rotors.

    *Minimum* = 0    *Maximum* = 0.4


:code:`drag_coefficient` : Float
    Equivalent drag coefficient to compute the aerodynamic forces
    generated on the hub.

    *Default* = 0.5

    *Minimum* = 0    *Maximum* = 2.0


:code:`flange_t2shell_t` : Float
    Ratio of flange thickness to shell thickness

    *Default* = 6.0

    *Minimum* = 0    *Maximum* = 20.0


:code:`flange_OD2hub_D` : Float
    Ratio of flange outer diameter to hub diameter

    *Default* = 0.6

    *Minimum* = 0    *Maximum* = 10.0


:code:`flange_ID2OD` : Float
    Check this

    *Default* = 0.8

    *Minimum* = 0    *Maximum* = 10.0


:code:`hub_blade_spacing_margin` : Float
    Ratio of flange thickness to shell thickness

    *Default* = 1.2

    *Minimum* = 0    *Maximum* = 20.0


:code:`hub_stress_concentration` : Float
    Stress concentration factor. Stress concentration occurs at all
    fillets,notches, lifting lugs, hatches and are accounted for by
    assigning a stress concentration factor

    *Default* = 3.0

    *Minimum* = 0    *Maximum* = 20.0


:code:`n_front_brackets` : Integer
    Number of front spinner brackets

    *Default* = 5

    *Minimum* = 0    *Maximum* = 20


:code:`n_rear_brackets` : Integer
    Number of rear spinner brackets

    *Default* = 5

    *Minimum* = 0    *Maximum* = 20


:code:`clearance_hub_spinner` : Float, m
    Clearance between spinner and hub

    *Default* = 0.5

    *Minimum* = 0    *Maximum* = 20.0


:code:`spin_hole_incr` : Float
    Ratio between access hole diameter in the spinner and blade root
    diameter. Typical value 1.2

    *Default* = 1.2

    *Minimum* = 0    *Maximum* = 20.0


:code:`pitch_system_scaling_factor` : Float
    Scaling factor to tune the total mass (0.54 is recommended for
    modern designs)

    *Default* = 0.54

    *Minimum* = 0    *Maximum* = 2.0


:code:`hub_material` : String
    Material of the shell of the hub

:code:`spinner_material` : String
    Material of the spinner



elastic_properties_mb
========================================

:code:`system_mass` : Float, kg
    Mass of the hub system, which includes the hub, the spinner, the
    blade bearings, the pitch actuators, the cabling, ....

    *Minimum* = 0

:code:`system_inertia` : Array of Floats, kgm2
    Inertia of the hub system, on the hub reference system, which has
    the x aligned with the rotor axis, and y and z perpendicular to
    it.

:code:`system_center_mass` : Array of Floats, m
    Center of mass of the hub system. Work in progress.



nacelle
########################################



drivetrain
========================================

Inputs to WISDEM specific drivetrain sizing tool, DrivetrainSE
:code:`uptilt` : Float, rad
    Tilt angle of the nacelle, always defined positive.

    *Default* = 0.08726

    *Minimum* = 0.0    *Maximum* = 0.2


:code:`distance_tt_hub` : Float, meter
    Vertical distance between the tower top and the hub center.

    *Default* = 2.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`distance_hub_mb` : Float, meter
    Distance from hub flange to first main bearing along shaft.

    *Default* = 2.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`distance_mb_mb` : Float, meter
    Distance from first to second main bearing along shaft.

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`overhang` : Float, meter
    Horizontal distance between the tower axis and the rotor apex.

    *Default* = 5.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`generator_length` : Float, meter
    Length of generator along the shaft

    *Default* = 2.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`generator_radius_user` : Float, m
    User input override of generator radius, only used when using
    simple generator scaling

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`generator_mass_user` : Float, kg
    User input override of generator mass, only used when using simple
    generator mass scaling

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000000000.0




generator_rpm_efficiency_user
----------------------------------------

User input override of generator rpm-efficiency values, with rpm as grid input and eff as values input
:code:`gear_ratio` : Float
    Gear ratio of the drivetrain. Set it to 1 for direct drive
    machines.

    *Default* = 1.0

    *Minimum* = 1    *Maximum* = 1000


:code:`gearbox_length_user` : Float, meter
    User input override of gearbox length along shaft, only used when
    using gearbox_mass_user is > 0

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`gearbox_radius_user` : Float, m
    User input override of gearbox radius, only used when using
    gearbox_mass_user is > 0

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 20.0


:code:`gearbox_mass_user` : Float, kg
    User input override of gearbox mass

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000000000.0


:code:`gearbox_efficiency` : Float
    Efficiency of the gearbox system.

    *Default* = 1.0

    *Minimum* = 0.8    *Maximum* = 1.0


:code:`damping_ratio` : Float
    Damping ratio for the drivetrain system

    *Default* = 0.005

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`lss_diameter` : Array of Floats, m
    Diameter of the low speed shaft at beginning (generator/gearbox)
    and end (hub) points

    *Default* = [0.3, 0.3]

:code:`lss_wall_thickness` : Array of Floats, m
    Thickness of the low speed shaft at beginning (generator/gearbox)
    and end (hub) points

    *Default* = [0.1, 0.1]

:code:`lss_material` : String
    Material name identifier

    *Default* = steel

:code:`hss_length` : Float, meter
    Length of the high speed shaft

    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`hss_diameter` : Array of Floats, m
    Diameter of the high speed shaft at beginning (generator) and end
    (generator) points

    *Default* = [0.3, 0.3]

:code:`hss_wall_thickness` : Array of Floats, m
    Thickness of the high speed shaft at beginning (generator) and end
    (generator) points

    *Default* = [0.1, 0.1]

:code:`hss_material` : String
    Material name identifier

    *Default* = steel

:code:`nose_diameter` : Array of Floats, m
    Diameter of the nose/turret at beginning (bedplate) and end (main
    bearing) points

    *Default* = [0.3, 0.3]

:code:`nose_wall_thickness` : Array of Floats, m
    Thickness of the nose/turret at beginning (bedplate) and end (main
    bearing) points

    *Default* = [0.1, 0.1]



bedplate_wall_thickness
----------------------------------------

Thickness of the hollow elliptical bedplate used in direct drive configurations
:code:`bedplate_flange_width` : Float, meter
    Bedplate I-beam flange width used in geared configurations

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 3.0


:code:`bedplate_flange_thickness` : Float, meter
    Bedplate I-beam flange thickness used in geared configurations

    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`bedplate_web_thickness` : Float, meter
    Bedplate I-beam web thickness used in geared configurations

    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`brake_mass_user` : Float, kg
    Override regular regression-based calculation of brake mass with
    this value

    *Default* = 0.0

    *Minimum* = 0.0

:code:`hvac_mass_coefficient` : Float, kg/kW
    Regression-based scaling coefficient on machine rating to get HVAC
    system mass

    *Default* = 0.025

    *Minimum* = 0.0

:code:`converter_mass_user` : Float, kg
    Override regular regression-based calculation of converter mass
    with this value

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transformer_mass_user` : Float, kg
    Override regular regression-based calculation of transformer mass
    with this value

    *Default* = 0.0

    *Minimum* = 0.0

:code:`bedplate_material` : String
    Material name identifier

    *Default* = steel

:code:`mb1Type` : String from, ['CARB', 'CRB', 'SRB', 'TRB']
    Type of bearing for first main bearing

    *Default* = CARB

:code:`mb2Type` : String from, ['CARB', 'CRB', 'SRB', 'TRB']
    Type of bearing for second main bearing

    *Default* = SRB

:code:`uptower` : Boolean
    If power electronics are located uptower (True) or at tower base
    (False)

    *Default* = True

:code:`gear_configuration` : String
    3-letter string of Es or Ps to denote epicyclic or parallel gear
    configuration

    *Default* = EEP

:code:`planet_numbers` : Array of Integers
    Number of planets for epicyclic stages (use 0 for parallel)

    *Default* = [3, 3, 0]

    *Minimum* = 0

    *Maximum* = 6



elastic_properties_mb
========================================

:code:`system_mass` : Float, kg
    Mass of the nacelle system, including the entire drivetrain system
    (shafts, gearbox if present, break, bearings, generator). It
    excludes the turbine rotor, the hub, and the yaw system.

    *Minimum* = 0

:code:`yaw_mass` : Float, kg
    Mass of the yaw system.

    *Minimum* = 0

:code:`system_inertia` : Array of Floats, kgm2
    Inertia of the nacelle system with respect to the center of mass.
    The sum includes the entire drivetrain system (shafts, gearbox if
    present, break, bearings, generator). It excludes the turbine
    rotor, the hub, and the yaw system.

:code:`system_inertia_tt` : Array of Floats, kgm2
    Inertia of the nacelle system with respect to the tower top. The
    sum includes the entire drivetrain system (shafts, gearbox if
    present, break, bearings, generator). It excludes the turbine
    rotor, the hub, and the yaw system.

:code:`system_center_mass` : Array of Floats, m
    Center of mass of the nacelle system, including the entire
    drivetrain system (shafts, gearbox if present, break, bearings,
    generator). It excludes the turbine rotor, the hub, and the yaw
    system.



tower
########################################



outer_shape_bem
========================================



outer_diameter
----------------------------------------



drag_coefficient
----------------------------------------



elastic_properties_mb
========================================



internal_structure_2d_fem
========================================

:code:`outfitting_factor` : Float
    Scaling factor for the tower mass to account for auxiliary
    structures, such as elevator, ladders, cables, platforms, etc

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 2.0




layers
----------------------------------------

:code:`name` : String
    structural component identifier

:code:`material` : String
    material identifier



thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

thickness of the laminate


monopile
########################################

:code:`transition_piece_mass` : Float, kg
    Total mass of transition piece

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transition_piece_cost` : Float, USD
    Total cost of transition piece

    *Default* = 0.0

    *Minimum* = 0.0

:code:`gravity_foundation_mass` : Float, kg
    Total mass of gravity foundation addition onto monopile

    *Default* = 0.0

    *Minimum* = 0.0



outer_shape
========================================



outer_diameter
----------------------------------------



drag_coefficient
----------------------------------------



elastic_properties_mb
========================================



internal_structure_2d_fem
========================================

:code:`outfitting_factor` : Float
    Scaling factor for the tower mass to account for auxiliary
    structures, such as elevator, ladders, cables, platforms, etc

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 2.0




layers
----------------------------------------

:code:`name` : String
    structural component identifier

:code:`material` : String
    material identifier



thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

thickness of the laminate


jacket
########################################

:code:`transition_piece_mass` : Float, kg
    Total mass of transition piece

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transition_piece_cost` : Float, USD
    Total cost of transition piece

    *Default* = 0.0

    *Minimum* = 0.0

:code:`gravity_foundation_mass` : Float, kg
    Total mass of gravity foundation addition onto monopile

    *Default* = 0.0

    *Minimum* = 0.0

:code:`material` : String
    Material of jacket members

    *Default* = steel

:code:`n_bays` : Integer
    Number of bays (x-joints) in the vertical direction for jackets.

:code:`n_legs` : Integer
    Number of legs for jacket.

:code:`r_foot` : Float
    Radius of foot (bottom) of jacket, in meters.

:code:`r_head` : Float
    Radius of head (top) of jacket, in meters.

:code:`height` : Float
    Overall jacket height, meters.

:code:`leg_thickness` : Float
    Leg thickness, meters. Constant throughout each leg.

:code:`x_mb` : Boolean
    Mud brace included if true.

:code:`leg_diameter` : Float
    Leg diameter, meters. Constant throughout each leg.



floating_platform
########################################

Ontology definition for floating platforms (substructures) suitable for use with the WEIS co-design analysis tool


joints
========================================

:code:`name` : String
    Unique name of the joint (node)

:code:`location` : Array of Floats, m
    Coordinates (x,y,z or r,θ,z) of the joint in the global coordinate
    system.

:code:`transition` : Boolean
    Whether the transition piece and turbine tower attach at this node

    *Default* = False

:code:`cylindrical` : Boolean
    Whether to use cylindrical coordinates (r,θ,z), with (r,θ) lying
    in the x/y-plane, instead of Cartesian coordinates.

    *Default* = False



reactions
----------------------------------------

If this joint is compliant is certain DOFs, then specify which are compliant (True) in the member/element coordinate system).  If not specified, default is all entries are False (completely rigid).  For instance, a ball joint would be Rx=Ry=Rz=False, Rxx=Ryy=Rzz=True
:code:`Rx` : Boolean


    *Default* = False

:code:`Ry` : Boolean


    *Default* = False

:code:`Rz` : Boolean


    *Default* = False

:code:`Rxx` : Boolean


    *Default* = False

:code:`Ryy` : Boolean


    *Default* = False

:code:`Rzz` : Boolean


    *Default* = False

:code:`Euler` : Array of Floats
    Euler angles [alpha, beta, gamma] that describe the rotation of
    the Reaction coordinate system relative to the global coordinate
    system α is a rotation around the z axis, β is a rotation around
    the x' axis, γ is a rotation around the z" axis.



members
========================================

:code:`name` : String
    Name of the member

:code:`joint1` : String
    Name of joint/node connection

:code:`joint2` : String
    Name of joint/node connection



outer_shape
----------------------------------------

:code:`shape` : String from, ['circular', 'polygonal']
    Specifies cross-sectional shape of the member.  If circular, then
    the outer_diameter field is required.  If polygonal, then the
    side_lengths, angles, and rotation fields are required



outer_diameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gridded values describing diameter at non-dimensional axis from joint1 to joint2
:code:`side_lengths1` : Array of Floats, m
    Polygon side lengths at joint1

    *Minimum* = 0

:code:`side_lengths2` : Array of Floats, m
    Polygon side lengths at joint1

    *Minimum* = 0

:code:`angles` : Array of Floats, rad
    Polygon angles with the ordering such that angle[i] is between
    side_length[i] and side_length[i+1]

    *Minimum* = 0

:code:`rotation` : Float, rad
    Angle between principle axes of the cross-section and the member
    coordinate system.  Essentially the rotation of the member if both
    joints were placed on the global x-y axis with the first side
    length along the z-axis



internal_structure
----------------------------------------

:code:`outfitting_factor` : Float
    Scaling factor for the member mass to account for auxiliary
    structures, such as elevator, ladders, cables, platforms,
    fasteners, etc

    *Default* = 1.0

    *Minimum* = 1.0



layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`name` : String
    structural component identifier

:code:`material` : String
    material identifier



thickness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gridded values describing thickness along non-dimensional axis from joint1 to joint2


ring_stiffeners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`material` : String
    material identifier

:code:`flange_thickness` : Float, m


    *Minimum* = 0

:code:`flange_width` : Float, m


    *Minimum* = 0

:code:`web_height` : Float, m


    *Minimum* = 0

:code:`web_thickness` : Float, m


    *Minimum* = 0



longitudinal_stiffeners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`material` : String
    material identifier

:code:`flange_thickness` : Float, m


    *Minimum* = 0

:code:`flange_width` : Float, m


    *Minimum* = 0

:code:`web_height` : Float, m


    *Minimum* = 0

:code:`web_thickness` : Float, m


    *Minimum* = 0



bulkhead
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`material` : String
    material identifier



thickness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

thickness of the bulkhead at non-dimensional locations of the member [0..1]


ballast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`variable_flag` : Boolean
    If true, then this ballast is variable and adjusted by control
    system.  If false, then considered permanent

:code:`material` : String
    material identifier

:code:`volume` : Float, m^3
    Total volume of ballast (permanent ballast only)

    *Minimum* = 0



axial_joints
----------------------------------------

:code:`name` : String
    Unique name of joint

:code:`grid` : Float
    Non-dimensional value along member axis

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`Ca` : Float
    User-defined added mass coefficient

    *Default* = 0.0

    *Minimum* = 0.0

:code:`Cp` : Float
    User-defined pressure coefficient

    *Default* = 0.0

:code:`Cd` : Float
    User-defined drag coefficient

    *Default* = 0.0

    *Minimum* = 0.0



rigid_bodies
========================================

:code:`joint1` : String
    Name of joint/node connection

:code:`mass` : Float, kg
    Mass of this rigid body

    *Minimum* = 0

:code:`cost` : Float, USD
    Cost of this rigid body

    *Minimum* = 0

:code:`cm_offset` : Array of Floats, m
    Offset from joint location to center of mass (CM) of body in dx,
    dy, dz

:code:`moments_of_inertia` : Array of Floats, kg*m^2
    Moments of inertia around body CM in Ixx, Iyy, Izz

    *Minimum* = 0

:code:`Ca` : Float
    User-defined added mass coefficient

    *Default* = 0.0

    *Minimum* = 0.0

:code:`Cp` : Float
    User-defined pressure coefficient

    *Default* = 0.0

:code:`Cd` : Float
    User-defined drag coefficient

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transition_piece_mass` : Float, kg
    Total mass of transition piece

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transition_piece_cost` : Float, USD
    Total cost of transition piece

    *Default* = 0.0

    *Minimum* = 0.0



mooring
########################################

Ontology definition for mooring systems suitable for use with the WEIS co-design analysis tool


nodes
========================================

:code:`name` : String
    Name or ID of this node for use in line segment

:code:`node_type` : String from, ['fixed', 'fix', 'connection', 'connect', 'free', 'vessel']


:code:`location` : Array of Floats, meter
    – Coordinates x, y, and z of the connection (relative to inertial
    reference frame if Fixed or Connect, relative to platform
    reference frame if Vessel). In the case of Connect nodes, it is
    simply an initial guess for position before MoorDyn calculates the
    equilibrium initial position.

:code:`joint` : String
    For anchor positions and fairlead attachments, reference a joint
    name from the "joints" section or an "axial_joint" on a member

    *Default* = none

:code:`anchor_type` : String
    Name of anchor type from anchor_type list

    *Default* = none

:code:`fairlead_type` : String from, ['rigid', 'actuated', 'ball']


    *Default* = rigid

:code:`node_mass` : Float, kilogram
    Clump weight mass

    *Default* = 0.0

    *Minimum* = 0.0

:code:`node_volume` : Float, meter^3
    Floater volume

    *Default* = 0.0

    *Minimum* = 0.0

:code:`drag_area` : Float, meter^2
    Product of drag coefficient and projected area (assumed constant
    in all directions) to calculate a drag force for the node

    *Default* = 0.0

    *Minimum* = 0.0

:code:`added_mass` : Float
    Added mass coefficient used along with node volume to calculate
    added mass on node

    *Default* = 0.0



lines
========================================

:code:`name` : String
    ID of this line

:code:`line_type` : String
    Reference to line type database

:code:`unstretched_length` : Float, meter
    length of line segment prior to tensioning

    *Minimum* = 0.0

:code:`node1` : String
    node id of first line connection

:code:`node2` : String
    node id of second line connection



line_types
========================================

:code:`name` : String
    Name of material or line type to be referenced by line segments

:code:`diameter` : Float, meter
    the volume-equivalent diameter of the line – the diameter of a
    cylinder having the same displacement per unit length

    *Minimum* = 0.0

:code:`type` : String from, ['chain', 'chain_stud', 'nylon', 'polyester', 'polypropylene', 'wire_fiber', 'fiber', 'wire', 'wire_wire', 'iwrc', 'Chain', 'Chain_Stud', 'Nylon', 'Polyester', 'Polypropylene', 'Wire', 'Wire_Fiber', 'Fiber', 'Wire', 'Wire_Wire', 'IWRC', 'CHAIN', 'CHAIN_STUD', 'NYLON', 'POLYESTER', 'POLYPROPYLENE', 'WIRE', 'WIRE_FIBER', 'FIBER', 'WIRE', 'WIRE_WIRE', 'custom', 'Custom', 'CUSTOM']
    Type of material for property lookup

:code:`mass_density` : Float, kilogram/meter
    mass per unit length (in air)

    *Minimum* = 0.0

:code:`stiffness` : Float, Newton
    axial line stiffness, product of elasticity modulus and cross-
    sectional area

    *Minimum* = 0.0

:code:`cost` : Float, USD/meter
    cost per unit length

    *Minimum* = 0.0

:code:`breaking_load` : Float, Newton
    line break tension

    *Minimum* = 0.0

:code:`damping` : Float, Newton * second
    internal damping (BA)

    *Default* = 0.0

:code:`transverse_added_mass` : Float
    transverse added mass coefficient (with respect to line
    displacement)

    *Default* = 0.0

    *Minimum* = 0.0

:code:`tangential_added_mass` : Float
    tangential added mass coefficient (with respect to line
    displacement)

    *Default* = 0.0

    *Minimum* = 0.0

:code:`transverse_drag` : Float
    transverse drag coefficient (with respect to frontal area, d*l)

    *Default* = 0.0

    *Minimum* = 0.0

:code:`tangential_drag` : Float
    tangential drag coefficient (with respect to surface area, π*d*l)

    *Default* = 0.0

    *Minimum* = 0.0



anchor_types
========================================

:code:`name` : String
    Name of anchor to be referenced by anchor_id in Nodes section

:code:`type` : String from, ['drag_embedment', 'suction', 'plate', 'micropile', 'sepla', 'Drag_Embedment', 'Suction', 'Plate', 'Micropile', 'Sepla', 'DRAG_EMBEDMENT', 'SUCTION', 'PLATE', 'MICROPILE', 'SEPLA', 'custom', 'Custom', 'CUSTOM']
    Type of anchor for property lookup

:code:`mass` : Float, kilogram
    mass of the anchor

    *Minimum* = 0.0

:code:`cost` : Float, USD
    cost of the anchor

    *Minimum* = 0.0

:code:`max_lateral_load` : Float, Newton
    Maximum lateral load (parallel to the sea floor) that the anchor
    can support

    *Minimum* = 0.0

:code:`max_vertical_load` : Float, Newton
    Maximum vertical load (perpendicular to the sea floor) that the
    anchor can support

    *Minimum* = 0.0



airfoils
****************************************

:code:`name` : String
    Name of the airfoil



coordinates
########################################

Airfoil coordinates described from trailing edge (x=1) along the suction side (y>0) to leading edge (x=0) back to trailing edge (x=1) along the pressure side (y<0)
:code:`x` : Array of Floats


    *Minimum* = 0.0

    *Maximum* = 1.0

:code:`y` : Array of Floats


    *Minimum* = -1.0

    *Maximum* = 1.0

:code:`relative_thickness` : Float
    Thickness of the airfoil expressed non-dimensional

    *Minimum* = 0    *Maximum* = 1


:code:`aerodynamic_center` : Float
    Non-dimensional chordwise coordinate of the aerodynamic center

    *Minimum* = 0    *Maximum* = 1




polars
########################################

Lift, drag and moment coefficients expressed in terms of angles of attack
:code:`configuration` : String
    Text to identify the setup for the definition of the polars

:code:`re` : Float
    Reynolds number of the polars



c_l
========================================



c_d
========================================



c_m
========================================



materials
****************************************

:code:`name` : String
    Name of the material

:code:`description` : String
    Optional field describing the material

:code:`source` : String
    Optional field describing where the data come from

:code:`orth` : Integer
    Flag to switch between isotropic (0) and orthotropic (1) materials

:code:`rho` : Float, kg/m3
    Density of the material. For composites, this is the density of
    the laminate once cured

    *Minimum* = 0    *Maximum* = 20000


:code:`ply_t` : Float, m
    Ply thickness of the composite material

    *Minimum* = 0    *Maximum* = 0.1


:code:`unit_cost` : Float, USD/kg
    Unit cost of the material. For composites, this is the unit cost
    of the dry fabric.

    *Minimum* = 0    *Maximum* = 1000


:code:`fvf` : Float
    Fiber volume fraction of the composite material

    *Minimum* = 0    *Maximum* = 1


:code:`fwf` : Float
    Fiber weight fraction of the composite material

    *Minimum* = 0    *Maximum* = 1


:code:`fiber_density` : Float, kg/m3
    Density of the fibers of a composite material.

    *Minimum* = 0    *Maximum* = 10000


:code:`area_density_dry` : Float, kg/m2
    Aerial density of a fabric of a composite material.

    *Minimum* = 0    *Maximum* = 10000


:code:`component_id` : Integer
    Flag used by the NREL blade cost model
    https://www.nrel.gov/docs/fy19osti/73585.pdf to define the
    manufacturing process behind the laminate. 0 - coating, 1 -
    sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5
    - TE reinf.

:code:`waste` : Float
    Fraction of material that ends up wasted during manufacturing.
    This quantity is used in the NREL blade cost model
    https://www.nrel.gov/docs/fy19osti/73585.pdf

    *Minimum* = 0    *Maximum* = 1


:code:`roll_mass` : Float, kg
    Mass of a fabric roll. This quantity is used in the NREL blade
    cost model https://www.nrel.gov/docs/fy19osti/73585.pdf

    *Minimum* = 0    *Maximum* = 10000


:code:`GIc` : Float, J/m^2
    Mode 1 critical energy-release rate. It is used by NuMAD from
    Sandia National Laboratories

:code:`GIIc` : Float, J/m^2
    Mode 2 critical energy-release rate. It is used by NuMAD from
    Sandia National Laboratories

:code:`alp0` : Float, rad
    Fracture angle under pure transverse compression. It is used by
    NuMAD from Sandia National Laboratories



control
****************************************



supervisory
########################################

:code:`Vin` : Float, m/s
    Cut-in wind speed of the wind turbine.

    *Minimum* = 0    *Maximum* = 10


:code:`Vout` : Float, m/s
    Cut-out wind speed of the wind turbine.

    *Minimum* = 0    *Maximum* = 50


:code:`maxTS` : Float, m/s
    Maximum allowable blade tip speed.

    *Minimum* = 60    *Maximum* = 120




pitch
########################################

:code:`min_pitch` : Float, rad
    Minimum pitch angle, where the default is 0 degrees. It is used by
    the ROSCO controller (https://github.com/NREL/ROSCO)

    *Default* = 0

    *Minimum* = -0.5    *Maximum* = 1.0


:code:`max_pitch_rate` : Float, rad/s
    Maximum pitch rate of the rotor blades.

    *Minimum* = 0    *Maximum* = 0.2




torque
########################################

:code:`max_torque_rate` : Float, Nm/s
    Maximum torque rate of the wind turbine generator.

    *Minimum* = 1000    *Maximum* = 100000000


:code:`tsr` : Float
    Rated tip speed ratio of the wind turbine. As default, it is
    maintained constant in region II.

    *Minimum* = 0    *Maximum* = 15


:code:`VS_minspd` : Float, rad/s
    Minimum rotor speed. It is used by the ROSCO controller
    (https://github.com/NREL/ROSCO)

    *Minimum* = 0    *Maximum* = 5


:code:`VS_maxspd` : Float, rad/s
    Maximum rotor speed. It is used by the ROSCO controller
    (https://github.com/NREL/ROSCO)

    *Default* = 10.0

    *Minimum* = 0



environment
****************************************

:code:`gravity` : Float, m/s/s
    Gravitational acceleration

    *Default* = 9.80665

    *Minimum* = 0    *Maximum* = 100.0


:code:`air_density` : Float, kg/m3
    Density of air.

    *Default* = 1.225

    *Minimum* = 0    *Maximum* = 1.5


:code:`air_dyn_viscosity` : Float, kg/(ms)
    Dynamic viscosity of air.

    *Default* = 1.81e-05

    *Minimum* = 0    *Maximum* = 2e-05


:code:`air_pressure` : Float, kg/(ms^2)
    Atmospheric pressure of air

    *Default* = 103500.0

    *Minimum* = 0    *Maximum* = 1000000.0


:code:`air_vapor_pressure` : Float, kg/(ms^2)
    Vapor pressure of fluid

    *Default* = 1700.0

    *Minimum* = 0    *Maximum* = 1000000.0


:code:`weib_shape_parameter` : Float
    Shape factor of the Weibull wind distribution.

    *Default* = 2.0

    *Minimum* = 1    *Maximum* = 3


:code:`air_speed_sound` : Float, m/s
    Speed of sound in air.

    *Default* = 340.0

    *Minimum* = 330.0    *Maximum* = 350.0


:code:`shear_exp` : Float
    Shear exponent of the atmospheric boundary layer.

    *Default* = 0.2

    *Minimum* = 0    *Maximum* = 1


:code:`water_density` : Float, kg/m3
    Density of water.

    *Default* = 1025.0

    *Minimum* = 950    *Maximum* = 1100


:code:`water_dyn_viscosity` : Float, kg/(ms)
    Dynamic viscosity of water.

    *Default* = 0.0013351

    *Minimum* = 0.001    *Maximum* = 0.002


:code:`water_depth` : Float, m
    Water depth for offshore environment.

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`soil_shear_modulus` : Float, Pa
    Shear modulus of the soil.

    *Default* = 140000000.0

    *Minimum* = 100000000.0    *Maximum* = 200000000.0


:code:`soil_poisson` : Float
    Poisson ratio of the soil.

    *Default* = 0.4

    *Minimum* = 0    *Maximum* = 0.6


:code:`V_mean` : Float
    Average inflow wind speed. If different than 0, this will
    overwrite the V mean of the IEC wind class

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 20.0




bos
****************************************

:code:`plant_turbine_spacing` : Float
    Distance between turbines in the primary grid streamwise direction
    in rotor diameters

    *Default* = 7

    *Minimum* = 1    *Maximum* = 100


:code:`plant_row_spacing` : Float
    Distance between turbine rows in the cross-wind direction in rotor
    diameters

    *Default* = 7

    *Minimum* = 1    *Maximum* = 100


:code:`commissioning_pct` : Float
    Fraction of total BOS cost that is due to commissioning

    *Default* = 0.01

    *Minimum* = 0    *Maximum* = 1


:code:`decommissioning_pct` : Float
    Fraction of total BOS cost that is due to decommissioning

    *Default* = 0.15

    *Minimum* = 0    *Maximum* = 1


:code:`distance_to_substation` : Float, km
    Distance from centroid of plant to substation in km

    *Default* = 2

    *Minimum* = 0    *Maximum* = 1000


:code:`distance_to_interconnection` : Float, km
    Distance from substation to grid connection in km

    *Default* = 50

    *Minimum* = 0    *Maximum* = 1000


:code:`distance_to_landfall` : Float, km
    Distance from plant centroid to export cable landfall for offshore
    plants

    *Default* = 100

    *Minimum* = 0    *Maximum* = 1000


:code:`distance_to_site` : Float, km
    Distance from port to plant centroid for offshore plants

    *Default* = 100

    *Minimum* = 0    *Maximum* = 1000


:code:`interconnect_voltage` : Float, kV
    Voltage of cabling to grid interconnection

    *Default* = 130

    *Minimum* = 0    *Maximum* = 1000


:code:`port_cost_per_month` : Float, USD
    Monthly port rental fees

    *Default* = 2000000.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`site_auction_price` : Float, USD
    Cost to secure site lease

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`site_assessment_plan_cost` : Float, USD
    Cost to do engineering plan for site assessment

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`site_assessment_cost` : Float, USD
    Cost to execute site assessment

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`construction_operations_plan_cost` : Float, USD
    Cost to do construction planning

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`boem_review_cost` : Float, USD
    Cost for additional review by U.S. Dept of Interior Bureau of
    Ocean Energy Management (BOEM)

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0


:code:`design_install_plan_cost` : Float, USD
    Cost to do installation planning

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000000000.0




costs
****************************************

:code:`wake_loss_factor` : Float
    Factor to model losses in annual energy production in a wind farm
    compared to the annual energy production at the turbine level
    (wakes mostly).

    *Default* = 0.15

    *Minimum* = 0    *Maximum* = 1


:code:`fixed_charge_rate` : Float
    Fixed charge rate to compute the levelized cost of energy. See
    this for inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf

    *Default* = 0.075

    *Minimum* = 0    *Maximum* = 1


:code:`bos_per_kW` : Float, USD/kW
    Balance of stations costs expressed in USD per kW. See this for
    inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 10000


:code:`opex_per_kW` : Float, USD/kW
    Operational expenditures expressed in USD per kW. See this for
    inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf

    *Default* = 0.0

    *Minimum* = 0    *Maximum* = 1000


:code:`turbine_number` : Integer
    Number of turbines in the park, used to compute levelized cost of
    energy. Often wind parks are assumed of 600 MW. See this for
    inspiration https://www.nrel.gov/docs/fy20osti/74598.pdf

    *Default* = 50

    *Minimum* = 0    *Maximum* = 10000


:code:`labor_rate` : Float, USD/h
    Hourly loaded wage per worker including all benefits and overhead.
    This is currently only applied to steel, column structures.

    *Default* = 58.8

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`painting_rate` : Float, USD/m^2
    Cost per unit area for finishing and surface treatments.  This is
    currently only applied to steel, column structures.

    *Default* = 30.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`blade_mass_cost_coeff` : Float, USD/kg
    Regression-based blade cost/mass ratio

    *Default* = 14.6

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`hub_mass_cost_coeff` : Float, USD/kg
    Regression-based hub cost/mass ratio

    *Default* = 3.9

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`pitch_system_mass_cost_coeff` : Float, USD/kg
    Regression-based pitch system cost/mass ratio

    *Default* = 22.1

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`spinner_mass_cost_coeff` : Float, USD/kg
    Regression-based spinner cost/mass ratio

    *Default* = 11.1

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`lss_mass_cost_coeff` : Float, USD/kg
    Regression-based low speed shaft cost/mass ratio

    *Default* = 11.9

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`bearing_mass_cost_coeff` : Float, USD/kg
    Regression-based bearing cost/mass ratio

    *Default* = 4.5

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`gearbox_mass_cost_coeff` : Float, USD/kg
    Regression-based gearbox cost/mass ratio

    *Default* = 12.9

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`hss_mass_cost_coeff` : Float, USD/kg
    Regression-based high speed side cost/mass ratio

    *Default* = 6.8

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`generator_mass_cost_coeff` : Float, USD/kg
    Regression-based generator cost/mass ratio

    *Default* = 12.4

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`bedplate_mass_cost_coeff` : Float, USD/kg
    Regression-based bedplate cost/mass ratio

    *Default* = 2.9

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`yaw_mass_cost_coeff` : Float, USD/kg
    Regression-based yaw system cost/mass ratio

    *Default* = 8.3

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`converter_mass_cost_coeff` : Float, USD/kg
    Regression-based converter cost/mass ratio

    *Default* = 18.8

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`transformer_mass_cost_coeff` : Float, USD/kg
    Regression-based transformer cost/mass ratio

    *Default* = 18.8

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`hvac_mass_cost_coeff` : Float, USD/kg
    Regression-based HVAC system cost/mass ratio

    *Default* = 124.0

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`cover_mass_cost_coeff` : Float, USD/kg
    Regression-based nacelle cover cost/mass ratio

    *Default* = 5.7

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`elec_connec_machine_rating_cost_coeff` : Float, USD/kW
    Regression-based electrical plant connection cost/rating ratio

    *Default* = 41.85

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`platforms_mass_cost_coeff` : Float, USD/kg
    Regression-based nacelle platform cost/mass ratio

    *Default* = 17.1

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`tower_mass_cost_coeff` : Float, USD/kg
    Regression-based tower cost/mass ratio

    *Default* = 2.9

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`controls_machine_rating_cost_coeff` : Float, USD/kW
    Regression-based controller and sensor system cost/rating ratio

    *Default* = 21.15

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`crane_cost` : Float, USD
    crane cost if present

    *Default* = 12000.0

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`electricity_price` : Float, USD/kW/h
    Electricity price used to compute value in beyond lcoe metrics

    *Default* = 0.04

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`reserve_margin_price` : Float, USD/kW/yr
    Reserve margin price used to compute value in beyond lcoe metrics

    *Default* = 120.0

    *Minimum* = 0.0    *Maximum* = 10000.0


:code:`capacity_credit` : Float
    Capacity credit used to compute value in beyond lcoe metrics

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`benchmark_price` : Float, USD/kW/h
    Benchmark price used to nondimensionalize value in beyond lcoe
    metrics

    *Default* = 0.071

    *Minimum* = 0.0    *Maximum* = 1.0




TMDs
****************************************

:code:`name` : String
    Unique name of the TMD

:code:`component` : String
    Component location of the TMD (tower or platform)

:code:`location` : Array of Floats
    Location of TMD in global coordinates

:code:`mass` : Float, kg
    Mass of TMD

    *Default* = 0

:code:`stiffness` : Float, N/m
    Stiffness of TMD

    *Default* = 0

:code:`damping` : Float, (N/(m/s))
    Damping of TMD

    *Default* = 0

:code:`X_DOF` : Boolean
    Dof on or off for StC X

    *Default* = False

:code:`Y_DOF` : Boolean
    Dof on or off for StC Y

    *Default* = False

:code:`Z_DOF` : Boolean
    Dof on or off for StC Z

    *Default* = False

:code:`natural_frequency` : Float, rad/s
    Natural frequency of TMD, will overwrite stiffness (-1 indicates
    that it's not used)

    *Default* = -1

:code:`damping_ratio` : Float, non-dimensional
    Daming ratio of TMD, will overwrite damping (-1 indicates that
    it's not used)

    *Default* = -1

:code:`preload_spring` : Boolean
    Ensure that equilibrium point of the TMD is at `location` by
    offseting the location based on the spring constant

    *Default* = True

