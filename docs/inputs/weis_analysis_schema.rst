******************************
/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_analysis_schema.yaml
******************************
Scehma that describes the analysis and optimization options for WEIS


/Users/dzalkind/Tools/WEIS-2/weis/inputs/weis_analysis_schema.



general
****************************************

:code:`folder_output` : String
    Name of folder to dump output files

    *Default* = output

:code:`fname_output` : String
    File prefix for output files

    *Default* = output



design_variables
****************************************

Sets the design variables in a design optimization and analysis


rotor_diameter
########################################

Adjust the rotor diameter by changing the blade length (all blade properties constant with respect to non-dimensional span coordinates)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`minimum` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`maximum` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0




blade
########################################

Design variables associated with the wind turbine blades


aero_shape
========================================

Design variables associated with the blade aerodynamic shape


twist
----------------------------------------

Blade twist as a design variable by adding or subtracting radians from the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`inverse` : Boolean
    Words TODO?

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the twist distribution along blade span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float, rad
    Maximum allowable decrease of twist at each DV location along
    blade span.

    *Default* = 0.1

:code:`max_increase` : Float, rad
    Maximum allowable increase of twist at each DV location along
    blade span.

    *Default* = 0.1

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



chord
----------------------------------------

Blade chord as a design variable by scaling (multiplying) the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the chord distribution along blade span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



af_positions
----------------------------------------

Adjust airfoil positions along the blade span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`af_start` : Integer
    Index of airfoil where the optimization can start shifting airfoil
    position. The airfoil at blade tip is always locked.

    *Default* = 4

    *Minimum* = 1



rthick
----------------------------------------

Blade relative thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the relative thickness distribution along blade
    span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



L/D
----------------------------------------

Lift to drag ratio as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the lift to drag ratio distribution along blade
    span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



c_d
----------------------------------------

Drag coefficient at rated conditions as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the drag coefficient distribution along blade span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



stall_margin
----------------------------------------

Stall margin at rated conditions as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the stall margin distribution along blade span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



z
----------------------------------------

INN design parameter z
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    z design parameter count

    *Default* = 3

:code:`lower_bound` : Float


    *Default* = -1.0

    *Minimum* = -1e+30    *Maximum* = 1e+30


:code:`upper_bound` : Float


    *Default* = 1.0

    *Minimum* = -1e+30    *Maximum* = 1e+30




structure
========================================

Design variables associated with the internal blade structure


spar_cap_ss
----------------------------------------

Blade suction-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the spar cap on the suction side.
    By default, the first point close to blade root and the last point
    close to blade tip are locked. This is done to impose a pre-
    defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



spar_cap_ps
----------------------------------------

Blade pressure-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`equal_to_suction` : Boolean
    If the pressure side spar cap should be equal to the suction side
    layer

    *Default* = True

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the spar cap on the pressure side.
    By default, the first point close to blade root and the last point
    close to blade tip are locked. This is done to impose a pre-
    defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



te_ss
----------------------------------------

Blade suction-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the trailing edge reinforcement on
    the suction side. By default, the first point close to blade root
    and the last point close to blade tip are locked. This is done to
    impose a pre-defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



te_ps
----------------------------------------

Blade pressure-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`equal_to_suction` : Boolean
    If the pressure side spar cap should be equal to the suction side
    layer

    *Default* = True

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the trailing edge reinforcement on
    the pressure side. By default, the first point close to blade root
    and the last point close to blade tip are locked. This is done to
    impose a pre-defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



control
########################################

Design variables associated with the control of the wind turbine


tsr
========================================

Adjust the tip-speed ratio (ratio between blade tip velocity and steady hub-height wind speed)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`minimum` : Float
    Minimum allowable value

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0


:code:`maximum` : Float
    Maximum allowable value

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0


:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



flaps
========================================



te_flap_end
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.5

    *Minimum* = 0.1    *Maximum* = 1.0


:code:`max` : Float


    *Default* = 0.98

    *Minimum* = 0.1    *Maximum* = 1.0




te_flap_ext
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.01

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`max` : Float


    *Default* = 0.2

    *Minimum* = 0.0    *Maximum* = 1.0




ps_percent
========================================

Percent peak shaving as a design variable
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.75

:code:`upper_bound` : Float


    *Default* = 1.0



servo
========================================



pitch_control
----------------------------------------



omega
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, rad/s


    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`max` : Float, rad/s


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0




zeta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`max` : Float, rad/s


    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0




Kp_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, s


    *Default* = -100

:code:`max` : Float, s


    *Default* = 0



ptfm_freq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, rad/s


    *Default* = 1e-05

    *Minimum* = 1e-05

:code:`max` : Float, rad/s


    *Default* = 1.5

    *Minimum* = 1e-05



stability_margin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.01

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`max` : Float


    *Default* = 0.01

    *Minimum* = 0.0    *Maximum* = 1.0




torque_control
----------------------------------------



omega
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, rad/s


    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`max` : Float, rad/s


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0




zeta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`max` : Float, rad/s


    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0




flap_control
----------------------------------------



flp_kp_norm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.01

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`max` : Float


    *Default* = 5.0

    *Minimum* = 0.0    *Maximum* = 10.0




flp_tau
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 5

    *Minimum* = 0.0    *Maximum* = 100.0


:code:`max` : Float


    *Default* = 30

    *Minimum* = 0.0    *Maximum* = 100.0




ipc_control
----------------------------------------



Kp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, s


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`max` : Float, s


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`ref` : Float


    *Default* = 1e-08

    *Minimum* = 1e-10    *Maximum* = 1e-05




Ki
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`max` : Float


    *Default* = 1e-07

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`ref` : Float


    *Default* = 1e-08

    *Minimum* = 1e-10    *Maximum* = 1e-05




hub
########################################

Design variables associated with the hub


cone
========================================

Adjust the blade attachment coning angle (positive values are always away from the tower whether upwind or downwind)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756


:code:`upper_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756




hub_diameter
========================================

Adjust the rotor hub diameter
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for hub diameter

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for hub diameter

    *Default* = 30.0

    *Minimum* = 0.0    *Maximum* = 30.0




drivetrain
########################################

Design variables associated with the drivetrain


uptilt
========================================

Adjust the drive shaft tilt angle (positive values tilt away from the tower whether upwind or downwind)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756


:code:`upper_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756




overhang
========================================

Adjust the x-distance, parallel to the ground or still water line, from the tower top center to the rotor apex.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_tt_hub
========================================

Adjust the z-dimension height from the tower top to the rotor apex
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_hub_mb
========================================

Adjust the distance along the drive staft from the hub flange to the first main bearing
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_mb_mb
========================================

Adjust the distance along the drive staft from the first to the second main bearing
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




generator_length
========================================

Adjust the distance along the drive staft between the generator rotor drive shaft attachment to the stator bedplate attachment
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




gear_ratio
========================================

For geared configurations only, adjust the gear ratio of the gearbox that multiplies the shaft speed and divides the torque
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 500.0


:code:`upper_bound` : Float


    *Default* = 150.0

    *Minimum* = 1.0    *Maximum* = 1000.0




lss_diameter
========================================

Adjust the diameter at the beginning and end of the low speed shaft (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




hss_diameter
========================================

Adjust the diameter at the beginning and end of the high speed shaft (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




nose_diameter
========================================

For direct-drive configurations only, adjust the diameter at the beginning and end of the nose/turret (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




lss_wall_thickness
========================================

Adjust the thickness at the beginning and end of the low speed shaft (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




hss_wall_thickness
========================================

Adjust the thickness at the beginning and end of the high speed shaft (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




nose_wall_thickness
========================================

For direct-drive configurations only, adjust the thickness at the beginning and end of the nose/turret (assumes a linear taper)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_wall_thickness
========================================

For direct-drive configurations only, adjust the wall thickness along the elliptical bedplate
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_web_thickness
========================================

For geared configurations only, adjust the I-beam web thickness of the bedplate
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_flange_thickness
========================================

For geared configurations only, adjust the I-beam flange thickness of the bedplate
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_flange_width
========================================

For geared configurations only, adjust the I-beam flange width of the bedplate
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




tower
########################################

Design variables associated with the tower or monopile


outer_diameter
========================================

Adjust the outer diamter of the cylindrical column at nodes along the height.  Linear tapering is assumed between the nodes, creating conical frustums in each section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




layer_thickness
========================================

Adjust the layer thickness of each section in the column
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




section_height
========================================

Adjust the height of each conical section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




E
========================================

Isotropic Young's modulus
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Pa
    Design variable bound

    *Default* = 200000000000.0

    *Minimum* = 1.0    *Maximum* = 1000000000000.0


:code:`upper_bound` : Float, Pa
    Design variable bound

    *Default* = 200000000000.0

    *Minimum* = 1.0    *Maximum* = 1000000000000.0




rho
========================================

Material density of the tower
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, kg/m**3
    Design variable bound

    *Default* = 7800

    *Minimum* = 1.0    *Maximum* = 100000.0


:code:`upper_bound` : Float, kg/m**3
    Design variable bound

    *Default* = 7800

    *Minimum* = 1.0    *Maximum* = 100000.0




monopile
########################################

Design variables associated with the tower or monopile


outer_diameter
========================================

Adjust the outer diamter of the cylindrical column at nodes along the height.  Linear tapering is assumed between the nodes, creating conical frustums in each section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




layer_thickness
========================================

Adjust the layer thickness of each section in the column
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




section_height
========================================

Adjust the height of each conical section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




E
========================================

Isotropic Young's modulus
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Pa
    Design variable bound

    *Default* = 200000000000.0

    *Minimum* = 1.0    *Maximum* = 1000000000000.0


:code:`upper_bound` : Float, Pa
    Design variable bound

    *Default* = 200000000000.0

    *Minimum* = 1.0    *Maximum* = 1000000000000.0




rho
========================================

Material density of the tower
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, kg/m**3
    Design variable bound

    *Default* = 7800

    *Minimum* = 1.0    *Maximum* = 100000.0


:code:`upper_bound` : Float, kg/m**3
    Design variable bound

    *Default* = 7800

    *Minimum* = 1.0    *Maximum* = 100000.0




jacket
########################################

Design variables associated with the jacket


foot_head_ratio
========================================

Adjust the ratio of the jacket foot (bottom) radius to that of the head (top)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float
    Design variable bound

    *Default* = 1.5

    *Minimum* = 1.0    *Maximum* = 100.0


:code:`upper_bound` : Float
    Design variable bound

    *Default* = 1.5

    *Minimum* = 1.0    *Maximum* = 100.0




r_head
========================================

Adjust the radius of the jacket head.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




leg_diameter
========================================

Adjust the diameter of the jacket legs.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 1.5

    *Minimum* = 0.1    *Maximum* = 10.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 1.5

    *Minimum* = 0.1    *Maximum* = 10.0




height
========================================

Overall jacket height, meters.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 70

    *Minimum* = 0.1    *Maximum* = 1000.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 70

    *Minimum* = 0.1    *Maximum* = 1000.0




leg_thickness
========================================

Adjust the leg thicknesses of the jacket.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0




brace_diameters
========================================

Adjust the brace diameters of the jacket.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0




brace_thicknesses
========================================

Adjust the brace thicknesses of the jacket.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.001    *Maximum* = 10.0




bay_spacing
========================================

Jacket bay nodal spacing.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`upper_bound` : Float
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 1.0




floating
########################################

Design variables associated with the floating platform


joints
========================================

Design variables associated with the node/joint locations used in the floating platform
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



z_coordinate
----------------------------------------

:code:`names` : Array of Strings
    Joint or member names of those that are linked

:code:`lower_bound` : Float, m
    Design variable bound

:code:`upper_bound` : Float, m
    Design variable bound



r_coordinate
----------------------------------------

:code:`names` : Array of Strings
    Joint or member names of those that are linked

:code:`lower_bound` : Float, m
    Design variable bound

:code:`upper_bound` : Float, m
    Design variable bound



members
========================================

Design variables associated with the members used in the floating platform
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



groups
----------------------------------------

:code:`names` : Array of Strings
    Joint or member names of those that are linked



diameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Diameter optimization of member group
:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`constant` : Boolean
    Should the diameters be constant

    *Default* = False



thickness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Thickness optimization of member group
:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




ballast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ballast volume optimization of member group
:code:`lower_bound` : Float, m^3
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float, m^3
    Design variable bound

    *Default* = 100000.0

    *Minimum* = 0.0



axial_joints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`names` : Array of Strings
    Joint or member names of those that are linked

:code:`lower_bound` : Float
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`upper_bound` : Float
    Design variable bound

    *Default* = 1.0

    *Minimum* = 0.0    *Maximum* = 1.0




stiffeners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stiffener optimization of member group


ring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ring stiffener optimization of member group


size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ring stiffener sizing multiplier on T-shape
:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float


    *Default* = 1.5



spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ring stiffener spacing along member axis
:code:`lower_bound` : Float
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.0



longitudinal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Longitudinal stiffener optimization of member group


size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Longitudinal stiffener sizing multiplier on T-shape
:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float


    *Default* = 1.5



spacing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Longitudinal stiffener spacing around member annulus
:code:`lower_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 3.141592653589793


:code:`upper_bound` : Float, rad
    Design variable bound

    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 3.141592653589793




mooring
########################################

Design variables associated with the mooring system


line_length
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float, m
    Design variable bound

    *Minimum* = 0.0



line_diameter
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float, m
    Design variable bound

    *Minimum* = 0.0



line_mass_density_coeff
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float, m
    Design variable bound

    *Minimum* = 0.0



line_stiffness_coeff
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0

:code:`upper_bound` : Float, m
    Design variable bound

    *Minimum* = 0.0



TMDs
########################################

Design variables associated with TMDs
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



groups
========================================

:code:`names` : Array of Strings
    TMD names of those that are linked



mass
----------------------------------------

Mass optimization of TMD group
:code:`lower_bound` : Float


    *Default* = 20000

:code:`upper_bound` : Float


    *Default* = 20000

:code:`initial` : Float
    Initial condition of TMD group

    *Default* = 100

:code:`const_omega` : Boolean
    Keep the natural frequency constant while the mass changes?

    *Default* = False

:code:`const_zeta` : Boolean
    Keep the damping ratio constant while the mass changes?

    *Default* = False



stiffness
----------------------------------------

Stiffness optimization of TMD group
:code:`lower_bound` : Float


    *Default* = 20000

:code:`upper_bound` : Float


    *Default* = 20000

:code:`initial` : Float
    Initial condition of TMD group

    *Default* = 100



damping
----------------------------------------

Damping optimization of TMD group
:code:`lower_bound` : Float


    *Default* = 20000

:code:`upper_bound` : Float


    *Default* = 20000

:code:`initial` : Float
    Initial condition of TMD group

    *Default* = 100



natural_frequency
----------------------------------------

Natural frequency optimization of TMD group
:code:`lower_bound` : Float


    *Default* = 20000

:code:`upper_bound` : Float


    *Default* = 20000

:code:`initial` : Float
    Initial condition of TMD group

    *Default* = 100

:code:`const_zeta` : Boolean
    Keep the damping ratio constant while the natural frequency
    changes?

    *Default* = False



damping_ratio
----------------------------------------

Damping ratio optimization of TMD group
:code:`lower_bound` : Float


    *Default* = 20000

:code:`upper_bound` : Float


    *Default* = 20000

:code:`initial` : Float
    Initial condition of TMD group

    *Default* = 100



constraints
****************************************

Activate the constraints that are applied to a design optimization


blade
########################################

Constraints associated with the blade design


strains_spar_cap_ss
========================================

Enforce a maximum allowable strain in the suction-side spar caps
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1


:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



strains_spar_cap_ps
========================================

Enforce a maximum allowable strain in the pressure-side spar caps
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1


:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



strains_te_ss
========================================

Enforce a maximum allowable strain in the suction-side trailing edge reinforcements
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1


:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



strains_te_ps
========================================

Enforce a maximum allowable strain in the pressure-side trailing edge reinforcements
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1


:code:`index_start` : Integer
    First index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 0

    *Minimum* = 0

:code:`index_end` : Integer
    Last index of the array of design variables/constraints that is
    optimized/constrained

    *Default* = 8

    *Minimum* = 0



tip_deflection
========================================

Enforce a maximum allowable blade tip deflection towards the tower expressed as a safety factor on the parked margin.  Meaning a parked distance to the tower of 30m and a constraint value here of 1.5 would mean that 30/1.5=20m of deflection is the maximum allowable
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`margin` : Float


    *Default* = 1.4175

    *Minimum* = 1.0    *Maximum* = 10.0




t_sc_joint
========================================

Enforce a maximum allowable spar cap thickness, expressed as the ratio of the required spar cap thickness at the joint location to the nominal spar cap thickness.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



rail_transport
========================================

Enforce sufficient blade flexibility such that they can be transported on rail cars without exceeding maximum blade strains or derailment.  User can activate either 8-axle flatcars or 4-axle
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`8_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`4_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False



stall
========================================

Ensuring blade angles of attacks do not approach the stall point. Margin is expressed in radians from stall.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`margin` : Float, radians


    *Default* = 0.05233

    *Minimum* = 0.0    *Maximum* = 0.5




chord
========================================

Enforcing the maximum chord length limit at all points along blade span.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, meter


    *Default* = 4.75

    *Minimum* = 0.1    *Maximum* = 20.0




root_circle_diameter
========================================

Enforcing the minimum blade root circle diameter.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max_ratio` : Float
    Maximum ratio between the recommended root circle diameter and the
    actual chord at blade root. The optimizer will make sure that the
    ratio stays below this value.

    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 10.0




frequency
========================================

Frequency separation constraint between blade fundamental frequency and blade passing (3P) frequency at rated conditions using gamma_freq margin. Can be activated for blade flap and/or edge modes.
:code:`flap_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`edge_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False



moment_coefficient
========================================

(EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.15

    *Minimum* = 0.01    *Maximum* = 5.0


:code:`max` : Float


    *Default* = 0.15

    *Minimum* = 0.01    *Maximum* = 5.0




match_cl_cd
========================================

(EXPERIMENTAL) Targeted airfoil cl/cd ratio (useful for inverse design approaches that is not recommendend for general use)
:code:`flag_cl` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`flag_cd` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`filename` : String
    file path to constraint data

    *Default* = 



match_L_D
========================================

(EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)
:code:`flag_L` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`flag_D` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`filename` : String
    file path to constraint data

    *Default* = 



AEP
========================================

Set a minimum bound on AEP in kWh when optimizing the blade and rotor parameters
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float, kWh


    *Default* = 1.0

    *Minimum* = 1.0



thrust_coeff
========================================

(EXPERIMENTAL) Bound the ccblade thrust coefficient away from unconstrained optimal when optimizing for power, for highly-loaded rotors
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower` : Float


    *Minimum* = 0.0

:code:`upper` : Float


    *Minimum* = 0.0



tower
########################################

Constraints associated with the tower design


height_constraint
========================================

Double-sided constraint to ensure total tower height meets target hub height when adjusting section heights
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-06    *Maximum* = 10.0


:code:`upper_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-06    *Maximum* = 10.0




stress
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================

Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================

Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



slope
========================================

Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



thickness_slope
========================================

Ensure that the thickness moving up the tower at any node is always equal or less than the thickness of the section preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



d_to_t
========================================

Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0


:code:`upper_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0




taper
========================================

Enforcing a max allowable conical frustum taper ratio per section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.5

    *Minimum* = 0.001    *Maximum* = 1.0




frequency
========================================

Frequency separation constraint between all tower modal frequencies and blade period (1P) and passing (3P) frequencies at rated conditions using gamma_freq margin.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



frequency_1
========================================

Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0


:code:`upper_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0




monopile
########################################

Constraints associated with the monopile design


stress
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================

Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================

Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



slope
========================================

Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



thickness_slope
========================================

Ensure that the thickness moving up the tower at any node is always equal or less than the thickness of the section preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



d_to_t
========================================

Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0


:code:`upper_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0




taper
========================================

Enforcing a max allowable conical frustum taper ratio per section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.5

    *Minimum* = 0.001    *Maximum* = 1.0




frequency_1
========================================

Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0


:code:`upper_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0




pile_depth
========================================

Ensures that the submerged suction pile depth meets a minimum value
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 200.0




tower_diameter_coupling
========================================

Ensures that the top diameter of the monopile is the same or larger than the base diameter of the tower
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



jacket
########################################

Constraints associated with the monopile design


stress
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================

Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================

Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



slope
========================================

Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



thickness_slope
========================================

Ensure that the thickness moving up the tower at any node is always equal or less than the thickness of the section preceding it
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



d_to_t
========================================

Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0


:code:`upper_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0




taper
========================================

Enforcing a max allowable conical frustum taper ratio per section
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.5

    *Minimum* = 0.001    *Maximum* = 1.0




frequency_1
========================================

Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0


:code:`upper_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 5.0




pile_depth
========================================

Ensures that the submerged suction pile depth meets a minimum value
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 200.0




tower_diameter_coupling
========================================

Ensures that the top diameter of the monopile is the same or larger than the base diameter of the tower
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



hub
########################################



hub_diameter
========================================

Ensure that the diameter of the hub is sufficient to accommodate the number of blades and blade root diameter
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



drivetrain
########################################



lss
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



hss
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



bedplate
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mb1
========================================

Ensure that the angular deflection at this meain bearing does not exceed the maximum allowable deflection for the bearing type
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mb2
========================================

Ensure that the angular deflection at this meain bearing does not exceed the maximum allowable deflection for the bearing type
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



length
========================================

Ensure that the bedplate length is sufficient to meet desired overhang value
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



height
========================================

Ensure that the bedplate height is sufficient to meet desired nacelle height value
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



access
========================================

For direct-drive configurations only, ensure that the inner diameter of the nose/turret is big enough to allow human access
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, meter
    Minimum size to ensure human maintenance access

    *Default* = 2.0

    *Minimum* = 0.1    *Maximum* = 5.0




shaft_deflection
========================================

Allowable non-torque deflection of the shaft, in meters, at the generator rotor attachment for direct drive or gearbox attachment for geared drive
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`upper_bound` : Float, meter
    Upper limit of deflection

    *Default* = 0.0001

    *Minimum* = 1e-06    *Maximum* = 1.0




shaft_angle
========================================

Allowable non-torque angular deflection of the shaft, in radians, at the generator rotor attachment for direct drive or gearbox attachment for geared drive
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`upper_bound` : Float, radian
    Upper limit of angular deflection

    *Default* = 0.001

    *Minimum* = 1e-05    *Maximum* = 1.0




stator_deflection
========================================

Allowable deflection of the nose or bedplate, in meters, at the generator stator attachment
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`upper_bound` : Float, meter
    Upper limit of deflection

    *Default* = 0.0001

    *Minimum* = 1e-06    *Maximum* = 1.0




stator_angle
========================================

Allowable non-torque angular deflection of the nose or bedplate, in radians, at the generator stator attachment
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`upper_bound` : Float, radian
    Upper limit of angular deflection

    *Default* = 0.001

    *Minimum* = 1e-05    *Maximum* = 1.0




ecc
========================================

For direct-drive configurations only, ensure that the elliptical bedplate length is greater than its height
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



floating
########################################



operational_heel
========================================

Ensure that the mooring system has enough restoring force to keep the heel/pitch angle below this limit
:code:`upper_bound` : Float, rad


    *Default* = 0.17453292519943295

    *Minimum* = 0.017453292519943295    *Maximum* = 0.7853981633974483




survival_heel
========================================

Ensure that the mooring system has enough restoring force to keep the heel/pitch angle below this limit
:code:`upper_bound` : Float, rad


    *Default* = 0.17453292519943295

    *Minimum* = 0.017453292519943295    *Maximum* = 0.7853981633974483




max_surge
========================================

Ensure that the mooring system has enough restoring force so that this surge distance, expressed as a fraction of water depth, is not exceeded
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`upper_bound` : Float


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0




buoyancy
========================================

Ensures that the platform displacement is sufficient to support the weight of the turbine system
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



fixed_ballast_capacity
========================================

Ensures that there is sufficient volume to hold the specified fixed (permanent) ballast
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



variable_ballast_capacity
========================================

Ensures that there is sufficient volume to hold the needed water (variable) ballast to achieve neutral buoyancy
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



metacentric_height
========================================

Ensures hydrostatic stability with a positive metacentric height
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, meter


    *Default* = 10.0

    *Minimum* = 0.0



freeboard_margin
========================================

Ensures that the freeboard (top points of structure) of floating platform stays above the waterline at the survival heel offset
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



draft_margin
========================================

Ensures that the draft (bottom points of structure) of floating platform stays beneath the waterline at the survival heel offset
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



fairlead_depth
========================================

Ensures that the mooring line attachment depth (fairlead) is sufficiently beneath the water line that it is not exposed at the significant wave height
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mooring_surge
========================================

Ensures that the mooring lines have sufficient restoring force to overcome rotor thrust at the max surge offset
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mooring_heel
========================================

Ensures that the mooring lines have sufficient restoring force to overcome rotor thrust at the max heel offset
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mooring_tension
========================================

Keep the mooring line tension below its breaking point
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mooring_length
========================================

Keep the mooring line length within the bounds for catenary hang or TLP tension
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



anchor_vertical
========================================

Ensure that the maximum vertical force on the anchor does not exceed limit
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



anchor_lateral
========================================

Ensure that the maximum lateral force on the anchor does not exceed limit
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



stress
========================================

Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================

Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================

Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



surge_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



sway_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



heave_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



roll_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



pitch_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



yaw_period
========================================

Ensure that the rigid body period stays within bounds
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01

:code:`upper_bound` : Float, s


    *Default* = 1.0

    *Minimum* = 0.01



Max_Offset
========================================

Maximum combined surge/sway offset. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, m


    *Default* = 20

    *Minimum* = 0.0    *Maximum* = 20000.0




control
########################################



flap_control
========================================

Words TODO
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1000000.0


:code:`max` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1000000.0




rotor_overspeed
========================================

(Maximum rotor speed / rated rotor speed) - 1.  Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`max` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0




Max_PtfmPitch
========================================

Maximum platform pitch displacement over all cases. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, deg


    *Default* = 6.0

    *Minimum* = 0.0    *Maximum* = 30.0




Std_PtfmPitch
========================================

Maximum platform pitch standard deviation over all cases. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, deg


    *Default* = 2.0

    *Minimum* = 0.0    *Maximum* = 30.0




Max_TwrBsMyt
========================================

Maximum platform pitch displacement
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, kN*m


    *Default* = 100000.0

    *Minimum* = 0.0    *Maximum* = 100000000.0




DEL_TwrBsMyt
========================================

Maximum platform pitch displacement
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, kN*m


    *Default* = 100000.0

    *Minimum* = 0.0    *Maximum* = 100000000.0




nacelle_acceleration
========================================

Maximum Nacelle IMU accelleration magnitude, i.e., sqrt(NcIMUTAxs^2 + NcIMUTAys^2 + NcIMUTAzs^2). Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, m/s^2


    *Default* = 3.2667

    *Minimum* = 0.0    *Maximum* = 30.0




avg_pitch_travel
========================================

Average pitch travel per second
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, deg/s


    *Default* = 5

    *Minimum* = 0.0    *Maximum* = 30.0




pitch_duty_cycle
========================================

Number of pitch direction changes per second of simulation
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, deg/s


    *Default* = 5

    *Minimum* = 0.0    *Maximum* = 30.0




damage
########################################



tower_base
========================================

Tower base damage constraint
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float


    *Default* = 1.0

    *Minimum* = 1e-05    *Maximum* = 30.0


:code:`log` : Boolean
    Use the logarithm of damage as the constraint.

    *Default* = False



openfast_failed
########################################

:code:`flag` : Boolean
    Constrain design to one where OpenFAST simulations don't
    fail_value

    *Default* = False

:code:`merit_figure` : String
    Objective function / merit figure for optimization

    *Default* = LCOE



driver
****************************************



optimization
########################################

Specification of the optimization driver (optimization algorithm) parameters
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`tol` : Float
    Convergence tolerance (relative)

    *Default* = 1e-06

    *Minimum* = 1e-12    *Maximum* = 1.0


:code:`max_iter` : Integer
    Max number of optimization iterations

    *Default* = 100

    *Minimum* = 0    *Maximum* = 100000


:code:`max_major_iter` : Integer
    Max number of major optimization iterations of SNOPT

    *Default* = 10

    *Minimum* = 0    *Maximum* = 100000


:code:`max_minor_iter` : Integer
    Max number of minor optimization iterations of SNOPT

    *Default* = 100

    *Minimum* = 0    *Maximum* = 100000


:code:`time_limit` : Integer
    Max seconds of major iteration runtime for SNOPT

    *Default* = 0

    *Minimum* = 0

:code:`max_function_calls` : Integer
    Max number of calls to objective function evaluation

    *Default* = 100000

    *Minimum* = 0    *Maximum* = 100000000


:code:`solver` : String from, ['SLSQP', 'CONMIN', 'COBYLA', 'SNOPT', 'Nelder-Mead', 'GA', 'GN_DIRECT', 'GN_DIRECT_L', 'GN_DIRECT_L_NOSCAL', 'GN_ORIG_DIRECT', 'GN_ORIG_DIRECT_L', 'GN_AGS', 'GN_ISRES', 'LN_COBYLA', 'LD_MMA', 'LD_CCSAQ', 'LD_SLSQP', 'NSGA2']
    Optimization driver.

    *Default* = SLSQP

:code:`step_size` : Float
    Maximum step size for finite difference approximation

    *Default* = 0.001

    *Minimum* = 1e-10    *Maximum* = 100.0


:code:`form` : String from, ['central', 'forward', 'complex']
    Finite difference calculation mode

    *Default* = central

:code:`step_calc` : String from, ['None', 'abs', 'rel_avg', 'rel_element', 'rel_legacy']
    Step type for computing the size of the finite difference step.

    *Default* = None

:code:`debug_print` : Boolean
    Toggle driver debug printing

    *Default* = False



design_of_experiments
########################################

Specification of the design of experiments driver parameters
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`run_parallel` : Boolean
    Toggle parallel model runs

    *Default* = True

:code:`generator` : String from, ['Uniform', 'FullFact', 'PlackettBurman', 'BoxBehnken', 'LatinHypercube']
    Type of model input generator.

    *Default* = Uniform

:code:`num_samples` : Integer
    Number of samples to evaluate model at (Uniform and LatinHypercube
    only)

    *Default* = 5

    *Minimum* = 1    *Maximum* = 1000000


:code:`seed` : Integer
    Random seed to use if design is randomized

    *Default* = 2

    *Minimum* = 1    *Maximum* = 1000000


:code:`levels` : Integer
    Number of evenly spaced levels between each design variable lower
    and upper bound (FullFactorial only)

    *Default* = 2

    *Minimum* = 1    *Maximum* = 1000000


:code:`criterion` : String from, ['None', 'center', 'c', 'maximin', 'm', 'centermaximin', 'cm', 'correelation', 'corr']
    Descriptor of sampling method for LatinHypercube generator

    *Default* = center

:code:`iterations` : Integer
    Number of iterations in maximin and correlations algorithms
    (LatinHypercube only)

    *Default* = 2

    *Minimum* = 1    *Maximum* = 1000000


:code:`debug_print` : Boolean
    Toggle driver debug printing

    *Default* = False



step_size_study
########################################

Specification of the step size study parameters
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`form` : String from, ['central', 'forward', 'complex']
    Finite difference calculation mode

    *Default* = central

:code:`driver_scaling` : Boolean
    When True, return derivatives that are scaled according to either
    the adder and scaler or the ref and ref0 values that were
    specified when add_design_var, add_objective, and add_constraint
    were called on the model.

    *Default* = False



recorder
****************************************

Optimization iteration recording via OpenMDAO
:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`file_name` : String
    OpenMDAO recorder output SQL database file

    *Default* = log_opt.sql

:code:`just_dvs` : Boolean
    If true, only record design variables.

    *Default* = False

