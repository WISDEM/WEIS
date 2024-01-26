$schema: http://json-schema.org/draft-07/schema#
$id: WEIS_add-ons_analysis
title: WEIS analysis ontology add-ons beyond WISDEM ontology
description: Scehma that describes the analysis and optimization options for WEIS
type: object
properties:
    general:
        type: object
        default: {}
        properties:
            folder_output: {type: string, default: output, description: Name of folder to dump output files}
            fname_output: {type: string, default: output, description: File prefix for output files}
    design_variables:
        type: object
        default: {}
        description: Sets the design variables in a design optimization and analysis
        properties:
            rotor_diameter:
                type: object
                default: {}
                description: Adjust the rotor diameter by changing the blade length (all blade properties constant with respect to non-dimensional span coordinates)
                properties:
                    flag: &id001 {type: boolean, default: false, description: Activates as a design variable or constraint}
                    minimum: {type: number, default: 0.0, minimum: 0.0, maximum: 1000.0, unit: m}
                    maximum: {type: number, default: 0.0, minimum: 0.0, maximum: 1000.0, unit: m}
            blade:
                type: object
                default: {}
                description: Design variables associated with the wind turbine blades
                properties:
                    aero_shape:
                        type: object
                        default: {}
                        description: Design variables associated with the blade aerodynamic shape
                        properties:
                            twist:
                                type: object
                                default: {}
                                description: Blade twist as a design variable by adding or subtracting radians from the initial value at spline control points along the span.
                                properties:
                                    flag: *id001
                                    inverse: {type: boolean, default: false, description: Words TODO?}
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the twist distribution along blade span.}
                                    max_decrease: {type: number, description: Maximum allowable decrease of twist at each DV location along blade span., default: 0.1, unit: rad}
                                    max_increase: {type: number, description: Maximum allowable increase of twist at each DV location along blade span., default: 0.1, unit: rad}
                                    index_start: &id002 {type: integer, default: 0, minimum: 0, unit: none, description: First index of the array of design variables/constraints that is optimized/constrained}
                                    index_end: &id003 {type: integer, default: 8, minimum: 0, unit: none, description: Last index of the array of design variables/constraints that is optimized/constrained}
                            chord:
                                type: object
                                default: {}
                                description: Blade chord as a design variable by scaling (multiplying) the initial value at spline control points along the span.
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the chord distribution along blade span.}
                                    max_decrease: &id004 {type: number, description: Maximum nondimensional decrease at each optimization location, default: 0.5}
                                    max_increase: &id005 {type: number, description: Maximum nondimensional increase at each optimization location, default: 1.5}
                                    index_start: *id002
                                    index_end: *id003
                            af_positions:
                                type: object
                                default: {}
                                description: Adjust airfoil positions along the blade span.
                                properties:
                                    flag: *id001
                                    af_start: {type: integer, default: 4, minimum: 1, description: Index of airfoil where the optimization can start shifting airfoil position. The airfoil at blade tip is always locked.}
                            rthick:
                                type: object
                                default: {}
                                description: Blade relative thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the relative thickness distribution along blade span.}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            L/D:
                                type: object
                                default: {}
                                description: Lift to drag ratio as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the lift to drag ratio distribution along blade span.}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            c_d:
                                type: object
                                default: {}
                                description: Drag coefficient at rated conditions as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the drag coefficient distribution along blade span.}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            stall_margin:
                                type: object
                                default: {}
                                description: Stall margin at rated conditions as a design variable by scaling (multiplying) the initial value at spline control points along the span. This requires the INN for airfoil design
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: Number of equally-spaced control points of the spline parametrizing the stall margin distribution along blade span.}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            z:
                                type: object
                                default: {}
                                description: INN design parameter z
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 3, description: z design parameter count}
                                    lower_bound: {type: number, default: -1.0, minimum: -1e+30, maximum: 1e+30}
                                    upper_bound: {type: number, default: 1.0, minimum: -1e+30, maximum: 1e+30}
                    structure:
                        type: object
                        default: {}
                        description: Design variables associated with the internal blade structure
                        properties:
                            spar_cap_ss:
                                type: object
                                description: Blade suction-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
                                default: {}
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: 'Number of equally-spaced control points of the spline parametrizing the thickness of the spar cap on the suction side. By default, the first point close to blade root and the last point close to blade tip are locked. This is done to impose a pre-defined taper to small thicknesses and mimic a blade manufacturability constraint.'}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            spar_cap_ps:
                                type: object
                                description: Blade pressure-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
                                default: {}
                                properties:
                                    flag: *id001
                                    equal_to_suction: &id006 {type: boolean, default: true, description: If the pressure side spar cap should be equal to the suction side layer}
                                    n_opt: {type: integer, default: 8, minimum: 4, description: 'Number of equally-spaced control points of the spline parametrizing the thickness of the spar cap on the pressure side. By default, the first point close to blade root and the last point close to blade tip are locked. This is done to impose a pre-defined taper to small thicknesses and mimic a blade manufacturability constraint.'}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            te_ss:
                                type: object
                                default: {}
                                description: Blade suction-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
                                properties:
                                    flag: *id001
                                    n_opt: {type: integer, default: 8, minimum: 4, description: 'Number of equally-spaced control points of the spline parametrizing the thickness of the trailing edge reinforcement on the suction side. By default, the first point close to blade root and the last point close to blade tip are locked. This is done to impose a pre-defined taper to small thicknesses and mimic a blade manufacturability constraint.'}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
                            te_ps:
                                type: object
                                description: Blade pressure-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.
                                default: {}
                                properties:
                                    flag: *id001
                                    equal_to_suction: *id006
                                    n_opt: {type: integer, default: 8, minimum: 4, description: 'Number of equally-spaced control points of the spline parametrizing the thickness of the trailing edge reinforcement on the pressure side. By default, the first point close to blade root and the last point close to blade tip are locked. This is done to impose a pre-defined taper to small thicknesses and mimic a blade manufacturability constraint.'}
                                    max_decrease: *id004
                                    max_increase: *id005
                                    index_start: *id002
                                    index_end: *id003
            control:
                type: object
                default: {}
                description: Design variables associated with the control of the wind turbine
                properties:
                    tsr:
                        type: object
                        default: {}
                        description: Adjust the tip-speed ratio (ratio between blade tip velocity and steady hub-height wind speed)
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            minimum: {type: number, default: 0.0, minimum: 0.0, maximum: 30.0, unit: none, description: Minimum allowable value}
                            maximum: {type: number, default: 0.0, minimum: 0.0, maximum: 30.0, unit: none, description: Maximum allowable value}
                            min_gain: {type: number, default: 0.5, unit: none, description: Lower bound on scalar multiplier that will be applied to value at control points}
                            max_gain: {type: number, default: 1.5, unit: none, description: Upper bound on scalar multiplier that will be applied to value at control points}
                    flaps:
                        type: object
                        default: {}
                        properties:
                            te_flap_end:
                                type: object
                                default: {}
                                properties:
                                    flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                    min: {type: number, maximum: 1.0, minimum: 0.1, default: 0.5}
                                    max: {type: number, maximum: 1.0, minimum: 0.1, default: 0.98}
                            te_flap_ext:
                                type: object
                                default: {}
                                properties:
                                    flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                    min: {type: number, maximum: 1.0, minimum: 0.0, default: 0.01}
                                    max: {type: number, maximum: 1.0, minimum: 0.0, default: 0.2}
                    ps_percent:
                        type: object
                        default: {}
                        description: Percent peak shaving as a design variable
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            lower_bound: {type: number, default: 0.75, unit: none}
                            upper_bound: {type: number, default: 1.0, unit: none}
                    servo:
                        type: object
                        default: {}
                        properties:
                            pitch_control:
                                type: object
                                default: {}
                                properties:
                                    omega:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.1, minimum: 0.0, maximum: 10.0, unit: rad/s}
                                            max: {type: number, default: 0.7, minimum: 0.0, maximum: 10.0, unit: rad/s}
                                    zeta:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.7, minimum: 0.0, maximum: 10.0, unit: none}
                                            max: {type: number, default: 1.5, minimum: 0.0, maximum: 10.0, unit: rad/s}
                                    Kp_float:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: -100, unit: s}
                                            max: {type: number, default: 0, unit: s}
                                    ptfm_freq:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 1e-05, minimum: 1e-05, unit: rad/s}
                                            max: {type: number, default: 1.5, minimum: 1e-05, unit: rad/s}
                                    stability_margin:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.01, minimum: 0.0, maximum: 1.0, unit: none}
                                            max: {type: number, default: 0.01, minimum: 0.0, maximum: 1.0, unit: none}
                            torque_control:
                                type: object
                                default: {}
                                properties:
                                    omega:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.1, minimum: 0.0, maximum: 10.0, unit: rad/s}
                                            max: {type: number, default: 0.7, minimum: 0.0, maximum: 10.0, unit: rad/s}
                                    zeta:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.7, minimum: 0.0, maximum: 10.0, unit: none}
                                            max: {type: number, default: 1.5, minimum: 0.0, maximum: 10.0, unit: rad/s}
                            flap_control:
                                type: object
                                default: {}
                                properties:
                                    flp_kp_norm:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.01, minimum: 0.0, maximum: 10.0, unit: none}
                                            max: {type: number, default: 5.0, minimum: 0.0, maximum: 10.0, unit: none}
                                    flp_tau:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 5, minimum: 0.0, maximum: 100.0, unit: none}
                                            max: {type: number, default: 30, minimum: 0.0, maximum: 100.0, unit: none}
                            ipc_control:
                                type: object
                                default: {}
                                properties:
                                    Kp:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.0, minimum: 0.0, maximum: 1000.0, unit: s}
                                            max: {type: number, default: 0.0, minimum: 0.0, maximum: 1000.0, unit: s}
                                            ref: {type: number, default: 1e-08, minimum: 1e-10, maximum: 1e-05}
                                    Ki:
                                        type: object
                                        default: {}
                                        properties:
                                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                                            min: {type: number, default: 0.0, minimum: 0.0, maximum: 1000.0, unit: none}
                                            max: {type: number, default: 1e-07, minimum: 0.0, maximum: 1000.0, unit: none}
                                            ref: {type: number, default: 1e-08, minimum: 1e-10, maximum: 1e-05}
            hub:
                type: object
                default: {}
                description: Design variables associated with the hub
                properties:
                    cone:
                        type: object
                        default: {}
                        description: Adjust the blade attachment coning angle (positive values are always away from the tower whether upwind or downwind)
                        properties:
                            flag: *id001
                            lower_bound: &id007 {type: number, minimum: 0.0, maximum: 0.5235987756, default: 0.0, unit: rad, description: Design variable bound}
                            upper_bound: *id007
                    hub_diameter:
                        type: object
                        default: {}
                        description: Adjust the rotor hub diameter
                        properties:
                            flag: *id001
                            lower_bound: {type: number, minimum: 0.0, maximum: 30.0, default: 0.0, unit: m, description: Lowest value allowable for hub diameter}
                            upper_bound: {type: number, minimum: 0.0, maximum: 30.0, default: 30.0, unit: m, description: Highest value allowable for hub diameter}
            drivetrain:
                type: object
                default: {}
                description: Design variables associated with the drivetrain
                properties:
                    uptilt:
                        type: object
                        default: {}
                        description: Adjust the drive shaft tilt angle (positive values tilt away from the tower whether upwind or downwind)
                        properties:
                            flag: *id001
                            lower_bound: *id007
                            upper_bound: *id007
                    overhang:
                        type: object
                        default: {}
                        description: Adjust the x-distance, parallel to the ground or still water line, from the tower top center to the rotor apex.
                        properties:
                            flag: *id001
                            lower_bound: &id008 {type: number, minimum: 0.1, maximum: 30.0, default: 0.1, unit: m, description: Lowest value allowable for design variable}
                            upper_bound: &id009 {type: number, minimum: 0.1, maximum: 30.0, default: 0.1, unit: m, description: Highest value allowable for design variable}
                    distance_tt_hub:
                        type: object
                        default: {}
                        description: Adjust the z-dimension height from the tower top to the rotor apex
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    distance_hub_mb:
                        type: object
                        default: {}
                        description: Adjust the distance along the drive staft from the hub flange to the first main bearing
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    distance_mb_mb:
                        type: object
                        default: {}
                        description: Adjust the distance along the drive staft from the first to the second main bearing
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    generator_length:
                        type: object
                        default: {}
                        description: Adjust the distance along the drive staft between the generator rotor drive shaft attachment to the stator bedplate attachment
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    gear_ratio:
                        type: object
                        default: {}
                        description: For geared configurations only, adjust the gear ratio of the gearbox that multiplies the shaft speed and divides the torque
                        properties:
                            flag: *id001
                            lower_bound: {type: number, minimum: 1.0, maximum: 500.0, default: 1.0, unit: none}
                            upper_bound: {type: number, minimum: 1.0, maximum: 1000.0, default: 150.0, unit: none}
                    lss_diameter:
                        type: object
                        default: {}
                        description: Adjust the diameter at the beginning and end of the low speed shaft (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    hss_diameter:
                        type: object
                        default: {}
                        description: Adjust the diameter at the beginning and end of the high speed shaft (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    nose_diameter:
                        type: object
                        default: {}
                        description: For direct-drive configurations only, adjust the diameter at the beginning and end of the nose/turret (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: *id008
                            upper_bound: *id009
                    lss_wall_thickness:
                        type: object
                        default: {}
                        description: Adjust the thickness at the beginning and end of the low speed shaft (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: &id010 {type: number, minimum: 0.001, maximum: 3.0, default: 0.001, unit: m}
                            upper_bound: &id011 {type: number, minimum: 0.01, maximum: 5.0, default: 1.0, unit: m}
                    hss_wall_thickness:
                        type: object
                        default: {}
                        description: Adjust the thickness at the beginning and end of the high speed shaft (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
                    nose_wall_thickness:
                        type: object
                        default: {}
                        description: For direct-drive configurations only, adjust the thickness at the beginning and end of the nose/turret (assumes a linear taper)
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
                    bedplate_wall_thickness:
                        type: object
                        default: {}
                        description: For direct-drive configurations only, adjust the wall thickness along the elliptical bedplate
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
                    bedplate_web_thickness:
                        type: object
                        default: {}
                        description: For geared configurations only, adjust the I-beam web thickness of the bedplate
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
                    bedplate_flange_thickness:
                        type: object
                        default: {}
                        description: For geared configurations only, adjust the I-beam flange thickness of the bedplate
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
                    bedplate_flange_width:
                        type: object
                        default: {}
                        description: For geared configurations only, adjust the I-beam flange width of the bedplate
                        properties:
                            flag: *id001
                            lower_bound: *id010
                            upper_bound: *id011
            tower: &id017
                type: object
                description: Design variables associated with the tower or monopile
                default: {}
                properties:
                    outer_diameter:
                        type: object
                        description: Adjust the outer diamter of the cylindrical column at nodes along the height.  Linear tapering is assumed between the nodes, creating conical frustums in each section
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id012 {type: number, minimum: 0.1, maximum: 100.0, default: 5.0, unit: m, description: Design variable bound}
                            upper_bound: *id012
                    layer_thickness:
                        type: object
                        default: {}
                        description: Adjust the layer thickness of each section in the column
                        properties:
                            flag: *id001
                            lower_bound: &id013 {type: number, minimum: 1e-05, maximum: 1.0, default: 0.01, unit: m, description: Design variable bound}
                            upper_bound: *id013
                    section_height:
                        type: object
                        default: {}
                        description: Adjust the height of each conical section
                        properties:
                            flag: *id001
                            lower_bound: &id014 {type: number, minimum: 0.1, maximum: 100.0, default: 5.0, unit: m, description: Design variable bound}
                            upper_bound: *id014
                    E:
                        type: object
                        default: {}
                        description: Isotropic Young's modulus
                        properties:
                            flag: *id001
                            lower_bound: &id015 {type: number, minimum: 1.0, maximum: 1000000000000.0, default: 200000000000.0, unit: Pa, description: Design variable bound}
                            upper_bound: *id015
                    rho:
                        type: object
                        default: {}
                        description: Material density of the tower
                        properties:
                            flag: *id001
                            lower_bound: &id016 {type: number, minimum: 1.0, maximum: 100000.0, default: 7800, unit: kg/m**3, description: Design variable bound}
                            upper_bound: *id016
            monopile: *id017
            jacket:
                type: object
                description: Design variables associated with the jacket
                default: {}
                properties:
                    foot_head_ratio:
                        type: object
                        description: Adjust the ratio of the jacket foot (bottom) radius to that of the head (top)
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id018 {type: number, minimum: 1.0, maximum: 100.0, default: 1.5, description: Design variable bound}
                            upper_bound: *id018
                    r_head:
                        type: object
                        description: Adjust the radius of the jacket head.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id019 {type: number, minimum: 0.1, maximum: 100.0, default: 5.0, unit: m, description: Design variable bound}
                            upper_bound: *id019
                    leg_diameter:
                        type: object
                        description: Adjust the diameter of the jacket legs.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id020 {type: number, minimum: 0.1, maximum: 10.0, default: 1.5, unit: m, description: Design variable bound}
                            upper_bound: *id020
                    height:
                        type: object
                        description: Overall jacket height, meters.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id021 {type: number, minimum: 0.1, maximum: 1000.0, default: 70, unit: m, description: Design variable bound}
                            upper_bound: *id021
                    leg_thickness:
                        type: object
                        description: Adjust the leg thicknesses of the jacket.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id022 {type: number, minimum: 0.001, maximum: 10.0, default: 0.1, unit: m, description: Design variable bound}
                            upper_bound: *id022
                    brace_diameters:
                        type: object
                        description: Adjust the brace diameters of the jacket.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: *id022
                            upper_bound: *id022
                    brace_thicknesses:
                        type: object
                        description: Adjust the brace thicknesses of the jacket.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: *id022
                            upper_bound: *id022
                    bay_spacing:
                        type: object
                        description: Jacket bay nodal spacing.
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id023 {type: number, minimum: 0.0, maximum: 1.0, default: 0.1, description: Design variable bound}
                            upper_bound: *id023
            floating:
                type: object
                description: Design variables associated with the floating platform
                default: {}
                properties:
                    joints:
                        type: object
                        description: Design variables associated with the node/joint locations used in the floating platform
                        default: {}
                        properties:
                            flag: *id001
                            z_coordinate: &id024
                                type: array
                                description: List of joints or members by name sets that should be adjusted. A single entry for an independent joint/member or a list of names for joints/members that are linked by symmetry
                                default: []
                                items:
                                    type: object
                                    properties:
                                        names: &id025
                                            type: array
                                            description: Joint or member names of those that are linked
                                            items: {type: string}
                                        lower_bound: {type: number, unit: m, description: Design variable bound}
                                        upper_bound: {type: number, unit: m, description: Design variable bound}
                            r_coordinate: *id024
                    members:
                        type: object
                        description: Design variables associated with the members used in the floating platform
                        default: {}
                        properties:
                            flag: *id001
                            groups:
                                type: array
                                description: Sets of members that share the same design
                                default: []
                                items:
                                    type: object
                                    properties:
                                        names: *id025
                                        diameter:
                                            type: object
                                            description: Diameter optimization of member group
                                            properties:
                                                lower_bound: *id012
                                                upper_bound: *id012
                                                constant: {type: boolean, description: Should the diameters be constant, default: false}
                                        thickness:
                                            type: object
                                            description: Thickness optimization of member group
                                            properties:
                                                lower_bound: *id013
                                                upper_bound: *id013
                                        ballast:
                                            type: object
                                            description: Ballast volume optimization of member group
                                            properties:
                                                lower_bound: {type: number, unit: m^3, description: Design variable bound, default: 0.0, minimum: 0.0}
                                                upper_bound: {type: number, unit: m^3, description: Design variable bound, minimum: 0.0, default: 100000.0}
                                        axial_joints:
                                            type: array
                                            description: List of axial joint sets in this member group that are optimized as one
                                            items:
                                                type: object
                                                default: {}
                                                properties:
                                                    names: *id025
                                                    lower_bound: {type: number, description: Design variable bound, default: 0.0, minimum: 0.0, maximum: 1.0}
                                                    upper_bound: {type: number, description: Design variable bound, minimum: 0.0, maximum: 1.0, default: 1.0}
                                        stiffeners:
                                            type: object
                                            description: Stiffener optimization of member group
                                            properties:
                                                ring:
                                                    type: object
                                                    description: Ring stiffener optimization of member group
                                                    properties:
                                                        size:
                                                            type: object
                                                            description: Ring stiffener sizing multiplier on T-shape
                                                            properties:
                                                                min_gain: &id026 {type: number, default: 0.5, unit: none, description: Lower bound on scalar multiplier that will be applied to value at control points}
                                                                max_gain: &id027 {type: number, default: 1.5, unit: none}
                                                        spacing:
                                                            type: object
                                                            description: Ring stiffener spacing along member axis
                                                            properties:
                                                                lower_bound: {type: number, unit: none, description: Design variable bound, default: 0.0, minimum: 0.0}
                                                                upper_bound: {type: number, unit: none, description: Design variable bound, default: 0.1, minimum: 0.0}
                                                longitudinal:
                                                    type: object
                                                    description: Longitudinal stiffener optimization of member group
                                                    properties:
                                                        size:
                                                            type: object
                                                            description: Longitudinal stiffener sizing multiplier on T-shape
                                                            properties:
                                                                min_gain: *id026
                                                                max_gain: *id027
                                                        spacing:
                                                            type: object
                                                            description: Longitudinal stiffener spacing around member annulus
                                                            properties:
                                                                lower_bound: {type: number, unit: rad, description: Design variable bound, default: 0.0, minimum: 0.0, maximum: 3.141592653589793}
                                                                upper_bound: {type: number, unit: rad, description: Design variable bound, default: 0.1, minimum: 0.0, maximum: 3.141592653589793}
            mooring:
                type: object
                description: Design variables associated with the mooring system
                default: {}
                properties:
                    line_length:
                        type: object
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id028 {type: number, unit: m, description: Design variable bound, default: 0.0, minimum: 0.0}
                            upper_bound: &id029 {type: number, unit: m, description: Design variable bound, minimum: 0.0}
                    line_diameter:
                        type: object
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: *id028
                            upper_bound: *id029
                    line_mass_density_coeff:
                        type: object
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: *id028
                            upper_bound: *id029
                    line_stiffness_coeff:
                        type: object
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: *id028
                            upper_bound: *id029
            TMDs:
                type: object
                description: Design variables associated with TMDs
                default: {}
                properties:
                    flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                    groups:
                        type: array
                        description: Sets of members that share the same design
                        default: []
                        items:
                            type: object
                            default: {}
                            properties:
                                names:
                                    type: array
                                    description: TMD names of those that are linked
                                    items: {type: string}
                                mass:
                                    type: object
                                    description: Mass optimization of TMD group
                                    properties:
                                        lower_bound: {type: number, default: 20000}
                                        upper_bound: {type: number, default: 20000}
                                        initial: {type: number, default: 100, description: Initial condition of TMD group}
                                        const_omega: {type: boolean, default: false, description: Keep the natural frequency constant while the mass changes?}
                                        const_zeta: {type: boolean, default: false, description: Keep the damping ratio constant while the mass changes?}
                                stiffness:
                                    type: object
                                    description: Stiffness optimization of TMD group
                                    properties:
                                        lower_bound: {type: number, default: 20000}
                                        upper_bound: {type: number, default: 20000}
                                        initial: {type: number, default: 100, description: Initial condition of TMD group}
                                damping:
                                    type: object
                                    description: Damping optimization of TMD group
                                    properties:
                                        lower_bound: {type: number, default: 20000}
                                        upper_bound: {type: number, default: 20000}
                                        initial: {type: number, default: 100, description: Initial condition of TMD group}
                                natural_frequency:
                                    type: object
                                    description: Natural frequency optimization of TMD group
                                    properties:
                                        lower_bound: {type: number, default: 20000}
                                        upper_bound: {type: number, default: 20000}
                                        initial: {type: number, default: 100, description: Initial condition of TMD group}
                                        const_zeta: {type: boolean, default: false, description: Keep the damping ratio constant while the natural frequency changes?}
                                damping_ratio:
                                    type: object
                                    description: Damping ratio optimization of TMD group
                                    properties:
                                        lower_bound: {type: number, default: 20000}
                                        upper_bound: {type: number, default: 20000}
                                        initial: {type: number, default: 100, description: Initial condition of TMD group}
    constraints:
        type: object
        default: {}
        description: Activate the constraints that are applied to a design optimization
        properties:
            blade:
                type: object
                default: {}
                description: Constraints associated with the blade design
                properties:
                    strains_spar_cap_ss:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable strain in the suction-side spar caps
                        properties:
                            flag: *id001
                            max: &id030 {type: number, description: Maximum allowable strain value, default: 0.004, minimum: 1e-08, maximum: 0.1}
                            index_start: *id002
                            index_end: *id003
                    strains_spar_cap_ps:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable strain in the pressure-side spar caps
                        properties:
                            flag: *id001
                            max: *id030
                            index_start: *id002
                            index_end: *id003
                    strains_te_ss:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable strain in the suction-side trailing edge reinforcements
                        properties:
                            flag: *id001
                            max: *id030
                            index_start: *id002
                            index_end: *id003
                    strains_te_ps:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable strain in the pressure-side trailing edge reinforcements
                        properties:
                            flag: *id001
                            max: *id030
                            index_start: *id002
                            index_end: *id003
                    tip_deflection:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable blade tip deflection towards the tower expressed as a safety factor on the parked margin.  Meaning a parked distance to the tower of 30m and a constraint value here of 1.5 would mean that 30/1.5=20m of deflection is the maximum allowable
                        properties:
                            flag: *id001
                            margin: {type: number, default: 1.4175, minimum: 1.0, maximum: 10.0}
                    t_sc_joint:
                        type: object
                        default: {}
                        description: Enforce a maximum allowable spar cap thickness, expressed as the ratio of the required spar cap thickness at the joint location to the nominal spar cap thickness.
                        properties:
                            flag: *id001
                    rail_transport:
                        type: object
                        default: {}
                        description: Enforce sufficient blade flexibility such that they can be transported on rail cars without exceeding maximum blade strains or derailment.  User can activate either 8-axle flatcars or 4-axle
                        properties:
                            flag: *id001
                            8_axle: *id001
                            4_axle: *id001
                    stall:
                        type: object
                        description: Ensuring blade angles of attacks do not approach the stall point. Margin is expressed in radians from stall.
                        default: {}
                        properties:
                            flag: *id001
                            margin: {type: number, default: 0.05233, minimum: 0.0, maximum: 0.5, unit: radians}
                    chord:
                        type: object
                        description: Enforcing the maximum chord length limit at all points along blade span.
                        default: {}
                        properties:
                            flag: *id001
                            max: {type: number, default: 4.75, minimum: 0.1, maximum: 20.0, unit: meter}
                    root_circle_diameter:
                        type: object
                        description: Enforcing the minimum blade root circle diameter.
                        default: {}
                        properties:
                            flag: *id001
                            max_ratio: {type: number, description: Maximum ratio between the recommended root circle diameter and the actual chord at blade root. The optimizer will make sure that the ratio stays below this value., default: 1.0, minimum: 0.01, maximum: 10.0}
                    frequency:
                        type: object
                        description: Frequency separation constraint between blade fundamental frequency and blade passing (3P) frequency at rated conditions using gamma_freq margin. Can be activated for blade flap and/or edge modes.
                        default: {}
                        properties:
                            flap_3P: *id001
                            edge_3P: *id001
                    moment_coefficient:
                        type: object
                        description: (EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)
                        default: {}
                        properties:
                            flag: *id001
                            min: &id031 {type: number, default: 0.15, minimum: 0.01, maximum: 5.0}
                            max: *id031
                    match_cl_cd:
                        type: object
                        description: (EXPERIMENTAL) Targeted airfoil cl/cd ratio (useful for inverse design approaches that is not recommendend for general use)
                        default: {}
                        properties:
                            flag_cl: *id001
                            flag_cd: *id001
                            filename: &id032 {type: string, description: file path to constraint data, default: ''}
                    match_L_D:
                        type: object
                        description: (EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)
                        default: {}
                        properties:
                            flag_L: *id001
                            flag_D: *id001
                            filename: *id032
                    AEP:
                        type: object
                        description: Set a minimum bound on AEP in kWh when optimizing the blade and rotor parameters
                        default: {}
                        properties:
                            flag: *id001
                            min: {type: number, units: kWh, default: 1.0, minimum: 1.0}
                    thrust_coeff:
                        type: object
                        description: (EXPERIMENTAL) Bound the ccblade thrust coefficient away from unconstrained optimal when optimizing for power, for highly-loaded rotors
                        default: {}
                        properties:
                            flag: *id001
                            lower: {type: number, minimum: 0.0}
                            upper: {type: number, minimum: 0.0}
            tower:
                type: object
                default: {}
                description: Constraints associated with the tower design
                properties:
                    height_constraint:
                        type: object
                        description: Double-sided constraint to ensure total tower height meets target hub height when adjusting section heights
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id033 {type: number, minimum: 1e-06, maximum: 10.0, default: 0.01, unit: m}
                            upper_bound: *id033
                    stress: &id036
                        type: object
                        default: {}
                        description: Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n
                        properties:
                            flag: *id001
                    global_buckling: &id037
                        type: object
                        default: {}
                        description: Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
                        properties:
                            flag: *id001
                    shell_buckling: &id038
                        type: object
                        default: {}
                        description: Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b
                        properties:
                            flag: *id001
                    slope: &id039
                        type: object
                        default: {}
                        description: Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it
                        properties:
                            flag: *id001
                    thickness_slope: &id040
                        type: object
                        default: {}
                        description: Ensure that the thickness moving up the tower at any node is always equal or less than the thickness of the section preceding it
                        properties:
                            flag: *id001
                    d_to_t: &id041
                        type: object
                        description: Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id034 {type: number, minimum: 1.0, maximum: 2000.0, default: 50.0, unit: none}
                            upper_bound: *id034
                    taper: &id042
                        type: object
                        description: Enforcing a max allowable conical frustum taper ratio per section
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: {type: number, minimum: 0.001, maximum: 1.0, default: 0.5, unit: none}
                    frequency:
                        type: object
                        description: Frequency separation constraint between all tower modal frequencies and blade period (1P) and passing (3P) frequencies at rated conditions using gamma_freq margin.
                        default: {}
                        properties:
                            flag: *id001
                    frequency_1: &id043
                        type: object
                        description: Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: &id035 {type: number, default: 0.1, minimum: 0.01, maximum: 5.0, unit: Hz}
                            upper_bound: *id035
            monopile: &id044
                type: object
                default: {}
                description: Constraints associated with the monopile design
                properties:
                    stress: *id036
                    global_buckling: *id037
                    shell_buckling: *id038
                    slope: *id039
                    thickness_slope: *id040
                    d_to_t: *id041
                    taper: *id042
                    frequency_1: *id043
                    pile_depth:
                        type: object
                        description: Ensures that the submerged suction pile depth meets a minimum value
                        default: {}
                        properties:
                            flag: *id001
                            lower_bound: {type: number, minimum: 0.0, maximum: 200.0, default: 0.0, unit: m}
                    tower_diameter_coupling:
                        type: object
                        description: Ensures that the top diameter of the monopile is the same or larger than the base diameter of the tower
                        default: {}
                        properties:
                            flag: *id001
            jacket: *id044
            hub:
                type: object
                default: {}
                properties:
                    hub_diameter:
                        type: object
                        default: {}
                        description: Ensure that the diameter of the hub is sufficient to accommodate the number of blades and blade root diameter
                        properties:
                            flag: *id001
            drivetrain:
                type: object
                default: {}
                properties:
                    lss: *id036
                    hss: *id036
                    bedplate: *id036
                    mb1: &id045
                        type: object
                        default: {}
                        description: Ensure that the angular deflection at this meain bearing does not exceed the maximum allowable deflection for the bearing type
                        properties:
                            flag: *id001
                    mb2: *id045
                    length:
                        type: object
                        default: {}
                        description: Ensure that the bedplate length is sufficient to meet desired overhang value
                        properties:
                            flag: *id001
                    height:
                        type: object
                        default: {}
                        description: Ensure that the bedplate height is sufficient to meet desired nacelle height value
                        properties:
                            flag: *id001
                    access:
                        type: object
                        default: {}
                        description: For direct-drive configurations only, ensure that the inner diameter of the nose/turret is big enough to allow human access
                        properties:
                            flag: *id001
                            lower_bound: {type: number, default: 2.0, minimum: 0.1, maximum: 5.0, unit: meter, description: Minimum size to ensure human maintenance access}
                    shaft_deflection:
                        type: object
                        default: {}
                        description: Allowable non-torque deflection of the shaft, in meters, at the generator rotor attachment for direct drive or gearbox attachment for geared drive
                        properties:
                            flag: *id001
                            upper_bound: &id046 {type: number, default: 0.0001, minimum: 1e-06, maximum: 1.0, unit: meter, description: Upper limit of deflection}
                    shaft_angle:
                        type: object
                        default: {}
                        description: Allowable non-torque angular deflection of the shaft, in radians, at the generator rotor attachment for direct drive or gearbox attachment for geared drive
                        properties:
                            flag: *id001
                            upper_bound: &id047 {type: number, default: 0.001, minimum: 1e-05, maximum: 1.0, unit: radian, description: Upper limit of angular deflection}
                    stator_deflection:
                        type: object
                        default: {}
                        description: Allowable deflection of the nose or bedplate, in meters, at the generator stator attachment
                        properties:
                            flag: *id001
                            upper_bound: *id046
                    stator_angle:
                        type: object
                        default: {}
                        description: Allowable non-torque angular deflection of the nose or bedplate, in radians, at the generator stator attachment
                        properties:
                            flag: *id001
                            upper_bound: *id047
                    ecc:
                        type: object
                        default: {}
                        description: For direct-drive configurations only, ensure that the elliptical bedplate length is greater than its height
                        properties:
                            flag: *id001
            floating:
                type: object
                default: {}
                properties:
                    operational_heel: &id048
                        type: object
                        default: {}
                        description: Ensure that the mooring system has enough restoring force to keep the heel/pitch angle below this limit
                        properties:
                            upper_bound: {type: number, default: 0.17453292519943295, minimum: 0.017453292519943295, maximum: 0.7853981633974483, unit: rad}
                    survival_heel: *id048
                    max_surge:
                        type: object
                        default: {}
                        description: Ensure that the mooring system has enough restoring force so that this surge distance, expressed as a fraction of water depth, is not exceeded
                        properties:
                            flag: *id001
                            upper_bound: {type: number, default: 0.1, minimum: 0.01, maximum: 1.0, unit: none}
                    buoyancy:
                        type: object
                        default: {}
                        description: Ensures that the platform displacement is sufficient to support the weight of the turbine system
                        properties:
                            flag: *id001
                    fixed_ballast_capacity:
                        type: object
                        default: {}
                        description: Ensures that there is sufficient volume to hold the specified fixed (permanent) ballast
                        properties:
                            flag: *id001
                    variable_ballast_capacity:
                        type: object
                        default: {}
                        description: Ensures that there is sufficient volume to hold the needed water (variable) ballast to achieve neutral buoyancy
                        properties:
                            flag: *id001
                    metacentric_height:
                        type: object
                        default: {}
                        description: Ensures hydrostatic stability with a positive metacentric height
                        properties:
                            flag: *id001
                            lower_bound: {type: number, default: 10.0, minimum: 0.0, unit: meter}
                    freeboard_margin:
                        type: object
                        default: {}
                        description: Ensures that the freeboard (top points of structure) of floating platform stays above the waterline at the survival heel offset
                        properties:
                            flag: *id001
                    draft_margin:
                        type: object
                        default: {}
                        description: Ensures that the draft (bottom points of structure) of floating platform stays beneath the waterline at the survival heel offset
                        properties:
                            flag: *id001
                    fairlead_depth:
                        type: object
                        default: {}
                        description: Ensures that the mooring line attachment depth (fairlead) is sufficiently beneath the water line that it is not exposed at the significant wave height
                        properties:
                            flag: *id001
                    mooring_surge:
                        type: object
                        default: {}
                        description: Ensures that the mooring lines have sufficient restoring force to overcome rotor thrust at the max surge offset
                        properties:
                            flag: *id001
                    mooring_heel:
                        type: object
                        default: {}
                        description: Ensures that the mooring lines have sufficient restoring force to overcome rotor thrust at the max heel offset
                        properties:
                            flag: *id001
                    mooring_tension:
                        type: object
                        default: {}
                        description: Keep the mooring line tension below its breaking point
                        properties:
                            flag: *id001
                    mooring_length:
                        type: object
                        default: {}
                        description: Keep the mooring line length within the bounds for catenary hang or TLP tension
                        properties:
                            flag: *id001
                    anchor_vertical:
                        type: object
                        default: {}
                        description: Ensure that the maximum vertical force on the anchor does not exceed limit
                        properties:
                            flag: *id001
                    anchor_lateral:
                        type: object
                        default: {}
                        description: Ensure that the maximum lateral force on the anchor does not exceed limit
                        properties:
                            flag: *id001
                    stress: *id036
                    global_buckling: *id037
                    shell_buckling: *id038
                    surge_period: &id050
                        type: object
                        default: {}
                        description: Ensure that the rigid body period stays within bounds
                        properties:
                            flag: *id001
                            lower_bound: &id049 {type: number, default: 1.0, minimum: 0.01, unit: s}
                            upper_bound: *id049
                    sway_period: *id050
                    heave_period: *id050
                    roll_period: *id050
                    pitch_period: *id050
                    yaw_period: *id050
                    Max_Offset:
                        type: object
                        default: {}
                        description: Maximum combined surge/sway offset. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 20, minimum: 0.0, maximum: 20000.0, unit: m}
            control:
                type: object
                default: {}
                properties:
                    flap_control:
                        type: object
                        description: Words TODO
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            min: {type: number, default: 0.05, minimum: 0.0, maximum: 1000000.0}
                            max: {type: number, default: 0.05, minimum: 0.0, maximum: 1000000.0}
                    rotor_overspeed:
                        type: object
                        description: (Maximum rotor speed / rated rotor speed) - 1.  Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            min: {type: number, default: 0.05, minimum: 0.0, maximum: 1.0}
                            max: {type: number, default: 0.05, minimum: 0.0, maximum: 1.0}
                    Max_PtfmPitch:
                        type: object
                        description: Maximum platform pitch displacement over all cases. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 6.0, minimum: 0.0, maximum: 30.0, unit: deg}
                    Std_PtfmPitch:
                        type: object
                        description: Maximum platform pitch standard deviation over all cases. Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 2.0, minimum: 0.0, maximum: 30.0, unit: deg}
                    Max_TwrBsMyt:
                        type: object
                        description: Maximum platform pitch displacement
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 100000.0, minimum: 0.0, maximum: 100000000.0, unit: kN*m}
                    DEL_TwrBsMyt:
                        type: object
                        description: Maximum platform pitch displacement
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 100000.0, minimum: 0.0, maximum: 100000000.0, unit: kN*m}
                    nacelle_acceleration:
                        type: object
                        description: Maximum Nacelle IMU accelleration magnitude, i.e., sqrt(NcIMUTAxs^2 + NcIMUTAys^2 + NcIMUTAzs^2). Can be computed in both RAFT and OpenFAST.  The higher fidelity option will be used when active.
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 3.2667, minimum: 0.0, maximum: 30.0, unit: m/s^2}
                    avg_pitch_travel:
                        type: object
                        description: Average pitch travel per second
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 5, minimum: 0.0, maximum: 30.0, unit: deg/s}
                    pitch_duty_cycle:
                        type: object
                        description: Number of pitch direction changes per second of simulation
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 5, minimum: 0.0, maximum: 30.0, unit: deg/s}
            damage:
                type: object
                default: {}
                properties:
                    tower_base:
                        type: object
                        description: Tower base damage constraint
                        default: {}
                        properties:
                            flag: {type: boolean, default: false, description: Activates as a design variable or constraint}
                            max: {type: number, default: 1.0, minimum: 1e-05, maximum: 30.0}
                            log: {type: boolean, default: false, description: Use the logarithm of damage as the constraint.}
            openfast_failed:
                type: object
                default: {}
                properties:
                    flag: {type: boolean, description: Constrain design to one where OpenFAST simulations don't fail_value, default: false}
    merit_figure: {type: string, description: Objective function / merit figure for optimization, default: LCOE}
    inverse_design:
        type: object
        description: For use with the inverse_design merit_figure. Specifies the reference output variable's 'prom_name' name and the desired value, accepts multiple variables. A normalized difference between the actual value and reference value is calculated for each variable. A Root Mean Square (RMS) is calculated with all variables and the optimizer minimizes the RMS. If the refernce output variable is an array, specify the element index number via "idx".
        default: {}
        additionalProperties:
            type: object
            required: [ref_value]
            optional: [indices, units]
            properties:
                ref_value:
                    type: [number, array]
                indices:
                    type: array
                    default: [0]
                units: {type: string}
    driver:
        type: object
        default: {}
        properties:
            optimization:
                type: object
                description: Specification of the optimization driver (optimization algorithm) parameters
                default: {}
                properties:
                    flag: *id001
                    tol: {type: number, description: Convergence tolerance (relative), default: 1e-06, minimum: 1e-12, maximum: 1.0, unit: none}
                    max_iter: {type: integer, description: Max number of optimization iterations, default: 100, minimum: 0, maximum: 100000}
                    max_major_iter: {type: integer, description: Max number of major optimization iterations of SNOPT, default: 10, minimum: 0, maximum: 100000}
                    max_minor_iter: {type: integer, description: Max number of minor optimization iterations of SNOPT, default: 100, minimum: 0, maximum: 100000}
                    time_limit: {type: integer, description: Max seconds of major iteration runtime for SNOPT, default: 0, minimum: 0}
                    max_function_calls: {type: integer, description: Max number of calls to objective function evaluation, default: 100000, minimum: 0, maximum: 100000000}
                    solver:
                        type: string
                        description: Optimization driver.
                        default: SLSQP
                        enum: [SLSQP, CONMIN, COBYLA, SNOPT, Nelder-Mead, GA, GN_DIRECT, GN_DIRECT_L, GN_DIRECT_L_NOSCAL, GN_ORIG_DIRECT, GN_ORIG_DIRECT_L, GN_AGS, GN_ISRES, LN_COBYLA, LD_MMA, LD_CCSAQ, LD_SLSQP, NSGA2]
                    step_size: {type: number, description: Maximum step size for finite difference approximation, default: 0.001, minimum: 1e-10, maximum: 100.0}
                    form:
                        type: string
                        description: Finite difference calculation mode
                        default: central
                        enum: [central, forward, complex]
                    step_calc:
                        type: string
                        description: Step type for computing the size of the finite difference step.
                        default: None
                        enum: [None, abs, rel_avg, rel_element, rel_legacy]
                    debug_print: &id051 {type: boolean, default: false, description: Toggle driver debug printing}
            design_of_experiments:
                type: object
                description: Specification of the design of experiments driver parameters
                default: {}
                properties:
                    flag: *id001
                    run_parallel: {type: boolean, default: true, description: Toggle parallel model runs}
                    generator:
                        type: string
                        description: Type of model input generator.
                        default: Uniform
                        enum: [Uniform, FullFact, PlackettBurman, BoxBehnken, LatinHypercube]
                    num_samples: {type: integer, description: Number of samples to evaluate model at (Uniform and LatinHypercube only), default: 5, minimum: 1, maximum: 1000000}
                    seed: {type: integer, description: Random seed to use if design is randomized, default: 2, minimum: 1, maximum: 1000000}
                    levels: {type: integer, description: Number of evenly spaced levels between each design variable lower and upper bound (FullFactorial only), default: 2, minimum: 1, maximum: 1000000}
                    criterion:
                        type: string
                        description: Descriptor of sampling method for LatinHypercube generator
                        default: center
                        enum: [None, center, c, maximin, m, centermaximin, cm, correelation, corr]
                    iterations: {type: integer, description: Number of iterations in maximin and correlations algorithms (LatinHypercube only), default: 2, minimum: 1, maximum: 1000000}
                    debug_print: *id051
            step_size_study:
                type: object
                description: Specification of the step size study parameters
                default: {}
                properties:
                    flag: *id001
                    step_sizes:
                        type: array
                        default: [0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-05, 1e-05, 5e-06, 1e-06, 5e-07, 1e-07, 5e-08, 1e-08]
                        description: List of step size values to use for the study
                    form:
                        type: string
                        description: Finite difference calculation mode
                        default: central
                        enum: [central, forward, complex]
                    of:
                        type: array
                        description: Functions of interest for which we'll compute total derivatives
                        default: []
                    wrt:
                        type: array
                        description: Design variables we'll perturb for the step size study
                        default: []
                    driver_scaling: {type: boolean, description: 'When True, return derivatives that are scaled according to either the adder and scaler or the ref and ref0 values that were specified when add_design_var, add_objective, and add_constraint were called on the model.', default: false}
    recorder:
        type: object
        default: {}
        description: Optimization iteration recording via OpenMDAO
        properties:
            flag: *id001
            file_name: {type: string, description: OpenMDAO recorder output SQL database file, default: log_opt.sql}
            just_dvs: {type: boolean, description: 'If true, only record design variables.', default: false}
            includes:
                type: array
                description: List of variables to include in recorder
                default: []
