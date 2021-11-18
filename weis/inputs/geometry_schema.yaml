$schema: "http://json-schema.org/draft-07/schema#"
$id: WEIS_add-ons_geom
title: WEIS geometry ontology add-ons beyond WISDEM ontology
description: Ontology definition for wind turbines as defined in WP1 of IEA Wind Task 37 - Phase II
type: object
properties:
    environment:
        type: object
        required:
            - air_density
            - air_dyn_viscosity
            - air_speed_sound
            - shear_exp
        optional:
            - gravity
            - weib_shape_parameter
            - water_density
            - water_dyn_viscosity
            - water_depth
            - soil_shear_modulus
            - soil_poisson
            - air_pressure
            - air_vapor_pressure
        properties:
            gravity:
                type: number
                description: Gravitational acceleration
                unit: m/s/s
                minimum: 0
                maximum: 100.0
                default: 9.80665
            air_density:
                type: number
                description: Density of air.
                unit: kg/m3
                minimum: 0
                maximum: 1.5
                default: 1.225
            air_dyn_viscosity:
                type: number
                description: Dynamic viscosity of air.
                unit: kg/(ms)
                minimum: 0
                maximum: 2.e-5
                default: 1.81e-5
            air_pressure:
                type: number
                description: Atmospheric pressure of air
                unit: kg/(ms^2)
                minimum: 0
                maximum: 1.e+6
                default: 1.035e+5
            air_vapor_pressure:
                type: number
                description: Vapor pressure of fluid
                unit: kg/(ms^2)
                minimum: 0
                maximum: 1.e+6
                default: 1.7e+3
            weib_shape_parameter:
                type: number
                description: Shape factor of the Weibull wind distribution.
                unit: none
                minimum: 1
                maximum: 3
                default: 2.
            air_speed_sound:
                type: number
                description: Speed of sound in air.
                unit: m/s
                minimum: 330.
                maximum: 350.
                default: 340.
            shear_exp:
                type: number
                description: Shear exponent of the atmospheric boundary layer.
                unit: none
                minimum: 0
                maximum: 1
                default: 0.2
            water_density:
                type: number
                description: Density of water.
                unit: kg/m3
                minimum: 950
                maximum: 1100
                default: 1025.0
            water_dyn_viscosity:
                type: number
                description: Dynamic viscosity of water.
                unit: kg/(ms)
                minimum: 1.e-3
                maximum: 2.e-3
                default: 1.3351e-3
            water_depth:
                type: number
                description: Water depth for offshore environment.
                unit: m
                minimum: 0.0
                maximum: 1.e+4
                default: 0.0
            soil_shear_modulus:
                type: number
                description: Shear modulus of the soil.
                unit: Pa
                minimum: 100.e+6
                maximum: 200.e+6
                default: 140.e+6
            soil_poisson:
                type: number
                description: Poisson ratio of the soil.
                unit: none
                minimum: 0
                maximum: 0.6
                default: 0.4
    TMDs:
        type: array
        description: Ontology definition for TMDs
        items:
            type: object
            required:
                - name
                - component
                - location
                - mass
                - stiffness
                - damping
            properties:
                name:
                    description: Unique name of the TMD
                    type: string
                component:
                    description: Component location of the TMD (tower or platform)
                    type: string
                    # enum:
                    #     - tower
                    #     - platform
                location:
                    description:  Location of TMD in global coordinates
                    type: array
                    items:
                        type: number
                        minIteams: 3
                        maxItems: 3
                mass:
                    description: Mass of TMD
                    type: number
                    unit: kg
                    default: 0
                stiffness:
                    description: Stiffness of TMD
                    type: number
                    unit: N/m
                    default: 0
                damping:
                    description: Damping of TMD
                    type: number
                    unit: (N/(m/s))
                    default: 0
                X_DOF:
                    description: Dof on or off for StC X
                    type: boolean
                    default: False
                Y_DOF:
                    description: Dof on or off for StC Y
                    type: boolean
                    default: False
                Z_DOF:
                    description: Dof on or off for StC Z
                    type: boolean
                    default: False
                natural_frequency:
                    description: Natural frequency of TMD, will overwrite stiffness (-1 indicates that it's not used)
                    type: number
                    unit: rad/s
                    default: -1
                damping_ratio:
                    description: Daming ratio of TMD, will overwrite damping (-1 indicates that it's not used)
                    type: number
                    unit: non-dimensional
                    default: -1
                preload_spring:
                    description: Ensure that equilibrium point of the TMD is at `location` by offseting the location based on the spring constant
                    type: boolean
                    default: True

                

        
definitions:
    distributed_data:
        grid_nd:
            type: array
            description: Grid along a beam expressed non-dimensional from 0 to 1
            items:
                type: number
                unit: none
                minItems: 2
                minimum: 0.0
                maximum: 1.0
                uniqueItems: true
        grid_al:
            type: array
            description: Grid along an arc length, expressed non dimensionally where 0 is the leading edge, -1 is the trailing edge on the pressure side and +1 the trailing edge on the pressure side
            items:
                type: number
                unit: none
                minItems: 2
                minimum: -1.0
                maximum: 1.0
                uniqueItems: true
        grid_aoa:
            type: array
            description: Grid of angles of attack to describe polars
            items:
                type: number
                unit: radians
                minItems: 2
                minimum: -3.14159265359
                maximum:  3.14159265359
                uniqueItems: true
        polar_coeff:
            type: array
            description: Lift, drag and moment coefficients
            items:
                type: number
                unit: none
                minItems: 2
                uniqueItems: false
        strings:
            type: array
            items:
                type: string
                minItems: 2
                uniqueItems: false
        nd:
            type: array
            description: Non dimensional quantity described along a beam and expressed non-dimensional
            items:
                type: number
                unit: none
                minItems: 2
                uniqueItems: false
        length:
            type: array
            description: Length quantity described along a beam, expressed in meter
            items:
                type: number
                unit: meter
                minItems: 2
                uniqueItems: false
        angle:
            type: array
            description: Angle quantity along a beam, expressed in radians
            items:
                type: number
                unit: radians
                minItems: 2
                uniqueItems: false
        mass_length:
            type: array
            description: Mass per unit length along a beam, expressed in kilogram per meter
            items:
                type: number
                unit: kg/m
                minItems: 2
                uniqueItems: false
        area:
            type: array
            description: Cross sectional area along a beam, expressed in meter squared
            items:
                type: number
                unit: m2
                minItems: 2
                uniqueItems: false
                description: Cross sectional area
        elast_mod:
            type: array
            description: Modulus of elasticity of a material along a beam, expressed in Newton over meter squared
            items:
                type: number
                unit: N m2
                minItems: 2
                uniqueItems: false
                description: Modulus of elasticity
        shear_mod:
            type: array
            description: Shear modulus of elasticity of a material along a beam, expressed in Newton over meter squared
            items:
                type: number
                unit: N/m2
                minItems: 2
                uniqueItems: false
                description: Shear modulus of elasticity
        area_moment:
            type: array
            description: Area moment of inertia of a section along a beam, expressed in meter to the power of four
            items:
                type: number
                unit: m4
                minItems: 2
                uniqueItems: false
                description: Area moment of inertia
        mass_moment:
            type: array
            description: Mass moment of inertia of a section along a beam, expressed in kilogram times meter squared per meter
            items:
                type: number
                unit: kg*m2/m
                minItems: 2
                uniqueItems: false
                description: Mass moment of inertia per unit span
        tors_stiff_const:
            type: array
            description: Torsional stiffness constant with respect to ze axis at the shear center [m4/rad]. For a circular section only this is identical to the polar moment of inertia
            items:
                type: number
                unit: m4/rad
                minItems: 2
                uniqueItems: false
        shear_stiff:
            type: array
            description: Shearing stiffness along the beam
            items:
                type: number
                unit: N
                minItems: 2
                uniqueItems: false
        axial_stiff:
            type: array
            description: Axial stiffness EA along the beam
            items:
                type: number
                unit: N
                minItems: 2
                uniqueItems: false
        bending_stiff:
            type: array
            description: Bending stiffness E11-E22 along the beam
            items:
                type: number
                unit: N/m2
                minItems: 2
                uniqueItems: false
        tors_stiff:
            type: array
            description: Torsional stiffness GJ along the beam
            items:
                type: number
                unit: N/m2
                minItems: 2
                uniqueItems: false
        nd_arc_position:
            type: object
            description: non-dimensional location of the point along the non-dimensional arc length
            properties:
                grid:
                    $ref: "#/definitions/distributed_data/grid_nd"
                values:
                    $ref: "#/definitions/distributed_data/grid_al"
                fixed:
                    type: string
                    description: Name of the layer to lock the edge
        offset:
            type: object
            description: dimensional offset in respect to the pitch axis along the x axis, which is the chord line rotated by a user-defined angle. Negative values move the midpoint towards the leading edge, positive towards the trailing edge
            required:
                - grid
                - values
            properties:
                grid:
                    $ref: "#/definitions/distributed_data/grid_nd"
                values:
                    $ref: "#/definitions/distributed_data/length"
        rotation:
            type: object
            description: rotation of the chord axis around the pitch axis
            properties:
                grid:
                    $ref: "#/definitions/distributed_data/grid_nd"
                values:
                    $ref: "#/definitions/distributed_data/angle"
                fixed:
                    type: string
                    description: Name of the layer to lock the edge
        axis_coordinates:
            type: object
            description: The reference system is located at blade root, with z aligned with the pitch axis, x pointing towards the suction sides of the airfoils (standard prebend will be negative) and y pointing to the trailing edge (standard sweep will be positive)
            required:
                - x
                - y
                - z
            properties:
                x:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                z:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
    beam:
        timoschenko_hawc:
            type: object
            description: Timoschenko beam as in HAWC2
            required:
                - reference_axis
                - A
                - E
                - G
                - I_x
                - I_y
                - K
                - dm
                - k_x
                - k_y
                - pitch
                - ri_x
                - ri_y
                - x_cg
                - x_e
                - x_sh
                - y_cg
                - y_e
                - y_sh
            properties:
                reference_axis:
                    $ref: "#/definitions/distributed_data/axis_coordinates"
                A:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/area"
                E:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/elast_mod"
                G:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/shear_mod"
                I_x:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/area_moment"
                I_y:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/area_moment"
                K:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/tors_stiff_const"
                dm:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/mass_length"
                k_x:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/nd"
                k_y:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/nd"
                pitch:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/angle"
                ri_x:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                ri_y:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                x_cg:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                x_e:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                x_sh:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_cg:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_e:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_sh:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
        cp_lambda_beam:
            type: object
            description: Geometrically exact beams with simplified properties
            required:
                - reference_axis
                - T11
                - T22
                - EA
                - E11
                - E22
                - GJ
                - x_ce
                - y_ce
                - dm
                - delta_theta
                - x_sh
                - y_sh
                - J1
                - J2
                - J3
                - x_cg
                - y_cg
            properties:
                reference_axis:
                    $ref: "#/definitions/distributed_data/axis_coordinates"
                T11:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/shear_stiff"
                T22:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/shear_stiff"
                EA:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/axial_stiff"
                E11:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/bending_stiff"
                E22:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/bending_stiff"
                GJ:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/tors_stiff"
                x_ce:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_ce:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                dm:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/mass_length"
                delta_theta:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/angle"
                x_sh:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_sh:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                J1:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/mass_moment"
                J2:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/mass_moment"
                J3:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/mass_moment"
                x_cg:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
                y_cg:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            $ref: "#/definitions/distributed_data/length"
        six_x_six:
            type: object
            required:
                - reference_axis
                - stiff_matrix
            properties:
                reference_axis:
                    $ref: "#/definitions/distributed_data/axis_coordinates"
                stiff_matrix:
                    type: object
                    required:
                        - grid
                        - values
                    properties:
                        grid:
                            $ref: "#/definitions/distributed_data/grid_nd"
                        values:
                            type: array
                            items:
                                type: array
                                description: Stiffness matrix 6x6, only upper diagonal reported line by line (21 elements), specified at each grid point
                                minItems: 21
                                maxItems: 21
                                uniqueItems: false

    actuator:
        type: string
        description: Actuator used as control output
        enum:
            - pitch
            - torque   #called generator above, keep consistent?
            - tower_TMD
            - hull_TMD
            - active_tension
            - passive_weather_vane
            - passive_buoy_can
    sensor:
        type: string
        description: Sensor used as control input, could be any OpenFAST output (in Simluink), enumerating avrSWAP now  # DZ: within ROSCO, probably not...needs to be in avrSWAP
        enum:
            - gen_speed
            - nac_IMU
            - wind_speed_estimate
            - gust_measure
            - RootMyc1
            - RootMyc2
            - RootMyc3
            - RootMyT   # MBC of above
            - RootMyY   # MBC of above
            - azimuth
            - YawBrTAxp
            - YawBrTAyp
            - RootMxc1
            - RootMxc2
            - RootMxc3
            - LSSTipMya
            - LSSTipMza
            - LSSTipMxa
            - LSSTipMys
            - LSSTipMzs
            - YawBrMyn
            - YawBrMzn
            - NcIMURAxs  # avrSWAP names don't always match OpenFAST output names
            - NcIMURAzs
            # No platform info from avrSWAP, will they be added?
    filter:
        type: object
        description: Linear filter, could be a LPF, HPF, NF, INF, or user_defined
        required:
            - filt_type
            - filt_def
        filt_type:
            type: string
            description: Type of filter used, could be a LPF, HPF, NF, INF, or user_defined
            enum:
                - LPF
                - HPF
                - NF
                - INF
                - user_defined
        filt_def:
            LPF:
                type: object
                description: Low pass filter
                required:
                    - omega
                    - order
                optional:
                    - damping
            HPF:
            NF:
            INF:
            user_defined:
                type: object
                description: User defined filter
                required:
                    - num
                    - den
                optional:
                    - dt
                num:
                    type: array
                    description: Numerator coefficients of linear filter
                    items:
                        type: number
                        unit: none
                        minItems: 0
                        uniqueItems: false
                den:
                    type: array
                    description: Numerator coefficients of linear filter
                    items:
                        type: number
                        unit: none
                        minItems: 1
                        uniqueItems: false
                dt:
                    type: number
                    description: Sampling rate of filter, -1 for continuous   # DZ: will probably convert all to dt of simulation... not sure how to handle this
                    minimum: -1

    state_space:
        type: object
        description: Linear state space model
        required:
            - ss_A
            - ss_B
            - ss_C
            - ss_D
        ss_A:
            type: array
            description: A matrix of linear state space model, flattened with n_states^2 elements
            items:
                type: number
                unit: none
                minItems: 1
                uniqueItems: false
        ss_B:
            type: array
            description: B matrix of linear state space model, flattened with n_states x n_inputs elements
            items:
                type: number
                unit: none
                minItems: 1
                uniqueItems: false
        ss_C:
            type: array
            description: C matrix of linear state space model, flattened with n_outputs x n_states elements
            items:
                type: number
                unit: none
                minItems: 1
                uniqueItems: false
        ss_D:
            type: array
            description: D matrix of linear state space model, flattened with n_outputs x n_inputs elements
            items:
                type: number
                unit: none
                minItems: 1
                uniqueItems: false
        # DZ: might be a good idea to check the number of states
        ss_dt:
            type: number
            description: Sampling rate of filter, -1 for continuous   # DZ: will probably convert all to dt of simulation... not sure how to handle this
            minimum: -1

    timeseries:
        type: object
        description: Array of time, value pairs  # DZ: can we check that the number of elements are equal?
        required:
            - time
            - value
        optional:
            - filename
        time:
            type: array
            description: Time in timeseries
            items:
                type: number
                unit: seconds
                minItems: 1
                uniqueItems: true   # probably?
        value:
            type: array
            description: Value in timeseries
            items:
                type: number
                unit: none  #any
                minItems: 1
                uniqueItems: false
                # can we check that number of elements are equal?
        filename:
            type: string
            description: Name of file with timeseries data

    activator:
        type: object
        description: Gain used to enable/disable control elements, can be used partially
        required:
            - wind_speeds
            - act_gain
        wind_speeds:
            type: array
            description: Array of wind speed breakpoints for activators
            items:
                type: number
                unit: m/s
                minItems: 1
                uniqueItems: true
        act_gain:
            type: array
            description: Array of gains from 0 to 1, enabling/disabling control element
            items:
                type: number
                unit: none
                minItems: 1
                uniqueItems: false
                minimum: 0
                maximum: 1
