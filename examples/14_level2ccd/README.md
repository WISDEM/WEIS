This example demonstrates how to use Level 2, linearized OpenFAST, in a CCD study.

We use the linearized models developed in example 12, with the LPV modeling scheme and open-loop optimal control problem formulation discussed in example 13 to setup and solve a nested CCD problem.

The objective of the CCD problem is to minimize the LCOE
```
merit_figure: weis_lcoe
```

The nested approach used in this examples involves solving a bi-level optimization problem, which comprises of an 'outer-loop' problem that seeks to find the optimal set of plant variables that minimizes the capital cost, and an 'inner-loop' problem that seeks to identify the optimal control inputs which maximizes the AEP. 

The optimization driver in WEIS is used to setup and solve the outer-loop problem.
The plant variables used in this examples are the column spacing between the inner and outer columns in the semisubmersible platform, and the ballast volume of all the columns.

```
design_variables:
  floating:
        joints:
            flag: True
            r_coordinate:
                - names: [col1_keel, col1_freeboard, col2_keel, col2_freeboard, col3_keel, col3_freeboard]
                  lower_bound: 36.
                  upper_bound: 66.
        members:
            flag: True
            groups:
                - names: [main_column]
                  ballast:
                      lower_bound: 350
                      upper_bound: 650
                - names: [column1,column2,column3]
                  ballast:
                      lower_bound: 350
                      upper_bound: 650

```
Only simple bound constraints are placed on the plant variables as indicated by `lower_bound` and `upper_bound`.

An open-loop optimal control problem is solved to identify the optimal trajectories for generator torque and blade pitch, similar to the studies outlined in example 13.

For every design iteration in the outer-loop, OpenFAST is called to evaluate the linearized models for that design.
These linearized models are used to setup and solve inner-loop optimal control problems for each load case.
The average power generated for each load case is used to calculate the AEP.

The following load cases are used in this example:

```
DLC_driver:
    DLCs:
        - DLC: "1.1"
          ws_bin_size: 2
          wind_speed: [4,6,8,10,12,14,16,18,20,22,24] 
          wave_height: [8.]
          wave_period: [2.]
          #wind_seeds: [3,4] #[1,2,3,4,5]
          n_seeds: 1
          analysis_time: 700.
          transient_time: 100.
          turbulent_wind:
              HubHt: 150.0
              Clockwise: True
              RefHt: 150.0
              PLExp: 0.11
```

For more information regarding the CCD problem solved in this example, please refer to:

[1] A. K. Sundarrajan, Y. H. Lee, J. T. Allison, D. S. Zalkind, D. R. Herber. 'Open-loop control co-design of semisubmersible floating offshore wind turbines using linear parameter-varying models.' (to appear) ASME Journal of Mechanical Design, 146(4), p. 041704, Apr 2024. doi: [https://doi.org/10.1115/1.4063969](https://doi.org/10.1115/1.4063969)

For more information regarding different solution strategies for CCD problems (nested vs. simultaneous) , please refer to the following publications:

[2] D. R. Herber, J. T. Allison. 'Nested and simultaneous solution strategies for general combined plant and control design problems.' ASME Journal of Mechanical Design, 141(1), p. 011402, Jan 2019. doi: [10.1115/1.4040705](10.1115/1.4040705)

[3] A. K. Sundarrajan, D. R. Herber. 'Towards a fair comparison between the nested and simultaneous control co-design methods using an active suspension case study.' In American Control Conference, May 2021. doi: [10.23919/ACC50511.2021.9482687](10.23919/ACC50511.2021.9482687)


Please reach out to Athul.Sundarrajan@colostate.edu and Daniel.Herber@colostate.edu for more information and clarifications.
