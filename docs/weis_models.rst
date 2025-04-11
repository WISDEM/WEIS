WEIS Models
==============

WISDEM 
-------


RAFT 
-------

BEM modeling
~~~~~~~~~~~~

RAFT uses pyHAMS to solve boundary element problem to obtain potential flow solutions for floating structures. To use BEM modeling in WEIS/RAFT, there are two ways:

1. First, if you want to mesh and use BEM modeling for all members of the platform, you can just specify *potential_model_override: 2*. This will enable BEM modeling for **all** members, however, without drag force computed from strip theory. 
2. Another approach is to specify the member names of the floating platform that are intended for BEM modeling in the *potential_bem_member* option in modeling options and use *potential_model_override: 0*. This allows using BEM modeling and computing viscous drag from strip theory for the listed members; for the members not listed, strip theory will be used to provide excitation, added mass, and drag forces. Note that RAFT uses strip theory for hydrostatics irrespective of BEM modeling options.

This table below summmarized the options for potential_model_override in RAFT. Note the hydrostatics(buoyancy) are always computed using strip theory, regardless of the potential_model_override option.

+--------------------------+---------------------+-----------------------------------------------------------------------+
| potential_model_override | BEM modeling        | Strip theory                                                          | 
+==========================+=====================+=======================================================================+
| 0                        | For listed members  | Drag for listed members; all forces from the strip theory for the rest|
+--------------------------+---------------------+-----------------------------------------------------------------------+
| 1                        | No                  | All forces from the strip theory for all members                      |
+--------------------------+---------------------+-----------------------------------------------------------------------+
| 2                        | Yes                 | No                                                                    |
+--------------------------+---------------------+-----------------------------------------------------------------------+


pyHAMS
^^^^^^

pyHAMS is a python wrapper for HAMS, a boundary element method for hydrodynamic analysis of submerged structures. This is the current the default BEM solver used in WEIS/RAFT to obtain potential flow solutions for analyses and optimizations. Check out the `pyHAMS github page <https://github.com/WISDEM/pyHAMS>`_ for more details. If you set up the options correctly in WEIS/RAFT to request potential flow solutions, the BEM meshes will be generated internally and pyHAMS will be run.


OpenFAST
-----------





pCrunch
----------
WEIS uses pCrunch internally to process the time series data from OpenFAST to provide merit of figures and constraints in the optimization. pCrunch is IO and Post Processing for generic time series data of multibody aeroelastic wind turbine simulations. Readers are provided for OpenFAST outputs, but the analysis tools are equally applicable to HAWC2, Bladed, QBlade, ADAMS, or other tools. pCrunch attempts to capture the best of legacy tools MCrunch, MLife, and MExtremes, while also taking inspiration from other similar utilities available on Github.  Some statistics pCrunch provides are: extreme values, fatigue calculation based on rainflow cycle counting and miner’s law, AEP calculation. pCrunch is usually installed as part of WEIS. For separate installation guide and more details on pCrunch, check out the `pCrunch github page 
<https://github.com/NREL/pCrunch/tree/master>`_. pCrunch provides examples using the two primary classes AeroelasticOutputs and Crunch, see 
`AeroelasticOutputs example <https://github.com/NREL/pCrunch/blob/master/examples/aeroelastic_output_example.ipynb>`_ and 
`Crunch example <https://github.com/NREL/pCrunch/blob/master/examples/crunch_example.ipynb>`_.

