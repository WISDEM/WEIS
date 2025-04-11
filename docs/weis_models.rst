WEIS Models
==============

WISDEM 
-------

WEIS wraps the systems engineering framework WISDEM. 
Both WISDEM and WEIS are built on top of `OpenMDAO <https://openmdao.org>_` and therefore share much of the same architecture and syntax. 
OpenMDAO is a generic open-source optimization framework implemented in Python by NASA. 
WISDEM has been developed by NREL researchers on top of OpenMDAO since 2014. 
Both WISDEM and WEIS allow users to optimize the design of single turbines. 
The key difference between WISDEM and WEIS is that WISDEM implements steady-state models, such as the steady-state blade element momentum solver CCBlade, the cross-sectional solver PreComp, and the steady-state beam solver frame3dd, whereas WEIS includes solvers capable of capturing dynamic effects using OpenFAST and RAFT. 
WEIS also includes the turbine controller ROSCO.
WISDEM and WEIS share the same three input files: the geometry yaml, which describes the turbine, the modeling yaml, which lists all modeling-related options, and the analysis yaml, which builds the optimization setup and lists design variables, figure of merit, constraints, and optimization settings. 
The only difference in inputs files between WISDEM and WEIS is that the modeling options yaml in WEIS has many more optional inputs, as it contains all the flags and settings that go into OpenFAST, RAFT, and ROSCO.

Within WEIS, WISDEM models the turbine and generates all those quantities that are needed by OpenFAST and RAFT. 
For the turbine rotor, WISDEM interpolates aerodynamic quantities and builds the lifting line and associated polars. 
WISDEM runs PreComp and computes the equivalent elastic properties of the blades. 
These are then formatted for the structural dynamic solvers BeamDyn and ElastoDyn, including the mode shapes. 
WISDEM computes the equivalent elastic properties of the nacelle components, of the tower, and of the offshore substructure. 
WISDEM connects floating platform joints using members along with the parameters needed for the hydrodynamic solvers in OpenFAST and RAFT. 
WISDEM finally computes the regulation trajectory for the turbine, which is used by ROSCO, so that WEIS can automatically create the DISCON input file.
In simple terms, WISDEM can be seen as the first tool for the conceptual design of a wind turbine. 
Once a first conceptual design is generated, WEIS becomes a critical tool for more sophisticated studies investigating the dynamic performance of the turbine, whether land-based or offshore.

WEIS users are encouraged to familiarize themselves with WISDEM through its documentation, which is available at `https://wisdem.readthedocs.io/en/master/ <https://wisdem.readthedocs.io/en/master/>_`. 
The list of WISDEM examples is often a good starting point.


RAFT 
-------


OpenFAST
-----------
