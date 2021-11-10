Model Structure
===============




RAFT Model Objects
------------------

RAFT represents a floating wind turbine system as a collection of different object types, as illustrated below.

.. image:: /images/objects.JPG
    :align: center
	
Each of these objects corresponds to a class in python. The Model class contains 
the full RAFT representation of a floating wind system. Within it, the FOWT class
describes the floating platform and turbine, while the mooring sytem is handled by
a separate model, `MoorPy <https://moorpy.readthedocs.io>`__. Within the FOWT class,
the floating platform and turbine tower are represented as a single rigid body 
composed of multiple Member objects that describe linear cylindrical or rectangular
geometries. The turbine rotor-nacelle assembly is represented by a Rotor class that 
describes lumped mass properties and distributed aerodynamic properties. These RAFT
objects are further described below.	

Model
^^^^^
An entire floating offshore wind turbine array can be stored in a "Model" object. A model includes a list of the FOWTs in the array,
the array-wide mooring system, and various modeling parameters, such as the design load cases (DLCs). These attributes are then run
to solve for the response amplitude operators (RAOs) of each FOWT in the array. There should only be one Model object per RAFT instance.



FOWT
^^^^
A FOWT object holds all components of a floating wind turbine platform other than the mooring system. This includes the floating substructure, the turbine tower, and the turbine RNA.
The floating substructure and turbine tower can be described as a list of cylindrical or rectangular Members objects. The RNA is represented as a Rotor object.

A FOWT object has many capabilities. It can compute the static properties of each member in the FOWT to fill in its 6x6 mass and
hydrostatic matrices. It can generate a mesh for each member to be used calculate the hydrodynamic properties of each member through
the use of the BEM solver, `pyHAMS <https://github.com/WISDEM/pyHAMS>`_, or it can use a Morison equation approximation for each member.
It can also calculate the hydrodynamic damping and forcing as a result of nonlinear drag, which is dependent upon the platform positions.
Lastly, it can calculate the aerodynamic added mass, damping, and forcing matrices of the rotor. 


Member
^^^^^^
A Member is any cylindrical or rectangular component of a FOWT. They are assumed to be completely rigid objects with no flexibility.
RAFT currently does not support structural flexibility.

There are many attributes that are used to describe members: bottom and top node positions, shape, length, diameters (or side lengths),
thicknesses, ballast properties, end cap and bulkheads, and drag and added mass coeffficients. More details on Members can be found in the
Theory section.



Rotor
^^^^^
The Rotor class is used to describe the turbine rotor-nacelle assembly to calculate the lumped mass and distributed 
aerodynamic and control properties of the FOWT. It sets up attributes like the hub height, shaft tilt angle, 
hub wind speeds, rotor RPM, and blade pitch to create CCAirfoil and CCBlade objects
(`CCBlade documentation <https://wisdem.readthedocs.io/en/latest/wisdem/ccblade/index.html>`_). These objects return 
the loads and derivatives of the aerodynamics to extract the thrust and torque with respect to wind speed, rotor RPM, and blade pitch, 
as well as the aerodynamic forcing, added mass, and damping.





RAFT Heirarchy and Interfaces
------------------------------------

Each RAFT object is described by a class in a separate source file, and these classes interact following
a set heirarchy. Some of the classes also interact with external models. These relationships are 
outlined below.

.. image:: /images/hierarchy.JPG
    :align: center


`MoorPy <https://moorpy.readthedocs.io/en/latest/>`__ is a quasi-static mooring system modeler 
that provides the mooring system representation for RAFT. It supports floating platform linear
hydrostatic properties and constant external loads. For each load case, RAFT provides MoorPy
the system hydrostatic properties and mean wind loads, and MoorPy calculates the mean floating
platform offsets for RAFT.


`CCBlade <https://wisdem.readthedocs.io/en/latest/wisdem/ccblade/index.html>`_ is a blade-element-momentum 
theory model that provides the aerodynamic coefficients for RAFT. These coefficients account for the mean
aerodynamic forces and moments of the wind turbine rotor at each wind speed, as well as the derivatives 
of those forces and moments with respect to changes in relative wind speed, rotor speed, and blade pitch
angle. From this information, RAFT can compute the overall aerodynamic effects of the system's rigid-body
response, including accounting for the impact of turbine control.

`pyHAMS <https://github.com/WISDEM/pyHAMS>`_ is a Python wrapper of the HAMS (Hydrodynamic Analysis of 
Marine Structures) tool for boundary-element-method solution of the potential flow problem. HAMS is 
developed by Yingyi Liu and available at https://github.com/YingyiLiu/HAMS. HAMS provides the option
for RAFT to use potential-flow hydrodynamic coefficients instead of, or in addition to, strip theory.

