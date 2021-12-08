Model Structure
===============


MoorPy Objects
---------------

MoorPy organizes a mooring system into objects following a very similar approach as `MoorDyn <http://moordyn.readthedocs.io>`_. 
Currently it supports three objects--Lines, Points, and Bodies--which are described below. Rod objects may be added later.

Lines
^^^^^

MoorPy uses a quasi-static catenary model to calculate the profile and tensions in a uniform length of a mooring line.
Any given Line has constant/uniform properties of unstretched length, diameter, density, and Young's modulus.  
Different Lines can have different 
sets of properties, and they can be connected together at the ends, enabling mooring systems with interconnected lines 
or with line assemblies featuring step changes in properties. 



Points
^^^^^^
.. _points:

The ends of each mooring line are defined by Point objects. There are three types:

- 1: Fixed nodes have a certain location and never move.  They can be used as anchor points. OR attached to a platform for platform-centric coupling
- -1: Coupled nodes can move under the control of an outside program.  They can be used as fairlead connections.
- 0: Free nodes are not fixed in space but rather are moved according to the forces acting on them.  

Points can be used to connect two or more mooring lines together.  The forces they experience can include the forces from the attached 
mooring lines as well as their own weight and buoyancy forces.  


Bodies
^^^^^^

Body objects provide a generic 6 DOF rigid-body representation based on a lumped-parameter model of translational 
and rotational properties.  Point objects can be added to Bodies at any location to facilitate the attachment of
mooring lines. Bodies are most commonly used to represent a floating platform. For this application, bodies can be
given hydrostatic properties through waterplane-area and metacenter-location parameters.Bodies can also have external
constant forces and moments applied to them. In this way, a Body can represent the complete linear hydrostatic behavior
of a floating platform including a wind turbine's steady thrust force.
