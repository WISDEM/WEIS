Model Structure
===============


MoorPy Objects
---------------

MoorPy organizes a mooring system into objects following a very similar approach as `MoorDyn <http://moordyn.readthedocs.io>`_. 
Currently it supports three objects--Lines, Points, and Bodies--which are described below. Rod objects may be added later.

Lines
^^^^^

MoorPy uses a quasi-static catenary model to calculate the profile and tensions in 
a uniform length of a mooring line.
Any given Line has constant/uniform properties of unstretched length, diameter, 
density, and Young's modulus.  Different Lines can have different sets of properties, 
and they can be connected together at the ends, enabling mooring systems with 
interconnected lines or with line assemblies featuring step changes in properties. 


MoorPy, like MoorDyn, keeps a dictionary of line types to describe the cross-sectional 
(or per-meter) properties of the mooring lines. Each line type has an alphanumeric name
to identify it, and contains the key properties needed for a quasi-static model: wet
weight and stiffness (EA). Each Line must be assigned to have one of the line types.

Each end of a Line most be connected to a Point object (described next). 
Multi-segmented mooring lines (where different sets of properties apply over different
parts of the line) can be achieved by using multiple Line objects, with different line,
with free Point objects connecting the Lines where the properties change.


Points
^^^^^^
.. _points:

Point objects attach to the ends of Lines and can be used to connect Lines to other things
or to each other. (In MAP and older versions of MoorDyn, these objects were called Connections).
A Point has three degrees of freedom and can have any number of Lines attached to it. 
There are three types of Points:

- **Fixed**: their location is fixed to ground (stationary) or a Body object. 
  They can be used as anchor points or as a way to attach mooring Lines to a Body.
- **Coupled**: they move under the control of the calling program/script.  
  They can be used as fairlead connections when the platform is modeled externally.
- **Free**: they are free to move according to the forces acting on them, which includes
  the tensions of attached lines as well as their own self weight and buoyancy, if applicable.  

Free Points facilitate more advanced mooring systems. They can be used to connect two 
or more mooring lines together, to create multi-segmented lines or junctions such as in a 
bridle mooring configuration. If a free Point is given nonzero volume or mass properties,
it can also represent a clump weight or float. 

Key optional Point properties that can be passed as named arguments when creating a Point or
accessed afterward are as follows:

 - **m**: mass [kg]. The default is 0.
 - **v**: volume [m^3]. The default is 0.


Bodies
^^^^^^

Body objects provide a generic 6 DOF rigid-body representation based on a lumped-parameter model of translational 
and rotational properties.  Point objects can be added to Bodies at any location to facilitate the attachment of
mooring lines. Bodies are most commonly used to represent a floating platform. For this application, bodies can be
given hydrostatic properties through waterplane-area and metacenter-location parameters. Bodies can also have external
constant forces and moments applied to them. In this way, a Body can represent the complete linear hydrostatic behavior
of a floating platform including a wind turbine's steady thrust force. 
Bodies, like Points, can be fixed (not very useful), free, or coupled.


Key optional Body properties that can be passed as named arguments when creating a Point or
accessed afterward are as follows:

 - **m**: mass, centered at CG [kg]. The default is 0.
 - **v**: volume, centered at reference point [m^3]. The default is 0.
 - **rCG**: center of gravity position in body reference frame [m]. The default is np.zeros(3).
 - **AWP**: waterplane area - used for hydrostatic heave stiffness if nonzero [m^2]. 
   The default is 0.
 - **rM**: coorindates or height of metacenter relative to body reference frame [m]. 
   The default is np.zeros(3).
 - **f6Ext**: applied external forces and moments vector in global orientation 
   (not including weight/buoyancy) [N]. The default is np.zeros(6).
   

The MoorPy System Object
------------------------

The System object in MoorPy is used to hold all the objects being simulated as well as additional
information like the line types dictionary, water density, and water depth. Most user
interactions with the mooring system are supported through methods of the System class. 
These methods are listed on the :ref:`API` page. More description to be added here in future...
