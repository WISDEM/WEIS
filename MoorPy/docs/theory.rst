.. _theory:

Theory and References
=====================

The theory behind MoorPy is in the process of being written up and published. 
Please check back later or contact us if in need of a specific clarification.



Points
^^^^^^

Points in MoorPy, like MoorDyn, are treated as point masses (3 degrees of freedom) 
with point-mass, buoyancy, and omnidirectional hydrodynamic properties.

However, MoorPy provides an additional feature to represent linear hydrostatic
properties so that a Point can be used to represent a surface-piercing object
that stays vertical and has a constant waterplane area over its height. The 
zSpan parameter defines the lower and upper extents of the point's 
height, relative to the point coordinate, r. This provides a constant 
hydrostatic stiffness when the Point crosses the free surface, but has no
effect the rest of the time.
