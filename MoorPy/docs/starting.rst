Getting Started
===============


Prerequisites
^^^^^^^^^^^^^

- Python 3
- The following Python packages: NumPy, MatPlotLib, yaml, scipy


Installation
^^^^^^^^^^^^

Clone the `MoorPy GitHub repository <https://github.com/NREL/MoorPy>`_.

To install for development use:

run ```python setup.py develop``` or ```pip install -e .``` from the command line in the main MoorPy directory.

To install for non-development use:

run ```python setup.py``` or ```pip install .``` from the command line in the main MoorPy directory.


Examples
^^^^^^^^

The MoorPy repository has an examples folder containing two example scripts:

- manual_system.py constructs a mooring system from scratch using functions for creating each line and attachment point.

- imported_system.py creates a mooring system by reading an included MoorDyn-style input file.

Running either of these scripts will produce a basic mooring system model that can be used in further analysis.


Creating a MoorPy System Manually
---------------------------------

MoorPy has internal functions to facilitate the orderly creation of a mooring system. The following
gives an example of how they work (from manual_system.py).

.. code-block:: python


  # ----- choose some system geometry parameters -----
  
  depth     = 600                             # water depth [m]
  angles    = np.radians([60, 180, 300])      # line headings list [rad]
  rAnchor   = 1600                            # anchor radius/spacing [m]
  zFair     = -21                             # fairlead z elevation [m]
  rFair     = 20                              # fairlead radius [m]
  lineLength= 1800                            # line unstretched length [m]
  typeName  = "chain1"                        # identifier string for the line type
  
  
  # ----- set up the mooring system and floating body -----
  
  # Create new MoorPy System and set its depth
  ms = mp.System(depth=depth)
  
  # add a line type
  ms.setLineType(dnommm=120, material='chain', name=typeName)  # this would be 120 mm chain
  
  # Add a free, body at [0,0,0] to the system (including some properties to make it hydrostatically stiff)
  ms.addBody(0, np.zeros(6), m=1e6, v=1e3, rM=100, AWP=1e3)
  
  # For each line heading, set the anchor point, the fairlead point, and the line itself
  for i, angle in enumerate(angles):
  
      # create end Points for the line
      ms.addPoint(1, [rAnchor*np.cos(angle), rAnchor*np.sin(angle), -depth])   # create anchor point (type 0, fixed)
      ms.addPoint(1, [  rFair*np.cos(angle),   rFair*np.sin(angle),  zFair])   # create fairlead point (type 0, fixed)
      
      # attach the fairlead Point to the Body (so it's fixed to the Body rather than the ground)
      ms.bodyList[0].attachPoint(2*i+2, [rFair*np.cos(angle), rFair*np.sin(angle), zFair]) 
  
      # add a Line going between the anchor and fairlead Points
      ms.addLine(lineLength, typeName, pointA=2*i+1, pointB=2*i+2)




Creating a MoorPy System for a MoorDyn Input File
-------------------------------------------------

.. _inputfile:

A MoorPy System can be initialized by reading in a MoorDyn-style input file. This is simply done by
passing the input file name when creating the System object (from imported_system.py):

.. code-block:: python

  ms = mp.System(file='the MoorDyn-style input file.txt')


The format of the input file is expected to follow the
MoorDyn v2 style, an example of which is shown below:


.. code-block:: none
 
  MoorDyn v2 Input File 
  Sample for input to MoorPy
  ---------------------- LINE TYPES --------------------------------------------------
  TypeName      Diam     Mass/m     EA     BA/-zeta     EI      Cd     Ca   CdAx  CaAx
  (name)        (m)      (kg/m)     (N)    (N-s/-)    (N-m^2)   (-)    (-)  (-)   (-)
  chain         0.2160   286.56    1.23e9   -1.0        0.00    1.00  1.00  0.00  0.00  
  --------------------- ROD TYPES -----------------------------------------------------
  TypeName      Diam     Mass/m    Cd     Ca      CdEnd    CaEnd
  (name)        (m)      (kg/m)    (-)    (-)     (-)      (-)
  ----------------------- BODIES ------------------------------------------------------
  ID   Attachment    X0     Y0     Z0     r0     p0     y0     Mass     CG*     I*      Volume   CdA*   Ca*
  (#)     (-)        (m)    (m)    (m)   (deg)  (deg)  (deg)   (kg)     (m)    (kg-m^2)  (m^3)   (m^2)  (-)
  1     coupled     0.00   0.00   -0.75  0.00   0.00   0.00    1.0e6    0.00    0.00     1.0e3   0.00   0.00
  ---------------------- RODS ---------------------------------------------------------
  ID   RodType  Attachment  Xa    Ya    Za    Xb    Yb    Zb   NumSegs  RodOutputs
  (#)  (name)    (#/key)    (m)   (m)   (m)   (m)   (m)   (m)  (-)       (-)
  ---------------------- POINTS -------------------------------------------------------
  ID  Attachment     X        Y        Z       Mass   Volume  CdA    Ca
  (#)   (-)         (m)      (m)      (m)      (kg)   (mˆ3)  (m^2)   (-)
  1    Fixed       800.00  1385.64  -600.00    0.00   0.00   0.00   0.00
  2    Body1        10.00    17.32   -21.00    0.00   0.00   0.00   0.00
  3    Fixed     -1600.00     0.00  -600.00    0.00   0.00   0.00   0.00
  4    Body1       -20.00     0.00   -21.00    0.00   0.00   0.00   0.00
  5    Fixed       800.00 -1385.64  -600.00    0.00   0.00   0.00   0.00
  6    Body1        10.00   -17.32   -21.00    0.00   0.00   0.00   0.00
  ---------------------- LINES --------------------------------------------------------
  ID    LineType   AttachA  AttachB  UnstrLen  NumSegs  LineOutputs
  (#)   (name)      (#)      (#)       (m)       (-)     (-)
  1     chain        1        2     1800.000     40       p
  2     chain        3        4     1800.000     40       p
  3     chain        5        6     1800.000     40       p
  ---------------------- OPTIONS ------------------------------------------------------
  600.0            depth
  --------------------- need this line ------------------------------------------------


Note that some parameters are only applicable to a dynamic model like MoorDyn, and are ignored by MoorPy.
Conversely, some Body parameters used by MoorPy for hydrostatics are not captured in a MoorDyn-style file.



Running the MoorPy Model
------------------------

Once the MoorPy System is set up, it can be analyzed, viewed, and manipulated using a handful of main
functions, as well as a variety of additional helper functions for more specialized tasks.

Here is an example showing one of the possible functions to analyze a mooring system:


.. code-block:: python
 
  ms.initialize()                                             # make sure everything's connected
  
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot()                                         # plot the system in original configuration
  ms.unload("sample.txt")                                     # export to MD input file
  
  ms.bodyList[0].f6Ext = np.array([3e6, 0, 0, 0, 0, 0])       # apply an external force on the body 
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)



**Documentation Overview**

An overview of how a mooring system is represented in MoorPy can be found in :ref:`The Model Structure page<Model Structure>`.

More documentation and examples of other functions that can be applied to a MoorPy mooring system can be 
found in :ref:`The Usage page<MoorPy Usage>`.

Detailed theory "under the hood" of the functions in MoorPy can be found in :ref:`The Theory Page<Theory and References>`.

Detailed inputs and outputs of MoorPy classes and functions can be found in :ref:`The API Page<API>`.

