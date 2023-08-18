MoorPy Usage
============

..
  customize code highlight color through "hll" span css

.. raw:: html

    <style> .highlight .hll {color:#000080; background-color: #eeeeff} </style>
    <style> .fast {color:#000080; background-color: #eeeeff} </style>
    <style> .stnd {color:#008000; background-color: #eeffee} </style>

.. role:: fast
.. role:: stnd


Setting up a Mooring System
---------------------------

In MoorPy, the full moored floating system is contained in a System object, which includes
lists of Body, Point, and Line objects that make up the full assembly. This collection of 
objects and their linkages can be set up manually via function calls, or they can be 
generated based on reading in a MoorDyn-style input file.


Creating a MoorPy System Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MoorPy has functions to facilitate the orderly creation of a mooring system. The following
gives an example of how they work.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _inputfile:

A MoorPy System can be initialized by reading in a MoorDyn-style input file. This is simply done by
passing the input file name when creating the System object:

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

Here is an example showing the most important functions:


.. code-block:: python
 
  ms.initialize()                                             # make sure everything's connected
  
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot()                                         # plot the system in original configuration
  ms.unload("sample.txt")                                     # export to MD input file
  
  ms.bodyList[0].f6Ext = np.array([3e6, 0, 0, 0, 0, 0])       # apply an external force on the body 
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)
  

(A list of key functions to be added here)


Additional Parameters in MoorPy
-------------------------------

Some of MoorPy's objects have additional parameters beyond those specified in the input file,
which can be used to add more features to the simulation.

- Body.Awp: This specifies a waterplane area for the body, which is used to provide a
  vertical hydrostatic stiffness.

- Point.zSpan: The zSpan parameter lists the lower and upper extents of the point's 
  volume, relative to the point coordinate, r. The Point's volume is 
  assumed evenly distributed between zSpan[0] and zSpan[1], and this 
  affects hydrostatic calculations when the Point crosses the free 
  surface (not used any other time).


Advice and Frequent Problems
----------------------------
   

General unexpected behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever the system is setup or modified (e.g., adding objects), the System.initialize
method must be called to register all connected objects in the system.

   
Errors when running from an imported MoorDyn file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When initializin a MoorPy System from a MoorDyn-style input file, there are several common sources of
error:

- The section headers (e.g., "--------- Lines ----------") may not have the keywords MoorPy is expecting.
  Refer to the sample :ref:`above <inputfile>` for the correct format. It has changed since 2021.
  
- The type keywords or number of expected entries in a line may be based on earlier MoorDyn version and 
  not match what MoorPy expects.
  
- The input file may not contain all the body information needed by MoorPy. Does the body type need to
  be specified as coupled? If the body is floating, its hydrostatic properties will need to be added
  in MoorPy manually because they are not contained in a MoorDyn input file.



Errors in finding system equilibrium
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solving system equilibrium can be the most difficult part of a MoorDyn analysis.
If the system equilibrium solve is unsuccessful, some of the possible causes are

- The system equilibrium sovle includes a floating body, and that body does not 
  have adequate hydrostatic properties.
  
- The mooring system results in a numerically-challenging stiffness matrix - for 
  example if some lines are much shorter than others, or if there are taut lines
  with very high stiffnesses.


Confusion with calculating system stiffness matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stiffness matrices can be calculated for multiple mooring objects using multiple different methods in MoorPy.
The two main methods of calculating stiffnesses are through a finite difference method and an analytical method.
The finite difference method slightly perturbs each DOF of the MoorPy object and calculates the change in force 
on the object before and after the perturbation. This change in force divided by the change in displacement provides
the stiffness value. The analytical stiffness method derives the stiffness of each mooring line using the Catenary 
equations and translates that stiffness to whichever point (or body) the mooring line is attached to.

- Finite difference
   - The 3x3 stiffness matrix of a Point object at its given location can be found by running point.getStiffness()
   - The 6x6 stiffness matrix of a Body object at its given location can be found by running body.getStiffness()
   - The "nDOFtype" x "nDOFtype" stiffness matrix of a System can be found by running system.getSystemStiffness()
   - The nCpldDOF x nCpldDOF stiffness matrix of a System can be found by running system.getCoupledStiffness()
- Analytical
   - The 2x2 or 3x3 analytical stiffness matrix of a Line object is calculated internally when solving for the Line's end forces in line.staticSolve()
   - The 3x3 analytical stiffness matrix of a Point object at its given location can be found by running point.getStiffnessA()
   - The 6x6 analytical stiffness matrix of a Body object at its given location can be found by running body.getStiffnessA()
   - The "nDOFtype" x "nDOFtype" analytical stiffness matrix of a System can be found by running system.getSystemStiffnessA()

The overall mooring system stiffness matrix is usually of interest to most users. This can be found by running one of the three 
System stiffness matrix methods. The one best to use depends on the types of other objects in the MoorPy System.

- The getSystemStiffness() method calculates the combined stiffness matrix of all "DOFtype" objects in the mooring system.
  It has a default value of "free" to the "DOFtype" input, meaning that it will calculate the combined stiffness matrix of 
  all "free" objects (e.g., points, bodies) in the system.
   - For example, a three-line mooring system with two line types in each mooring line, where each connecting point between 
     the two line types is a "free" floating Point object, and a free floating Body object on the surface, will result in a 
     15x15 stiffness matrix (a 6x6 matrix for the body and 3 3x3 matrices for the connecting points)
- The getCoupledStiffness() method calculates the system stiffness matrix for all "coupled" DOFs specifically, while 
  equilibrating the free, uncoupled DOFs. This would be similar to running getSystemStiffness(DOFtype="coupled"), except this method 
  solves for equilibrium in the free floating points while calculating the stiffness of the coupled DOFs, rather than only solving 
  for the stiffness matrix of all "coupled" DOFs.
   - For example, using the same setup described above, except that the body is a "coupled" DOFtype, the result of running 
     getCoupledStiffness() will be a 6x6 matrix, since the body is the only "coupled" object in the mooring system.
- The getSystemStiffnessA() method calculates the combined analytical stiffness matrix of all "DOFtype" objects in the mooring system. 
  It calls the analytical stiffness calculation methods of other objects in the mooring system and combines their analytical stiffness 
  matrices into one global system stiffness matrix. It has a default value of "free" to the "DOFtype" input.



Other errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are too many variables at play to provide decisive general guidance, but
we aim to expand the advice in this section as more user feedback is received.


