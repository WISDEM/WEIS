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

MoorPy has functions to facilitate the orderly creation of a mooring system. 

.. code-block:: none


  # ----- choose some system geometry parameters -----
  
  depth     = 600                             # water depth [m]
  angles    = np.radians([60, 180, 300])      # line headings list [rad]
  rAnchor   = 1600                            # anchor radius/spacing [m]
  zFair     = -21                             # fairlead z elevation [m]
  rFair     = 20                              # fairlead radius [m]
  lineLength= 1800                            # line unstretched length [m]
  typeName  = "chain"                         # identifier string for the line type
  
  
  # ----- set up the mooring system and floating body -----
  
  # Create new MoorPy System and set its depth
  ms = mp.System(depth=depth)
  
  # add a line type
  ms.lineTypes[typeName] = getLineProps(120, name=typeName)  # this would be 120 mm chain
  
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


.. code-block:: none      
  
  # ----- run the model to demonstrate -----
  
  ms.initialize()                                             # make sure everything's connected
  
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot()                                         # plot the system in original configuration
  ms.unload("sample.txt")                                     # export to MD input file
  
  ms.bodyList[0].f6Ext = np.array([3e6, 0, 0, 0, 0, 0])       # apply an external force on the body 
  ms.solveEquilibrium3()                                      # equilibrate
  fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)
  
  print(f"Body offset position is {ms.bodyList[0].r6}")
          
  plt.show()



Creating a MoorPy System for a MoorDyn Input File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _inputfile:

A MoorPy System can be initialized by reading in a MoorDyn-style input file. This is simply done by
passing the input file name when creating the System object:

.. code-block:: none

  ms = mp.System(file='the MoorDyn-style input file.txt')


The format of the input file is expected to follow the
MoorDyn v2 style, which is still being finalized. As a working example for reference, see below:


.. code-block:: none
 
  MoorDyn v2 Input File 
  Sample for input to MoorPy
  ---------------------- LINE TYPES -----------------------------------------------------
  LineType         Diam     MassDen   EA        cIntDamp     EI     Can    Cat    Cdn    Cdt
     (-)           (m)      (kg/m)    (N)        (Pa-s)    (N-m^2)  (-)    (-)    (-)    (-)
  chain            0.2160   286.56 1.230e+09 -1.000e+00 0.000e+00 1.000   0.000   1.000   0.000  
  ----------------------- BODIES -----------------------------------
  BodyID      X0    Y0    Z0     r0     p0     y0    Xcg   Ycg   Zcg     M      V        IX       IY       IZ     CdA  Ca
   (-)        (m)   (m)   (m)   (deg)  (deg)  (deg)  (m)   (m)   (m)    (kg)   (m^3)  (kg-m^2) (kg-m^2) (kg-m^2) (m^2) (-)
  1Coupled   0.00  0.00  -0.75  -0.00  0.00   0.00   0.00  0.00  0.00  1.0e6   1000.0    0        0        0      0    0
  ---------------------- POINTS ---------------------------------------------------------
  Node    Type         X        Y        Z        M      V      FX     FY     FZ    CdA    Ca 
  (-)     (-)         (m)      (m)      (m)      (kg)   (m^3)  (kN)   (kN)   (kN)   (m2)   ()
  1    Fixed          800.00  1385.64  -600.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  2    Body1           10.00    17.32   -21.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  3    Fixed        -1600.00     0.00  -600.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  4    Body1          -20.00     0.00   -21.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  5    Fixed          800.00 -1385.64  -600.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  6    Body1           10.00   -17.32   -21.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  ---------------------- LINES -----------------------------------------------------
  Line      LineType   UnstrLen  NumSegs  AttachA  AttachB  Outputs
  (-)         (-)       (m)        (-)     (-)      (-)     (-)
  1    chain           1800.000    40       1        2      p
  2    chain           1800.000    40       3        4      p
  3    chain           1800.000    40       5        6      p
  ---------------------- OPTIONS ----------------------------------------
  0.0002   dtM          - time step to use in mooring integration
  3        WaveKin      - wave kinematics flag (1=include(unsupported), 0=neglect, 3=currentprofile.txt)
  3.0e+06  kb           - bottom stiffness
  3.0e+05  cb           - bottom damping
  600.00   WtrDpth      - water depth
  2.0      ICDfac       - factor by which to scale drag coefficients during dynamic relaxation IC gen
  0.01     ICthresh     - threshold for IC convergence
  10       ICTmax       - threshold for IC convergence
  ----------------------------OUTPUTS--------------------------------------------
  FairTen1
  FairTen2
  FairTen3
  END
  --------------------- need this line ------------------


Note that some parameters are only applicable to a dynamic model like MoorDyn, and are ignored by MoorPy.
Conversely, some Body parameters used by MoorPy for hydrostatics are not captured in a MoorDyn-style file.



Running the MoorPy Model
------------------------

Once the MoorPy System is set up, it can be analyzed, viewed, and manipulated using a handful of main
functions, as well as a variety of additional helper functions for more specialized tasks.

Here is an example showing the most important functions:


.. code-block:: none
 
  ms.initialize()                                             # make sure everything's connected
  
  ms.solveEquilibrium()                                       # equilibrate
  fig, ax = ms.plot()                                         # plot the system in original configuration
  ms.unload("sample.txt")                                     # export to MD input file
  
  ms.bodyList[0].f6Ext = np.array([3e6, 0, 0, 0, 0, 0])       # apply an external force on the body 
  ms.solveEquilibrium3()                                      # equilibrate
  fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)
  

(A list of key functions to be added here)


Advice and Frequent Problems
----------------------------
   
   
Errors when running from an imported MoorDyn file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When initializin a MoorPy System from a MoorDyn-style input file, there are several common sources of
error:

- The section headers (e.g., "--------- Lines ----------") may not have the keywords MoorPy is expecting.
  Refer to the sample :ref:`above <inputfile>` for a format that works.
  
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

There are too many variables at play to provide decisive general guidance, but
we aim to expand the advice in this section as more user feedback is received.


