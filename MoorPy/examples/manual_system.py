# MoorPy Example Script:
# Example of manually setting up a mooring system in MoorPy and solving equilibrium.

import numpy as np
import matplotlib.pyplot as plt
import moorpy as mp
from moorpy.MoorProps import getLineProps


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



