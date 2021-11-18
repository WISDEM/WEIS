# MoorPy Example Script:
# Example of manually setting up a mooring system in MoorPy and solving equilibrium.

import numpy as np
import matplotlib.pyplot as plt
import moorpy as mp



# ----- set up the mooring system and floating body -----

# Create new MoorPy System and set its depth
ms = mp.System(file='sample.txt')

# ----- run the model to demonstrate -----

ms.initialize()                                             # make sure everything's connected

# find equilibrium of just the mooring lines (the body is 'coupled' and will by default not be moved)
ms.solveEquilibrium()                                       # equilibrate
fig, ax = ms.plot()                                         # plot the system in original configuration

# add hydrostatic stiffness properties to the body, then add a load and see how it responds
ms.bodyList[0].rM=np.array([0,0,100])                       # set a very high metacenter location to avoid pitch and roll [m]
ms.bodyList[0].AWP=1e3                                      # set a very large waterplane area to avoid heave [m^2]
ms.bodyList[0].f6Ext = np.array([3e6, 0, 0, 0, 0, 0])       # apply an external force on the body [N]
ms.solveEquilibrium3(DOFtype='both')                        # equilibrate (specify 'both' to enable both free and coupled DOFs)
fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)

print(f"Body offset position is {ms.bodyList[0].r6}")
        
plt.show()



