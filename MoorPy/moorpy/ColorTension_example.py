# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:41:24 2021

@author: elozon
"""

import numpy as np
import matplotlib.pyplot as plt
import moorpy as mp
from moorpy.MoorProps import getLineProps
import matplotlib as mpl

depth     = 600
angle     = np.arange(3)*np.pi*2/3  # line headings list
anchorR   = 1600                    # anchor radius/spacing
fair_depth= 21 
fairR     = 20
LineLength= 1800
typeName  = "chain"                 # identifier string for line type

# --------------- set up mooring system ---------------------

# Create blank system object
ms = mp.System()

# Set the depth of the system to the depth of the input value
ms.depth = depth

# add a line type
ms.lineTypes[typeName] = getLineProps(120, name=typeName)

# Add a free, body at [0,0,0] to the system (including some properties to make it hydrostatically stiff)
ms.addBody(0, np.zeros(6), m=1e6, v=1e3, rM=100, AWP=1e3)

# Set the anchor points of the system
anchors = []
for i in range(len(angle)):
    ms.addPoint(1, np.array([anchorR*np.cos(angle[i]), anchorR*np.sin(angle[i]), -ms.depth], dtype=float))
    anchors.append(len(ms.pointList))

# Set the points that are attached to the body to the system
bodypts = []
for i in range(len(angle)):
    ms.addPoint(1, np.array([fairR*np.cos(angle[i]), fairR*np.sin(angle[i]), -fair_depth], dtype=float))
    bodypts.append(len(ms.pointList))
    ms.bodyList[0].attachPoint(ms.pointList[bodypts[i]-1].number, ms.pointList[bodypts[i]-1].r - ms.bodyList[0].r6[:3])

# Add and attach lines to go from the anchor points to the body points
for i in range(len(angle)):    
    ms.addLine(LineLength, typeName)
    line = len(ms.lineList)
    ms.pointList[anchors[i]-1].attachLine(ms.lineList[line-1].number, 0)
    ms.pointList[bodypts[i]-1].attachLine(ms.lineList[line-1].number, 1)

# ------- simulate it ----------

ms.initialize()                                             # make sure everything's connected
ms.solveEquilibrium3()                                      # equilibrate
ms.unload("sample.txt")                                     # export to MD input file
fig, ax = ms.plot()                                         # plot the system in original configuration

ms.solveEquilibrium3()                                      # equilibrate
fig, ax = ms.plot(ax=ax, color='red')                       # plot the system in displaced configuration (on the same plot, in red)

print(f"Body offset position is {ms.bodyList[0].r6}")
        
plt.show()

fig, ax, minten, maxten = ms.colortensions()
fig.show()
#Read in tension data
#dataff, unitsff = ml.read_output_file("C:/code/Analyzing Outputs/Python scripts for OpenFAST outputs/FF Mooring lines/", "steadyfarm.FarmMD.MD.out")
#time = dataff['Time']
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
bounds = range(int(minten),int(maxten), int((maxten-minten)/10))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal',
             label="Tension [N]")
