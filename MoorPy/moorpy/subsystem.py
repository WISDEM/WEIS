
import numpy as np
import yaml

from moorpy.system import System
from moorpy.body import Body
from moorpy.point import Point
from moorpy.line import Line
from moorpy.lineType import LineType
from moorpy.helpers import (rotationMatrix, rotatePosition, getH, printVec, 
                            set_axes_equal, dsolve2, SolveError, MoorPyError, 
                            loadLineProps, getLineProps, read_mooring_file, 
                            printMat, printVec, getInterpNums, unitVector,
                            getFromDict, addToDict)



class Subsystem(System, Line):
    '''A class for a mooring line or dynamic cable subsystem.
    It includes Line sections but also can fit into a larger System
    the same way a Line can.
    
    A subsystem tracks its own objects in its local coordinate system.
    This local coordinate system puts the x axis along the line heading,
    from anchor to fairlead, with the anchor at x=0.
    
    For a multi-section line or cable (the main reason to use a SubSystem),
    the free DOFs of the points are contained in the SubSystem and are not 
    seen by the parent System. Solving their equilibrium is an internal
    solve nested within the larger System's equilibrium solve.
    
    Key adjustments: replace staticSolve from Line to do an internal equilibrium solve using System.solveEquilibrium
    '''
    
    
    def __init__(self, depth=0, rho=1025, g=9.81, qs=1, Fortran=True, lineProps=None, **kwargs):
        '''Shortened initializer for just the SubSystem aspects.'''
        
        # lists to hold mooring system objects
        self.bodyList = []
        self.rodList = []
        self.pointList = []
        self.lineList = []
        self.lineTypes = {}
        self.rodTypes = {}
        
        # load mooring line property scaling coefficients for easy use when creating line types
        self.lineProps = loadLineProps(lineProps)
        
        # the ground body (number 0, type 1[fixed]) never moves but is the parent of all anchored things
        self.groundBody = Body(self, 0, 1, np.zeros(6))
        
        # constants used in the analysis
        self.depth = depth  # water depth [m]
        self.rho   = rho    # water density [kg/m^3]
        self.g     = g      # gravitational acceleration [m/s^2]
        
        # water current - currentMod 0 = no current; 1 = steady uniform current
        self.currentMod = 0         # flag for current model to use
        self.current = np.zeros(3)  # current velocity vector [m/s]
        if 'current' in kwargs:
            self.currentMod = 1
            self.current = getFromDict(kwargs, 'current', shape=3)
            
        # seabed bathymetry - seabedMod 0 = flat; 1 = uniform slope, 2 = grid
        self.seabedMod = 0
        
        if 'xSlope' in kwargs or 'ySlope' in kwargs:
            self.seabedMod = 1
            self.xSlope = getFromDict(kwargs, 'xSlope', default=0)
            self.ySlope = getFromDict(kwargs, 'ySlope', default=0)
        
        if 'bathymetry' in kwargs:
            self.seabedMod = 2
            self.bathGrid_Xs, self.bathGrid_Ys, self.bathGrid = self.readBathymetryFile(kwargs['bathymetry'])
        
        
        # initializing variables and lists        
        self.nDOF = 0       # number of (free) degrees of freedom of the mooring system (needs to be set elsewhere)        
        self.freeDOFs = []  # array of the values of the free DOFs of the system at different instants (2D list)
        
        self.nCpldDOF = 0   # number of (coupled) degrees of freedom of the mooring system (needs to be set elsewhere)        
        self.cpldDOFs = []  # array of the values of the coupled DOFs of the system at different instants (2D list)
        
        self.display = 0    # a flag that controls how much printing occurs in methods within the System (Set manually. Values > 0 cause increasing output.)
        
        self.MDoptions = {} # dictionary that can hold any MoorDyn options read in from an input file, so they can be saved in a new MD file if need be
    
    
    
    def setEndPosition(self, r, endB):
        '''Sets the end position of the line based on the input endB value.

        Parameters
        ----------
        r : array
            x,y,z coorindate position vector of the line end [m].
        endB : boolean
            An indicator of whether the r array is at the end or beginning of the line

        Raises
        ------
        LineError
            If the given endB value is not a 1 or 0

        Returns
        -------
        None.

        '''
        
        if endB == 1:
            self.rB = np.array(r, dtype=np.float_)
        elif endB == 0:
            self.rA = np.array(r, dtype=np.float_)
        else:
            raise LineError("setEndPosition: endB value has to be either 1 or 0")
        
        
    def staticSolve(self, reset=False, tol=0.0001, profiles=0):
    
        # transform end positions to SubSystem internal coordinate system
        # inputs are self.rA, rB in global frame
        # outputs should be pointList[0] and [N] .r
        
        # get equilibrium
        self(System).solveEquilibrium(...)
        
        # transform coordinates and forces back into global frame
    
    
    def getDOFs(self):
        '''need any wrapper or custom function here?'''
    
    def drawLine2d(self, Time, ax, color="k", Xuvec=[1,0,0], Yuvec=[0,0,1], Xoff=0, Yoff=0, colortension=False, cmap='rainbow', plotnodes=[], plotnodesline=[], label="", alpha=1.0):
        '''wrapper to System.plot2d with some transformation applied'''


    def drawLine(self, Time, ax, color="k", endpoints=False, shadow=True, colortension=False, cmap_tension='rainbow'):
        '''wrapper to System.plot with some transformation applied'''
    
