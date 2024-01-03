import pdb

import numpy as np
from matplotlib import cm
from moorpy.Catenary import catenary
from moorpy.nonlinear import nonlinear                                      
from moorpy.helpers import (unitVector, LineError, CatenaryError, 
                     rotationMatrix, makeTower, read_mooring_file, 
                     quiver_data_to_segments, printVec, printMat)
from os import path

 
class Line():
    '''A class for any mooring line that consists of a single material'''

    def __init__(self, mooringSys, num, L, lineType, nSegs=100, cb=0, isRod=0, attachments = [0,0]):
        '''Initialize Line attributes

        Parameters
        ----------
        mooringSys : system object
            The system object that contains the point object
        num : int
            indentifier number
        L : float
            line unstretched length [m]
        lineType : dict
            dictionary containing the coefficients needed to describe the line (could reference an entry of System.lineTypes).
        nSegs : int, optional
            number of segments to split the line into. Used in MoorPy just for plotting. The default is 100.
        cb : float, optional
            line seabed friction coefficient (will be set negative if line is fully suspended). The default is 0.
        isRod : boolean, optional
            determines whether the line is a rod or not. The default is 0.
        attachments : TYPE, optional
            ID numbers of any Points attached to the Line. The default is [0,0]. << consider removing

        Returns
        -------
        None.

        '''
        
        self.sys    = mooringSys       # store a reference to the overall mooring system (instance of System class)
        
        self.number = num
        self.isRod = isRod
            
        self.L = L              # line unstretched length (may be modified if using nonlinear elasticity) [m]
        self.L0 = L             # line reference unstretched length [m]
        self.type = lineType    # dictionary of a System.lineTypes entry
        
        self.EA = self.type['EA']  # use the default stiffness value for now (may be modified if using nonlinear elasticity) [N]
        
        self.nNodes = int(nSegs) + 1
        self.cb = float(cb)    # friction coefficient (will automatically be set negative if line is fully suspended)
        self.sbnorm = []    # Seabed Normal Vector (to be filled with a 3x1 normal vector describing seabed orientation)
        
        self.rA = np.zeros(3) # end coordinates
        self.rB = np.zeros(3)
        self.fA = np.zeros(3) # end forces
        self.fB = np.zeros(3)
        
        #Perhaps this could be made less intrusive by defining it using a line.addpoint() method instead, similar to point.attachline().
        #self.attached = attachments  # ID numbers of the Points at the Line ends [a,b] >>> NOTE: not fully supported <<<<
        self.th = 0           # heading of line from end A to B
        self.HF = 0           # fairlead horizontal force saved for next solve
        self.VF = 0           # fairlead vertical force saved for next solve
        self.KA = []          # to be filled with the 2x2 end stiffness matrix from catenary
        self.KB = []          # to be filled with the 2x2 end stiffness matrix from catenary
        self.info = {}        # to hold all info provided by catenary
        
        self.qs = 1  # flag indicating quasi-static analysis (1). Set to 0 for time series data
        self.show = True      # a flag that will be set to false if we don't want to show the line (e.g. if results missing)
        #print("Created Line "+str(self.number))
        self.color = 'k'
        self.lw=0.5
        
        self.fCurrent = np.zeros(3)  # total current force vector on the line [N]
        
        

    
    def loadData(self, dirname, rootname, sep='.MD.'):
        '''Loads line-specific time series data from a MoorDyn output file'''
        
        self.qs = 0 # signals time series data
        
        if self.isRod==1:
            strtype='Rod'
        elif self.isRod==0:
            strtype='Line'

        filename = dirname+rootname+sep+strtype+str(self.number)+'.out'
        
        if path.exists(filename):


        # try:
        
            # load time series data
            data, ch, channels, units = read_mooring_file("", filename) # remember number starts on 1 rather than 0

            # get time info
            if ("Time" in ch):
                self.Tdata = data[:,ch["Time"]]
                self.dt = self.Tdata[1]-self.Tdata[0]
            else:
                raise LineError("loadData: could not find Time channel for mooring line "+str(self.number))
        
            
            nT = len(self.Tdata)  # number of time steps
            
            # check for position data <<<<<<
            
            self.xp = np.zeros([nT,self.nNodes])
            self.yp = np.zeros([nT,self.nNodes])
            self.zp = np.zeros([nT,self.nNodes])
            
            
            for i in range(self.nNodes):
                self.xp[:,i] = data[:, ch['Node'+str(i)+'px']]
                self.yp[:,i] = data[:, ch['Node'+str(i)+'py']]
                self.zp[:,i] = data[:, ch['Node'+str(i)+'pz']]
            
            '''
            if self.isRod==0:
                self.Te = np.zeros([nT,self.nNodes-1])   # read in tension data if available
                if "Seg1Te" in ch:
                    for i in range(self.nNodes-1):
                        self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Te']]
                        
                self.Ku = np.zeros([nT,self.nNodes])   # read in curvature data if available
                if "Node0Ku" in ch:
                    for i in range(self.nNodes):
                        self.Ku[:,i] = data[:, ch['Node'+str(i)+'Ku']]
            else:
                # read in Rod buoyancy force data if available
                if "Node0Box" in ch:
                    self.Bx = np.zeros([nT,self.nNodes])   
                    self.By = np.zeros([nT,self.nNodes])
                    self.Bz = np.zeros([nT,self.nNodes])
                    for i in range(self.nNodes):
                        self.Bx[:,i] = data[:, ch['Node'+str(i)+'Box']]
                        self.By[:,i] = data[:, ch['Node'+str(i)+'Boy']]
                        self.Bz[:,i] = data[:, ch['Node'+str(i)+'Boz']]

            if "Node0Ux" in ch:
                self.Ux = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Uy = np.zeros([nT,self.nNodes])
                self.Uz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Ux[:,i] = data[:, ch['Node'+str(i)+'Ux']]
                    self.Uy[:,i] = data[:, ch['Node'+str(i)+'Uy']]
                    self.Uz[:,i] = data[:, ch['Node'+str(i)+'Uz']]
            
            #Read in tension data if available
            if "Seg1Ten" in ch:
                self.Ten = np.zeros([nT,self.nNodes-1])   
                for i in range(self.nNodes-1):
                    self.Ten[:,i] = data[:, ch['Seg'+str(i+1)+'Ten']]
            '''
            
            
            
            # --- Read in additional data if available ---

            # segment tension  <<< to be changed to nodal tensions in future MD versions
            if "Seg1Ten" in ch:
                self.Tendata = True
                self.Te = np.zeros([nT,self.nNodes-1])
                for i in range(self.nNodes-1):
                    self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Ten']]
            elif "Seg1Te" in ch:
                self.Tendata = True
                self.Te = np.zeros([nT,self.nNodes-1])
                for i in range(self.nNodes-1):
                    self.Te[:,i] = data[:, ch['Seg'+str(i+1)+'Te']]
            else:
                self.Tendata = False
                        
            # curvature at node
            if "Node0Ku" in ch:
                self.Kudata = True
                self.Ku = np.zeros([nT,self.nNodes])   
                for i in range(self.nNodes):
                    self.Ku[:,i] = data[:, ch['Node'+str(i)+'Ku']]
            else:
                self.Kudata = False
            
            # water velocity data 
            if "Node0Ux" in ch:  
                self.Udata = True
                self.Ux = np.zeros([nT,self.nNodes])
                self.Uy = np.zeros([nT,self.nNodes])
                self.Uz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Ux[:,i] = data[:, ch['Node'+str(i)+'Ux']]
                    self.Uy[:,i] = data[:, ch['Node'+str(i)+'Uy']]
                    self.Uz[:,i] = data[:, ch['Node'+str(i)+'Uz']]
            else:
                self.Udata = False
                
            # buoyancy force data
            if "Node0Box" in ch:  
                self.Bdata = True
                self.Bx = np.zeros([nT,self.nNodes])
                self.By = np.zeros([nT,self.nNodes])
                self.Bz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Bx[:,i] = data[:, ch['Node'+str(i)+'Box']]
                    self.By[:,i] = data[:, ch['Node'+str(i)+'Boy']]
                    self.Bz[:,i] = data[:, ch['Node'+str(i)+'Boz']]
            else:
                self.Bdata = False
                
            # hydro drag data
            if "Node0Dx" in ch: 
                self.Ddata = True
                self.Dx = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Dy = np.zeros([nT,self.nNodes])
                self.Dz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Dx[:,i] = data[:, ch['Node'+str(i)+'Dx']]
                    self.Dy[:,i] = data[:, ch['Node'+str(i)+'Dy']]
                    self.Dz[:,i] = data[:, ch['Node'+str(i)+'Dz']]
            else:
                self.Ddata = False
                
            # weight data
            if "Node0Wx" in ch: 
                self.Wdata = True
                self.Wx = np.zeros([nT,self.nNodes])   # read in fluid velocity data if available
                self.Wy = np.zeros([nT,self.nNodes])
                self.Wz = np.zeros([nT,self.nNodes])
                for i in range(self.nNodes):
                    self.Wx[:,i] = data[:, ch['Node'+str(i)+'Wx']]
                    self.Wy[:,i] = data[:, ch['Node'+str(i)+'Wy']]
                    self.Wz[:,i] = data[:, ch['Node'+str(i)+'Wz']]
            else:
                self.Wdata = False
            
            
            
            # initialize positions (is this used?)
            self.xpi= self.xp[0,:]
            self.ypi= self.yp[0,:]
            self.zpi= self.zp[0,:]
            
            # calculate the dynamic LBot !!!!!!! doesn't work for sloped bathymetry yet !!!!!!!!!!
            for i in range(len(self.zp[0])):
                if np.max(self.zp[:,i]) > self.zp[0,0]:
                    inode = i
                    break
                else:
                    inode = i
            self.LBotDyn = (inode-1)*self.L/(self.nNodes-1)
            
            # get length (constant)
            #self.L = np.sqrt( (self.xpi[-1]-self.xpi[0])**2 + (self.ypi[-1]-self.ypi[0])**2 + (self.zpi[-1]-self.zpi[0])**2 )
            # ^^^^^^^ why are we changing the self.L value to not the unstretched length specified in MoorDyn?
            # moved this below the dynamic LBot calculation because I wanted to use the original self.L
            # >>> this is probably needed for Rods - should look into using for Rods only <<<
            
            # check for tension data <<<<<<<
            
            self.show = True
            
        else:
            self.Tdata = []
            self.show = False
            print(f"Error geting data for {'Rod' if self.isRod else 'Line'} {self.number}: {filename}")
            print("dirname: {} or rootname: {} is incorrect".format(dirname, rootname))
            
         
        # >>> this was another option for handling issues - maybe no longer needed <<<
        #except Exception as e:
        #    # don't fail if there's an issue finding data, just flag that the line shouldn't be shown/plotted
        #    print(f"Error geting data for {'Rod' if self.isRod else 'Line'} {self.number}: ")
        #    print(e)
        #    self.show = False
        
    
    def setL(self, L):
        '''Sets the line unstretched length [m], and saves it for use with
        static-dynamic stiffness adjustments. Also reverts to static 
        stiffness to avoid an undefined state of having changing the line
        length in a state with adjusted dynamic EA and L values.'''
        self.L = L
        self.L0 = L
        self.revertToStaticStiffness()


    def getTimestep(self, Time):
        '''Get the time step to use for showing time series data'''
        
        if Time < 0: 
            ts = np.int_(-Time)  # negative value indicates passing a time step index
        else:           # otherwise it's a time in s, so find closest time step
            if len(self.Tdata) > 0:
                for index, item in enumerate(self.Tdata):                
                    ts = -1
                    if item > Time:
                        ts = index
                        break
                if ts==-1:
                    raise LineError(self.number, "getTimestep: requested time likely out of range")
            else:
                raise LineError(self.number, "getTimestep: zero time steps are stored")

        return ts
        
        

    def getLineCoords(self, Time, n=0, segmentTensions=False):
        '''Gets the updated line coordinates for drawing and plotting purposes.'''
        
        if n==0: n = self.nNodes
    
        # special temporary case to draw a rod for visualization. This assumes the rod end points have already been set somehow
        if self.qs==1 and self.isRod > 0:
        
            # make points for appropriately sized cylinder
            d = self.type['d_vol']
            Xs, Ys, Zs = makeTower(self.L, np.array([d/2, d/2]))   # add in makeTower method once you start using Rods
            
            # get unit vector and orientation matrix
            k = (self.rB-self.rA)/self.L
            Rmat = np.array(rotationMatrix(0, np.arctan2(np.hypot(k[0],k[1]), k[2]), np.arctan2(k[1],k[0])))
        
            # translate and rotate into proper position for Rod
            coords = np.vstack([Xs, Ys, Zs])
            newcoords = np.matmul(Rmat,coords)
            Xs = newcoords[0,:] + self.rA[0]
            Ys = newcoords[1,:] + self.rA[1]
            Zs = newcoords[2,:] + self.rA[2]
            
            return Xs, Ys, Zs, None
        
    
        # if a quasi-static analysis, just call the catenary function to return the line coordinates
        elif self.qs==1:
            
            self.staticSolve(profiles=1) # call with flag to tell Catenary to return node info
            
            #Xs = self.rA[0] + self.info["X"]*self.cosBeta 
            #Ys = self.rA[1] + self.info["X"]*self.sinBeta 
            #Zs = self.rA[2] + self.info["Z"]
            #Ts = self.info["Te"]
            Xs = self.Xs
            Ys = self.Ys
            Zs = self.Zs
            Ts = self.Ts
            return Xs, Ys, Zs, Ts
            
        # otherwise, count on read-in time-series data
        else:

            # figure out what time step to use
            ts = self.getTimestep(Time)
            
            # drawing rods
            if self.isRod > 0:
            
                k1 = np.array([ self.xp[ts,-1]-self.xp[ts,0], self.yp[ts,-1]-self.yp[ts,0], self.zp[ts,-1]-self.zp[ts,0] ]) / self.L # unit vector
                
                k = np.array(k1) # make copy
            
                Rmat = np.array(rotationMatrix(0, np.arctan2(np.hypot(k[0],k[1]), k[2]), np.arctan2(k[1],k[0])))  # <<< should fix this up at some point, MattLib func may be wrong
                
                # make points for appropriately sized cylinder
                d = self.type['d_vol']
                Xs, Ys, Zs = makeTower(self.L, np.array([d/2, d/2]))   # add in makeTower method once you start using Rods
                
                # translate and rotate into proper position for Rod
                coords = np.vstack([Xs, Ys, Zs])
                newcoords = np.matmul(Rmat,coords)
                Xs = newcoords[0,:] + self.xp[ts,0]
                Ys = newcoords[1,:] + self.yp[ts,0]
                Zs = newcoords[2,:] + self.zp[ts,0]
                
                return Xs, Ys, Zs, None
                
            # drawing lines
            else:
                
                # handle whether or not there is tension data
                try:  # use average to go from segment tension to node tensions <<< can skip this once MD is updated to output node tensions
                    if segmentTensions:
                        Te = self.Te[ts,:]  # return tensions of segments rather than averaging to get tensions of nodes
                    else:
                        Te = 0.5*(np.append(self.Te[ts,0], self.Te[ts,:]) +np.append(self.Te[ts,:], self.Te[ts,-1]))
                except: # otherwise return zeros to avoid an error (might want a warning in some cases?)
                    Te = np.zeros(self.nNodes)
                
                return self.xp[ts,:], self.yp[ts,:], self.zp[ts,:], Te
    
    
    def getCoordinate(self, s, n=100):
        '''Returns position and tension at a specific point along the line's unstretched length'''
        
        dr =  self.rB - self.rA                 
        LH = np.hypot(dr[0], dr[1])  
            
        Ss = np.linspace(0, self.L, n)
        Xs, Ys, Zs, Ts = self.getLineCoords(0.0, n=n)
        
        X = np.interp(s, Ss, Xs)*dr[0]/LH
        Y = np.interp(s, Ss, Ys)*dr[1]/LH
        Z = np.interp(s, Ss, Zs)
        T = np.interp(s, Ss, Ts)
        
        return X, Y, Z, T
        
    
    
    def drawLine2d(self, Time, ax, color="k", Xuvec=[1,0,0], Yuvec=[0,0,1], Xoff=0, Yoff=0, colortension=False, cmap='rainbow', plotnodes=[], plotnodesline=[], label="", alpha=1.0):
        '''Draw the line on 2D plot (ax must be 2D)

        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        Xuvec : list, optional
            plane at which the x-axis is desired. The default is [1,0,0].
        Yuvec : list, optional
            plane at which the y-axis is desired. The default is [0,0,1].
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'

        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted

        '''
        
        linebit = []  # make empty list to hold plotted lines, however many there are
        
        if self.isRod > 0:
            
            Xs, Ys, Zs, Te = self.getLineCoords(Time)
        
            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs*Xuvec[0] + Ys*Xuvec[1] + Zs*Xuvec[2] 
            Ys2d = Xs*Yuvec[0] + Ys*Yuvec[1] + Zs*Yuvec[2] 
        
            for i in range(int(len(Xs)/2-1)):
                linebit.append(ax.plot(Xs2d[2*i:2*i+2]    ,Ys2d[2*i:2*i+2]    , lw=0.5, color=color))  # side edges
                linebit.append(ax.plot(Xs2d[[2*i,2*i+2]]  ,Ys2d[[2*i,2*i+2]]  , lw=0.5, color=color))  # end A edges
                linebit.append(ax.plot(Xs2d[[2*i+1,2*i+3]],Ys2d[[2*i+1,2*i+3]], lw=0.5, color=color))  # end B edges
        
        # drawing lines...
        else:            
            # >>> can probably streamline the next bit of code a fair bit <<<
            if self.qs==1:
                Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            elif self.qs==0:
                Xs, Ys, Zs, Ts = self.getLineCoords(Time)
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
            
            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs*Xuvec[0] + Ys*Xuvec[1] + Zs*Xuvec[2] + Xoff
            Ys2d = Xs*Yuvec[0] + Ys*Yuvec[1] + Zs*Yuvec[2] + Yoff
            
            if colortension:    # if the mooring lines want to be plotted with colors based on node tensions
                maxT = np.max(Ts); minT = np.min(Ts)
                for i in range(len(Xs)-1):          # for each node in the line
                    color_ratio = ((Ts[i] + Ts[i+1])/2 - minT)/(maxT - minT)  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap)    # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)    # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs2d[i:i+2], Ys2d[i:i+2], color=rgba))
            else:
                linebit.append(ax.plot(Xs2d, Ys2d, lw=1, color=color, label=label, alpha=alpha)) # previously had lw=1 (linewidth)
            
            if len(plotnodes) > 0:
                for i,node in enumerate(plotnodes):
                    if self.number==plotnodesline[i]:
                        linebit.append(ax.plot(Xs2d[node], Ys2d[node], 'o', color=color, markersize=5))   
            
        self.linebit = linebit # can we store this internally?
        
        self.X = np.array([Xs, Ys, Zs])
            
        return linebit

    

    def drawLine(self, Time, ax, color="k", endpoints=False, shadow=True, colortension=False, cmap_tension='rainbow'):
        '''Draw the line in 3D
        
        Parameters
        ----------
        Time : float
            time value at which to draw the line
        ax : axis
            the axis on which the line is to be drawn
        color : string, optional
            color identifier in one letter (k=black, b=blue,...). The default is "k".
        endpoints : bool, optional
            toggle to plot the end points of the lines. The default is False
        shadow : bool, optional
            toggle to plot the mooring line shadow on the seabed. The default is True
        colortension : bool, optional
            toggle to plot the lines in a colormap based on node tensions. The default is False
        cmap : string, optional
            colormap string type to plot tensions when colortension=True. The default is 'rainbow'
            
        Returns
        -------
        linebit : list
            list of axes and points on which the line can be plotted
        '''
        
        if not self.show:  # exit if this line isn't set to be shown
            return 0
        
        if color == 'self':
            color = self.color  # attempt to allow custom colors
            lw = self.lw
        else:
            lw = 1
        
        linebit = []  # make empty list to hold plotted lines, however many there are
    
        if self.isRod > 0:
            
            if color==None:
                color = [0.3, 0.3, 0.3]  # if no color provided, default to dark grey rather than rainbow rods
                
            Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            
            for i in range(int(len(Xs)/2-1)):
                linebit.append(ax.plot(Xs[2*i:2*i+2],Ys[2*i:2*i+2],Zs[2*i:2*i+2]            , color=color))  # side edges
                linebit.append(ax.plot(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]],Zs[[2*i,2*i+2]]      , color=color))  # end A edges
                linebit.append(ax.plot(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]],Zs[[2*i+1,2*i+3]], color=color))  # end B edges
            
            # scatter points for line ends 
            #if endpoints == True:
            #    linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color = color))
        
        # drawing lines...
        else:
            # >>> can probably streamline the next bit of code a fair bit <<<
            if self.qs==1:  # returns the node positions and tensions of the line, doesn't matter what time
                Xs, Ys, Zs, tensions = self.getLineCoords(Time)
            elif self.qs==0: # returns the node positions and time data at the given time
                Xs, Ys, Zs, tensions = self.getLineCoords(Time)
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
            
            if colortension:    # if the mooring lines want to be plotted with colors based on node tensions
                maxT = np.max(tensions); minT = np.min(tensions)
                for i in range(len(Xs)-1):          # for each node in the line
                    color_ratio = ((tensions[i] + tensions[i+1])/2 - minT)/(maxT - minT)  # ratio of the node tension in relation to the max and min tension
                    cmap_obj = cm.get_cmap(cmap_tension)    # create a cmap object based on the desired colormap
                    rgba = cmap_obj(color_ratio)    # return the rbga values of the colormap of where the node tension is
                    linebit.append(ax.plot(Xs[i:i+2], Ys[i:i+2], Zs[i:i+2], color=rgba, zorder=100))
            else:
                linebit.append(ax.plot(Xs, Ys, Zs, color=color, lw=lw, zorder=100))
            
            if shadow:
                ax.plot(Xs, Ys, np.zeros_like(Xs)-self.sys.depth, color=[0.5, 0.5, 0.5, 0.2], lw=lw, zorder = 1.5) # draw shadow
            
            if endpoints == True:
                linebit.append(ax.scatter([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], [Zs[0], Zs[-1]], color = color))
                
                    
            # draw additional data if available (should make this for rods too eventually - drawn along their axis nodes)
            if self.qs == 0:              
                ts = self.getTimestep(Time)
                
                if self.Tendata:
                    pass
                if self.Kudata:
                    pass        
                if self.Udata:
                    self.Ubits = ax.quiver(Xs, Ys, Zs, self.Ux[ts,:], self.Uy[ts,:], self.Uz[ts,:], color="blue")  # make quiver plot and save handle to line object
                if self.Bdata:
                    self.Bbits = ax.quiver(Xs, Ys, Zs, self.Bx[ts,:], self.By[ts,:], self.Bz[ts,:], color="red")
                if self.Ddata:
                    self.Dbits = ax.quiver(Xs, Ys, Zs, self.Dx[ts,:], self.Dy[ts,:], self.Dz[ts,:], color="green")
                if self.Wdata:
                    self.Wbits = ax.quiver(Xs, Ys, Zs, self.Wx[ts,:], self.Wy[ts,:], self.Wz[ts,:], color="orange")
                
                
        self.linebit = linebit # can we store this internally?
        
        self.X = np.array([Xs, Ys, Zs])
        
            
        return linebit
    
    

        
        
    def redrawLine(self, Time, colortension=False, cmap_tension='rainbow', drawU=True):  #, linebit):
        '''Update 3D line drawing based on instantaneous position'''
        
        linebit = self.linebit
        
        if self.isRod > 0:
            
            Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            
            for i in range(int(len(Xs)/2-1)):
                        
                linebit[3*i  ][0].set_data(Xs[2*i:2*i+2],Ys[2*i:2*i+2])    # side edges (x and y coordinates)
                linebit[3*i  ][0].set_3d_properties(Zs[2*i:2*i+2])         #            (z coordinates)             
                linebit[3*i+1][0].set_data(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]])           # end A edges
                linebit[3*i+1][0].set_3d_properties(Zs[[2*i,2*i+2]])                    
                linebit[3*i+2][0].set_data(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]])   # end B edges
                linebit[3*i+2][0].set_3d_properties(Zs[[2*i+1,2*i+3]])
        
        # drawing lines...
        else:
        
            Xs, Ys, Zs, Ts = self.getLineCoords(Time)
            
            if colortension:
                self.rA = np.array([Xs[0], Ys[0], Zs[0]])       # update the line ends based on the MoorDyn data
                self.rB = np.array([Xs[-1], Ys[-1], Zs[-1]])
                maxT = np.max(Ts); minT = np.min(Ts)
                cmap_obj = cm.get_cmap(cmap_tension)               # create the colormap object
                
                for i in range(len(Xs)-1):  # for each node in the line, find the relative tension of the segment based on the max and min tensions
                    color_ratio = ((Ts[i] + Ts[i+1])/2 - minT)/(maxT - minT)
                    rgba = cmap_obj(color_ratio)
                    linebit[i][0]._color = rgba         # set the color of the segment to a new color based on its updated tension
                    linebit[i][0].set_data(Xs[i:i+2],Ys[i:i+2])     # set the x and y coordinates
                    linebit[i][0].set_3d_properties(Zs[i:i+2])      # set the z coorindates
            
            else:
                linebit[0][0].set_data(Xs,Ys)    # (x and y coordinates)
                linebit[0][0].set_3d_properties(Zs)         # (z coordinates) 
                    
            
        
            # draw additional data if available (should make this for rods too eventually - drawn along their axis nodes)
            if self.qs == 0:
                ts = self.getTimestep(Time)                    
                s = 0.0002
                
                if self.Tendata:
                    pass
                if self.Kudata:
                    pass        
                if self.Udata:
                    self.Ubits.set_segments(quiver_data_to_segments(Xs, Ys, Zs, self.Ux[ts,:], self.Uy[ts,:], self.Uz[ts,:], scale=10.))
                if self.Bdata:
                    self.Bbits.set_segments(quiver_data_to_segments(Xs, Ys, Zs, self.Bx[ts,:], self.By[ts,:], self.Bz[ts,:], scale=s))
                if self.Ddata:
                    self.Dbits.set_segments(quiver_data_to_segments(Xs, Ys, Zs, self.Dx[ts,:], self.Dy[ts,:], self.Dz[ts,:], scale=s))
                if self.Wdata:
                    self.Wbits.set_segments(quiver_data_to_segments(Xs, Ys, Zs, self.Wx[ts,:], self.Wy[ts,:], self.Wz[ts,:], scale=s))
                
                
        
        return linebit
        
        
    
    
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
        '''Solves static equilibrium of line. Sets the end forces of the line based on the end points' positions.

        Parameters
        ----------
        reset : boolean, optional
            Determines if the previous fairlead force values will be used for the catenary iteration. The default is False.

        tol : float
            Convergence tolerance for catenary solver measured as absolute error of x and z values in m.
            
        profiles : int
            Values greater than 0 signal for line profile data to be saved (used for plotting, getting distributed tensions, etc).

        Raises
        ------
        LineError
            If the horizontal force at the fairlead (HF) is less than 0

        Returns
        -------
        None.

        '''

        # deal with horizontal tension starting point
        if self.HF < 0:
            raise LineError("Line HF cannot be negative") # this could be a ValueError too...
            
        if reset==True:   # Indicates not to use previous fairlead force values to start catenary 
            self.HF = 0   # iteration with, and insteady use the default values.
        
        
        # ensure line profile information is computed if needed for computing current loads
        if self.sys.currentMod == 1 and profiles == 0:
            profiles = 1

        # get seabed depth and slope under each line end
        depthA, nvecA = self.sys.getDepthFromBathymetry(self.rA[0], self.rA[1])
        depthB, nvecB = self.sys.getDepthFromBathymetry(self.rB[0], self.rB[1])
        
        # deal with height off seabed issues
        if self.rA[2] < -depthA:
            self.rA[2] = -depthA
            self.cb = 0
            #raise LineError("Line {} end A is lower than the seabed.".format(self.number)) <<< temporarily adjust to seabed depth
        elif self.rB[2] < -depthB:
            raise LineError("Line {} end B is lower than the seabed.".format(self.number))
        else:
            self.cb = -depthA - self.rA[2]  # when cb < 0, -cb is defined as height of end A off seabed (in catenary)

        
        # ----- Perform rotation/transformation to 2D plane of catenary -----
        
        dr =  self.rB - self.rA
        
        # if a current force is present, include it in the catenary solution
        if np.sum(np.abs(self.fCurrent)) > 0:
        
            # total line exernal force per unit length vector (weight plus current drag)
            w_vec = self.fCurrent/self.L + np.array([0, 0, -self.type["w"]])
            w = np.linalg.norm(w_vec)
            w_hat = w_vec/w
            
            # get rotation matrix from gravity down to w_vec being down
            if w_hat[0] == 0 and w_hat[1] == 0: 
                if w_hat[2] < 0:
                    R_curr = np.eye(3,3)
                else:
                    R_curr = -np.eye(3,3)
            else:
                R_curr = RotFrm2Vect(w_hat, np.array([0, 0, -1]))  # rotation matrix to make w vertical
        
            # vector from A to B needs to be put into the rotated frame
            dr = np.matmul(R_curr, dr)  
        
        # if no current force, things are simple
        else:
            R_curr = np.eye(3,3)
            w = self.type["w"]
        
        
        # apply a rotation about Z' to align the line profile with the X'-Z' plane
        theta_z = -np.arctan2(dr[1], dr[0])
        R_z = rotationMatrix(0, 0, theta_z)
        
        # overall rotation matrix (global to catenary plane)
        R = np.matmul(R_z, R_curr)   
        
        # figure out slope in plane (only if contacting the seabed)
        if self.rA[2] <= -depthA or self.rB[2] <= -depthB:
            nvecA_prime = np.matmul(R, nvecA)
        
            dz_dx = -nvecA_prime[0]*(1.0/nvecA_prime[2])  # seabed slope components
            dz_dy = -nvecA_prime[1]*(1.0/nvecA_prime[2])  # seabed slope components
            # we only care about dz_dx since the line is in the X-Z plane in this rotated situation
            alpha = np.degrees(np.arctan(dz_dx))
            cb = self.cb
        else:
            if np.sum(np.abs(self.fCurrent)) > 0 or nvecA[2] < 1: # if there is current or seabed slope
                alpha = 0
                cb = min(0, dr[2]) - 100  # put the seabed out of reach (model limitation)
            else:  # otherwise proceed as usual (this is the normal case)
                alpha = 0
                cb = self.cb
        
        # horizontal and vertical dimensions of line profile (end A to B)
        LH = np.linalg.norm(dr[:2])
        LV = dr[2]
        
        
        # ----- call catenary function or alternative and save results -----
        
        #If EA is found in the line properties we will run the original catenary function 
        if 'EA' in self.type:
            try:
                (fAH, fAV, fBH, fBV, info) = catenary(LH, LV, self.L, self.EA, w,
                                                      CB=cb, alpha=alpha, HF0=self.HF, VF0=self.VF, 
                                                      Tol=tol, nNodes=self.nNodes, plots=profiles)                                                    
            except CatenaryError as error:
                raise LineError(self.number, error.message)       
        #If EA isnt found then we will use the ten-str relationship defined in the input file 
        else:
            (fAH, fAV, fBH, fBV, info) = nonlinear(LH, LV, self.L, self.type['Str'], self.type['Ten'],np.linalg.norm(w)) 
    
    
        # save line profile coordinates in global frame (involves inverse rotation)
        if profiles > 0:
            # note: instantiating new arrays rather than writing directly to self.Xs 
            # seems to be necessary to avoid plots auto-updating to the current 
            # profile of the Line object.
            Xs = np.zeros(self.nNodes)
            Ys = np.zeros(self.nNodes)
            Zs = np.zeros(self.nNodes)
            # apply inverse rotation to node positions
            for i in range(0,self.nNodes):
                temp_array = np.array([info['X'][i], 0 ,info['Z'][i]])
                unrot_pos = np.matmul(temp_array, R)
                
                Xs[i] = self.rA[0] + unrot_pos[0]
                Ys[i] = self.rA[1] + unrot_pos[1]
                Zs[i] = self.rA[2] + unrot_pos[2]

            self.Xs = Xs
            self.Ys = Ys
            self.Zs = Zs
            self.Ts = info["Te"]
        
        # save fairlead tension components for use as ICs next iteration
        self.HF = info["HF"]
        self.VF = info["VF"]
        
        # save other important info
        self.LBot = info["LBot"]
        self.info = info
        
        # save forces in global reference frame
        self.fA = np.matmul(np.array([fAH, 0, fAV]), R)
        self.fB = np.matmul(np.array([fBH, 0, fBV]), R)
        self.TA = np.linalg.norm(self.fA) # end tensions
        self.TB = np.linalg.norm(self.fB)
        
        # save 3d stiffness matrix in global orientation for both line ends (3 DOF + 3 DOF)
        self.KA  = from2Dto3Drotated(info['stiffnessA'], -fBH, LH, R.T)  # reaction at A due to motion of A
        self.KB  = from2Dto3Drotated(info['stiffnessB'], -fBH, LH, R.T)  # reaction at B due to motion of B
        self.KAB = from2Dto3Drotated(info['stiffnessAB'], fBH, LH, R.T)  # reaction at B due to motion of A

        # may want to skip stiffness calcs when just getting profiles for plotting...
        
        
        # ----- calculate current loads if applicable, for use next time -----
        
        if self.sys.currentMod == 1: 

            U = self.sys.current  # 3D current velocity [m/s]  (could be changed to depth-dependent profile)
            
            fCurrent = np.zeros(3)  # total current force on line in x, y, z [N]        
            
            # Loop through each segment along the line and add up the drag forces.
            # This is in contrast to MoorDyn calculating for nodes.
            for i in range(self.nNodes-1):
                #For each segment find the tangent vector and then calculate the current loading
                dr_seg = np.array([self.Xs[i+1] - self.Xs[i], 
                                   self.Ys[i+1] - self.Ys[i], 
                                   self.Zs[i+1] - self.Zs[i]])  # segment vector
                ds_seg = np.linalg.norm(dr_seg)
                
                if ds_seg > 0:                   # only include if segment length > 0
                    q = dr_seg/ds_seg
                    # transverse and axial current velocity components
                    Uq = np.dot(U, q) * q
                    Up = U - Uq          
                    # transverse and axial drag forces on segment
                    dp = 0.5*self.sys.rho*self.type["Cd"]        *self.type["d_vol"]*ds_seg*np.linalg.norm(Up)*Up
                    dq = 0.5*self.sys.rho*self.type["CdAx"]*np.pi*self.type["d_vol"]*ds_seg*np.linalg.norm(Uq)*Uq
                    # add to total current force on line
                    fCurrent += dp + dq    
            
            self.fCurrent = fCurrent  # save for use next call
        else:
            self.fCurrent = np.zeros(3)  # if no current, ensure this force is zero


        # ----- plot the profile if requested -----
        if profiles > 1:
            import matplotlib.pyplot as plt
            plt.plot(self.info['X'], self.info['Z'])
            plt.show()
    
    
    """ These 3 functions no longer used - can delete  
    def getEndForce(self, endB):
        '''Returns the force of the line at the specified end based on the endB value

        Parameters
        ----------
        endB : boolean
            An indicator of which end of the line is the force wanted

        Raises
        ------
        LineError
            If the given endB value is not a 1 or 0

        Returns
        -------
        fA or fB: array
            The force vector at the end of the line

        '''
        
        if endB == 1:
            return self.fB
        elif endB == 0:
            return self.fA
        else:
            raise LineError("getEndForce: endB value has to be either 1 or 0")
            
            
    def getStiffnessMatrix(self):
        '''Returns the stiffness matrix of a line derived from analytic terms in the jacobian of catenary

        Raises
        ------
        LineError
            If a singluar matrix error occurs while taking the inverse of the Line's Jacobian matrix.

        Returns
        -------
        K2_rot : matrix
            the analytic stiffness matrix of the line in the rotated frame.

        '''

        # take the inverse of the Jacobian to get the starting analytic stiffness matrix
        '''
        if np.isnan(self.jacobian[0,0]): #if self.LBot >= self.L and self.HF==0. and self.VF==0.  << handle tricky cases here?
            K = np.array([[0., 0.], [0., 1.0/self.jacobian[1,1] ]])
        else:
            try:
                K = np.linalg.inv(self.jacobian)
            except:
                raise LineError(self.number, f"Check Line Length ({self.L}), it might be too long, or check catenary ProfileType")
        '''
        
        # solve for required variables to set up the perpendicular stiffness. Keep it horizontal
        L_xy = np.linalg.norm(self.rB[:2] - self.rA[:2])
        T_xy = np.linalg.norm(self.fB[:2])
        Kt = T_xy/L_xy
        
        # initialize the line's analytic stiffness matrix in the "in-line" plane
        KA = np.array([[self.KA2[0,0], 0 , self.KA2[0,1]],
                       [     0      , Kt,      0      ],
                       [self.KA2[1,0], 0 , self.KA2[1,1]]])
                       
        KB = np.array([[self.KB2[0,0], 0 , self.KB2[0,1]],
                       [     0      , Kt,      0      ],
                       [self.KB2[1,0], 0 , self.KB2[1,1]]])
        
        # create the rotation matrix based on the heading angle that the line is from the horizontal
        R = rotationMatrix(0,0,self.th)
        
        # rotate the matrix to be about the global frame [K'] = [R][K][R]^T
        KA_rot = np.matmul(np.matmul(R, KA), R.T)
        KB_rot = np.matmul(np.matmul(R, KB), R.T)
        
        return KA_rot, KB_rot
    
    
    def getLineTens(self):
        '''Calls the catenary function to return the tensions of the Line for a quasi-static analysis'''
        
        self.staticSolve(profiles=1) # call with flag to tell Catenary to return node info (may be unnecessary)

        Ts = self.info["Te"]
        return Ts
    """

    def getTension(self, s):
        '''Returns tension at a given point along the line
        
        Parameters
        ----------
        
        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.
        
        Returns
        -------
        
        tension value(s)
        
        '''
        #if s < 0:
        #    s = -s*self.L            
        #if s > self.L:
        #    raise ValueError('Specified arc length is larger than the line unstretched length.')
        
        Te = np.interp(s, self.info['s'], self.info['Te'])
        
        return Te


    def getPosition(self, s):
        '''Returns position at a given point along the line
        
        Parameters
        ----------
        
        s : scalar or array-like
            Value or array of values for the arc length along the line from end A to end B at which
            the information is desired. Positive values are arc length in m, negative values are a
            relative location where 0 is end A, -1 is end B, and -0.5 is the midpoint.
        
        Returns
        -------
        
        position vector(s)
        
        '''
        
        # >>> should be merged with getLineCoords and getCoordinate functionality <<<
        
        x = np.interp(s, self.info['s'], self.info['X'])
        z = np.interp(s, self.info['s'], self.info['Z'])
        
        
        dr =  self.rB - self.rA                 
        LH = np.hypot(dr[0], dr[1])
        Xs = self.rA[0] + x*dr[0]/LH
        Ys = self.rA[1] + x*dr[1]/LH
        Zs = self.rA[2] + z
        
        return np.vstack([ Xs, Ys, Zs])

    
    def attachLine(self, lineID, endB):
        pass


    def activateDynamicStiffness(self, display=0):
        '''Switch mooring line model to dynamic line stiffness
        value, including potential unstretched line length
        adjustment. This only works when dynamic line properties
        are used.'''
        
        if self.type['EAd'] > 0:
            # switch to dynamic stiffness value
            EA_old = self.type['EA']
            EA_new = self.type['EAd'] + self.type['EAd_Lm']*np.mean([self.TA, self.TB])  # this implements the sloped Krd = alpha + beta*Lm
            self.EA = np.max([EA_new, EA_old])  # only if the dynamic stiffness is higher than the static stiffness, activate the dynamic stiffness
            
            # adjust line length to maintain current tension (approximate)
            self.L = self.L0 * (1 + self.TB/EA_old)/(1 + self.TB/EA_new)
            
        else:
            if display > 0:
                print(f'Line {self.number} has zero dynamic stiffness coefficient so activateDynamicStiffness does nothing.')
        
    
    def revertToStaticStiffness(self):
        '''Switch mooring line model to dynamic line stiffness
        values, including potential unstretched line length
        adjustment. This only works when dynamic line properties
        are used.'''
        
        # switch to static/default stiffness value
        self.EA = self.type['EA']
        
        # revert to original line length
        self.L = self.L0
    

def from2Dto3Drotated(K2D, F, L, R): 
    '''Initialize a line end's analytic stiffness matrix in the 
    plane of the catenary then rotate the matrix to be about the 
    global frame using [K'] = [R][K][R]^T
    
    Parameters
    ----------
    K2D : 2x2 matrix
        Planar stiffness matrix of line end [N/m]
    F : float
        Line horizontal tension component [N]
    L : float
        Line horizontal distance end-to-end [m]
    R : 3x3 matrix
        Rotation matrix from global frame to plane to the local
        X-Z plane of the line
        
    Returns
    -------
    3x3 stiffness matrix in global orientation [N/m].
    '''

    if L > 0:
        Kt = F/L         # transverse stiffness term
    else:
        Kt = 0.0
    
    K2 = np.array([[K2D[0,0], 0 , K2D[0,1]],
                   [  0     , Kt,   0     ],
                   [K2D[1,0], 0 , K2D[1,1]]])
    
    return np.matmul(np.matmul(R, K2), R.T)    
    


def RotFrm2Vect( A, B):
    '''Rodriguez rotation function, which returns the rotation matrix 
    that transforms vector A into Vector B.
    '''
    
    v = np.cross(A,B)
    ssc = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])
         
    R =  np.eye(3,3) + ssc + np.matmul(ssc,ssc)*(1-np.dot(A,B))/(np.linalg.norm(v)*np.linalg.norm(v))            

    return R