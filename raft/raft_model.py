# 2020-05-03: This is a start at a frequency-domain floating support structure model for WEIS-WISDEM.
#             Based heavily on the GA-based optimization framework from Hall 2013

# 2020-05-23: Starting as a script that will evaluate a design based on properties/data provided in separate files.

import os
import os.path as osp
import sys
import moorpy as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import raft.member2pnl as pnl
import pyhams.pyhams as ph

#import F6T1RNA as structural    # import turbine structural model functions
# reload the libraries each time in case we make any changes
import importlib
mp   = importlib.reload(mp)
ph   = importlib.reload(ph)
pnl   = importlib.reload(pnl)


class Env:
    '''This could be a simple environmental parameters class <<< needed most immediately to pass rho and g info.'''
    def __init__(self):
        self.rho = 1025.0
        self.g = 9.81
        self.Hs = 1.0
        self.Tp = 10.0
        self.spectrum = "unit"
        self.V = 10.0
        self.beta = 0.0


## This class represents linear (for now cylindrical and rectangular) components in the substructure.
#  It is meant to correspond to Member objects in the WEIS substructure ontology, but containing only
#  the properties relevant for the Level 1 frequency-domain model, as well as additional data strucutres
#  used during the model's operation.
class Member:

    def __init__(self, mi, nw, BEM=[]):
        '''Initialize a Member. For now, this function accepts a space-delimited string with all member properties.

        PARAMETERS
        ----------

        min : dict
            Dictionary containing the member description data structure
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Rotation to apply to the coordinates when setting up the member - used for circular patterns of members.

        '''

        # overall member parameters
        self.id    = int(1)                                          # set the ID value of the member
        self.name  = str(mi['name'])
        self.type  = int(mi['type'])                                 # set the type of the member (for now, just arbitrary numbers: 0,1,2, etc.)

        self.rA = np.array(mi['rA'], dtype=np.double)                # [x,y,z] coordinates of lower node [m]
        self.rB = np.array(mi['rB'], dtype=np.double)                # [x,y,z] coordinates of upper node [m]

        shape      = str(mi['shape'])                                # the shape of the cross section of the member as a string (the first letter should be c or r)

        rAB = self.rB-self.rA                                        # The relative coordinates of upper node from lower node [m]
        self.l = np.linalg.norm(rAB)                                 # member length [m]

        self.potMod = getFromDict(mi, 'potMod', dtype=bool, default=False)     # hard coding BEM analysis enabled for now <<<< need to move this to the member YAML input instead <<<
        

        # heading feature for rotation members about the z axis (used for rotated patterns)
        heading = getFromDict(mi, 'heading', default=0.0)            # rotation about z axis to apply to the member [deg]
        if heading != 0.0:
            c = np.cos(np.deg2rad(heading))
            s = np.sin(np.deg2rad(heading))
            rotMat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.rA = np.matmul(rotMat, self.rA)
            self.rB = np.matmul(rotMat, self.rB)


        # station positions
        n = len(mi['stations'])                                     # number of stations
        if n < 2:
            raise ValueError("At least two stations entries must be provided")

        A = np.array(mi['stations'], dtype=float)
        self.stations = (A - A[0])/(A[-1] - A[0])*self.l             # calculate station positions along the member axis from 0 to l [m]


        # shapes
        if shape[0].lower() == 'c':
            self.shape = 'circular'
            self.d     = getFromDict(mi, 'd', shape=n)               # diameter of member nodes [m]  <<< should maybe check length of all of these

            self.gamma = 0                                           # twist angle about the member's z-axis [degrees] (don't need this for a circular member)

        elif shape[0].lower() == 'r':   # <<< this case not checked yet since update <<<
            self.shape = 'rectangular'

            self.sl    = getFromDict(mi, 'd', shape=[n,2])           # array of side lengths of nodes along member [m]

            self.gamma = getFromDict(mi, 'gamma', default=0.)        # twist angle about the member's z-axis [degrees] (if gamma=90, then the side lengths are flipped)

        else:
            raise ValueError('The only allowable shape strings are circular and rectangular')


        self.t         = getFromDict(mi, 't', shape=n)               # shell thickness of the nodes [m]

        self.l_fill    = getFromDict(mi, 'l_fill'  , shape=-1, default=0.0)   # length of member (from end A to B) filled with ballast [m]
        self.rho_fill  = getFromDict(mi, 'rho_fill', shape=-1, default=0.0)   # density of ballast in member [kg/m^3]

        self.rho_shell = getFromDict(mi, 'rho_shell', default=8500.) # shell mass density [kg/m^3]


        # initialize member orientation variables
        self.q = rAB/self.l                                         # member axial unit vector
        self.p1 = np.zeros(3)                                       # member transverse unit vectors (to be filled in later)
        self.p2 = np.zeros(3)                                       # member transverse unit vectors
        self.R = np.eye(3)                                          # rotation matrix from global x,y,z to member q,p1,p2


        # store end cap and bulkhead info

        cap_stations = getFromDict(mi, 'cap_stations', shape=-1, default=[])   # station location inputs before scaling
        if cap_stations == []:
            self.cap_t        = []
            self.cap_d_in     = []
            self.cap_stations = []
        else:
            self.cap_t        = getFromDict(mi, 'cap_t'   , shape=cap_stations.shape)   # thicknesses [m]
            self.cap_d_in     = getFromDict(mi, 'cap_d_in', shape=cap_stations.shape)   # inner diameter (if it isn't a solid plate) [m]
            self.cap_stations = (cap_stations - A[0])/(A[-1] - A[0])*self.l             # calculate station positions along the member axis from 0 to l [m]
            

        # Drag coefficients
        self.Cd_q   = getFromDict(mi, 'Cd_q' , shape=n, default=0.0 )     # axial drag coefficient
        self.Cd_p1  = getFromDict(mi, 'Cd'   , shape=n, default=0.6 )     # transverse1 drag coefficient
        self.Cd_p2  = getFromDict(mi, 'Cd'   , shape=n, default=0.6 )     # transverse2 drag coefficient
        self.Cd_End = getFromDict(mi, 'CdEnd', shape=n, default=0.6 )     # end drag coefficient
        # Added mass coefficients
        self.Ca_q   = getFromDict(mi, 'Ca_q' , shape=n, default=0.0 )     # axial added mass coefficient
        self.Ca_p1  = getFromDict(mi, 'Ca'   , shape=n, default=0.97)     # transverse1 added mass coefficient
        self.Ca_p2  = getFromDict(mi, 'Ca'   , shape=n, default=0.97)     # transverse2 added mass coefficient
        self.Ca_End = getFromDict(mi, 'CaEnd', shape=n, default=0.6 )     # end added mass coefficient


        # discretize into strips with a node at the midpoint of each strip (flat surfaces have dl=0)
        dorsl  = list(self.d) if self.shape=='circular' else list(self.sl)   # get a variable that is either diameter of side length pair
        dlsMax = mi['dlsMax']
        #dlsMax = 5.0                  # maximum node spacing <<< this should be an optional input at some point <<<
        ls     = [0.0]                 # list of lengths along member axis where a node is located <<< should these be midpoints instead of ends???
        dls    = [0.0]                 # lumped node lengths (end nodes have half the segment length)
        ds     = [0.5*dorsl[0]]       # mean diameter or side length pair of each strip
        drs    = [0.5*dorsl[0]]       # change in radius (or side half-length pair) over each strip (from node i-1 to node i)

        for i in range(1,n):

            lstrip = self.stations[i]-self.stations[i-1]             # the axial length of the strip

            if lstrip > 0.0:
                ns= int(np.ceil( (lstrip) / dlsMax ))
                dlstrip = lstrip/ns
                m   = 0.5*(dorsl[i] - dorsl[i-1])/lstrip          # taper ratio
                ls  += [self.stations[i-1] + dlstrip*(0.5+j) for j in range(ns)] # add node locations
                dls += [dlstrip]*ns
                ds  += [dorsl[i-1] + dlstrip*2*m*(0.5+j) for j in range(ns)]
                drs += [dlstrip*m]*ns
                
            elif lstrip == 0.0:                                      # flat plate case (ends, and any flat transitions)
                ns = 1
                dlstrip = 0
                ls  += [self.stations[i-1]]                          # add node location
                dls += [dlstrip]
                ds  += [0.5*(dorsl[i-1] + dorsl[i])]               # set diameter as midpoint diameter
                drs += [0.5*(dorsl[i] - dorsl[i-1])]

        self.ns  = len(ls)                                           # number of hydrodynamic strip theory nodes per member
        self.ls  = np.array(ls, dtype=float)                          # node locations along member axis
        #self.dl = 0.5*(np.diff([0.]+lh) + np.diff(lh+[lh[-1]]))
        self.dls = np.array(dls)
        self.ds  = np.array(ds)
        self.drs = np.array(drs)
        self.mh  = np.array(m)

        self.r   = np.zeros([self.ns,3])                             # undisplaced node positions along member  [m]

        for i in range(self.ns):
            self.r[i,:] = self.rA + (ls[i]/self.l)*rAB               # locations of hydrodynamics nodes

        #self.slh[i,0] = np.interp(lh[i], self.stations, self.sl1)
        #self.slh[i,1] = np.interp(lh[i], self.stations, self.sl2)

        # complex frequency-dependent amplitudes of quantities at each node along member (to be filled in later)
        self.dr        = np.zeros([self.ns,3,nw], dtype=complex)            # displacement
        self.v         = np.zeros([self.ns,3,nw], dtype=complex)            # velocity
        self.a         = np.zeros([self.ns,3,nw], dtype=complex)            # acceleration
        self.u         = np.zeros([self.ns,3,nw], dtype=complex)            # wave velocity
        self.ud        = np.zeros([self.ns,3,nw], dtype=complex)            # wave acceleration
        self.pDyn      = np.zeros([self.ns,  nw], dtype=complex)            # dynamic pressure
        self.F_exc_iner= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from inertia (Froude-Krylov)
        self.F_exc_drag= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from linearized drag



    def calcOrientation(self):
        '''Calculates member direction vectors q, p1, and p2 as well as member orientation matrix R
        based on the end positions and twist angle gamma.'''

        rAB = self.rB-self.rA                                       # displacement vector from end A to end B [m]
        q = rAB/np.linalg.norm(rAB)                                 # member axial unit vector

        beta = np.arctan2(q[1],q[0])                                # member incline heading from x axis
        phi  = np.arctan2(np.sqrt(q[0]**2 + q[1]**2), q[2])         # member incline angle from vertical

        # trig terms for Euler angles rotation based on beta, phi, and gamma
        s1 = np.sin(beta)
        c1 = np.cos(beta)
        s2 = np.sin(phi)
        c2 = np.cos(phi)
        s3 = np.sin(np.deg2rad(self.gamma))
        c3 = np.cos(np.deg2rad(self.gamma))

        R = np.array([[ c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3,  c1*s2],
                      [ c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3,  s1*s2],
                      [   -c3*s2      ,      s2*s3     ,    c2 ]])  #Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix


        p1 = np.matmul( R, [1,0,0] )               # unit vector that is perpendicular to the 'beta' plane if gamma is zero
        p2 = np.cross( q, p1 )                     # unit vector orthogonal to both p1 and q

        self.R  = R
        self.q  = q
        self.p1 = p1
        self.p2 = p2

        # matrices of vector multiplied by vector transposed, used in computing force components
        self.qMat  = VecVecTrans(self.q)
        self.p1Mat = VecVecTrans(self.p1)
        self.p2Mat = VecVecTrans(self.p2)


        return q, p1, p2  # also return the unit vectors for convenience



    def getInertia(self):
        '''Calculates member inertia properties: mass, center of mass, moments of inertia.
        Assumes that the members are continuous and symmetrical (i.e. no weird shapes)'''

        # Moment of Inertia Helper Functions -----------------------
        def FrustumMOI(dA, dB, H, p):
            '''returns the radial and axial moments of inertia of a potentially tapered circular member about the end node.
            Previously used equations found in a HydroDyn paper, now it uses newly derived ones. Ask Stein for reference if needed'''
            if H==0:        # if there's no height, mainly refering to no ballast, there shouldn't be any extra MoI
                I_rad = 0                                                   # radial MoI about end node [kg-m^2]
                I_ax = 0                                                    # axial MoI about axial axis [kg-m^2]
            else:
                if dA==dB:  # if it's a cylinder
                    r1 = dA/2                                               # bottom radius [m]
                    r2 = dB/2                                               # top radius [m]
                    I_rad = (1/12)*(p*H*np.pi*r1**2)*(3*r1**2 + 4*H**2)     # radial MoI about end node [kg-m^2]
                    I_ax = (1/2)*p*np.pi*H*r1**4                            # axial MoI about axial axis [kg-m^2]
                else:       # if it's a tapered cylinder (frustum)
                    r1 = dA/2                                               # bottom radius [m]
                    r2 = dB/2                                               # top radius [m]
                    I_rad = (1/20)*p*np.pi*H*(r2**5 - r1**5)/(r2 - r1) + (1/30)*p*np.pi*H**3*(r1**2 + 3*r1*r2 + 6*r2**2) # radial MoI about end node [kg-m^2]
                    I_ax = (1/10)*p*np.pi*H*(r2**5-r1**5)/(r2-r1)           # axial MoI about axial axis [kg-m^2]

            return I_rad, I_ax

        def RectangularFrustumMOI(La, Wa, Lb, Wb, H, p):
            '''returns the moments of inertia about the end node of a cuboid that can be tapered.
            - Inputs the lengths and widths at the top and bottom of the cuboid, as well as the height and material density.
            - L is the side length along the local x-direction, W is the side length along the local y-direction.
            - Does not work for members that are not symmetrical about the axial axis.
            - Works for cases when it is a perfect cuboid, a truncated pyramid, and a truncated triangular prism
            - Equations derived by hand, ask Stein for reference if needed'''

            if H==0: # if there's no height, mainly refering to no ballast, there shouldn't be any extra MoI
                Ixx = 0                                         # MoI around the local x-axis about the end node [kg-m^2]
                Iyy = 0                                         # MoI around the local y-axis about the end node [kg-m^2]
                Izz = 0                                         # MoI around the local z-axis about the axial axis [kg-m^2]
            else:
                if La==Lb and Wa==Wb: # if it's a cuboid
                    L = La                                      # length of the cuboid (La=Lb) [m]
                    W = Wa                                      # width of the cuboid (Wa=Wb) [m]
                    M = p*L*W*H                                 # mass of the cuboid [kg]

                    Ixx = (1/12)*M*(W**2 + 4*H**2)              # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = (1/12)*M*(L**2 + 4*H**2)              # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = (1/12)*M*(L**2 + W**2)                # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La!=Lb and Wa!=Wb: # if it's a truncated pyramid for both side lengths

                    x2 = (1/12)*p* ( (Lb-La)**3*H*(Wb/5 + Wa/20) + (Lb-La)**2*La*H(3*Wb/4 + Wa/4) + \
                                     (Lb-La)*La**2*H*(Wb + Wa/2) + La**3*H*(Wb/2 + Wa/2) )

                    y2 = (1/12)*p* ( (Wb-Wa)**3*H*(Lb/5 + La/20) + (Wb-Wa)**2*Wa*H(3*Lb/4 + La/4) + \
                                     (Wb-Wa)*Wa**2*H*(Lb + La/2) + Wa**3*H*(Lb/2 + La/2) )

                    z2 = p*( Wb*Lb/5 + Wa*Lb/20 + La*Wb/20 + Wa*La*(8/15) )

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La==Lb and Wa!=Wb: # if it's a truncated triangular prism where only the lengths are the same on top and bottom
                    L = La                                      # length of the truncated triangular prism [m]

                    x2 = (1/24)*p*(L**3)*H*(Wb+Wa)
                    y2 = (1/48)*p*L*H*( Wb**3 + Wa*Wb**2 + Wa**2*Wb + Wa**3 )
                    z2 = (1/12)*p*L*(H**3)*( 3*Wb + Wa )

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La!=Lb and Wa==Wb: # if it's a truncated triangular prism where only the widths are the same on top and bottom
                    W = Wa                                      # width of the truncated triangular prism [m]

                    x2 = (1/48)*p*W*H*( Lb**3 + La*Lb**2 + La**2*Lb + La**3 )
                    y2 = (1/24)*p*(W**3)*H*(Lb+La)
                    z2 = (1/12)*p*W*(H**3)*( 3*Lb + La )

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                else:
                    raise ValueError('You either have inconsistent inputs, or you are trying to calculate the MoI of a member that is not supported')

            return Ixx, Iyy, Izz



        # ------- member inertial calculations ---------
        n = len(self.stations)                          # set n as the number of stations = number of sub-members plus 1

        mass_center = 0                                 # total sum of mass the center of mass of the member [kg-m]
        mshell = 0                                      # total mass of the shell material only of the member [kg]
        mfill = []                                      # list of ballast masses in each submember [kg]
        pfill = []                                      # list of ballast densities in each submember [kg]
        self.M_struc = np.zeros([6,6])                  # member mass/inertia matrix [kg, kg-m, kg-m^2]

        # loop through each sub-member
        for i in range(1,n):                            # start at 1 rather than 0 because we're looking at the sections (from station i-1 to i)

            # initialize common variables
            rA = self.rA + self.q*self.stations[i-1]    # lower node position of the submember [m]
            l = self.stations[i]-self.stations[i-1]     # length of the submember [m]
            if l==0.0:
                mass = 0
                center = np.zeros(3)
                m_shell = 0
                m_fill = 0
                rho_fill = 0
            else:
                # if the following variables are input as scalars, keep them that way, if they're vectors, take the [i-1]th value
                rho_shell = self.rho_shell              # density of the shell material [kg/m^3]
                if np.isscalar(self.l_fill):            # set up l_fill and rho_fill based on whether it's scalar or not
                    l_fill = self.l_fill
                else:
                    l_fill = self.l_fill[i-1]
                if np.isscalar(self.rho_fill):
                    rho_fill = self.rho_fill
                else:
                    rho_fill = self.rho_fill[i-1]
    
                
                if self.shape=='circular':
                    # MASS AND CENTER OF GRAVITY
                    dA = self.d[i-1]                        # outer diameter of the lower node [m]
                    dB = self.d[i]                          # outer diameter of the upper node [m]
                    dAi = self.d[i-1] - 2*self.t[i-1]       # inner diameter of the lower node [m]
                    dBi = self.d[i] - 2*self.t[i]           # inner diameter of the upper node [m]
                    
                    V_outer, hco = FrustumVCV(dA, dB, l)    # volume and center of volume of solid frustum with outer diameters [m^3] [m]
                    V_inner, hci = FrustumVCV(dAi, dBi, l)  # volume and center of volume of solid frustum with inner diameters [m^3] [m] 
                    v_shell = V_outer-V_inner               # volume of hollow frustum with shell thickness [m^3]
                    m_shell = v_shell*rho_shell             # mass of hollow frustum [kg]
                    
                    hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)  # center of volume of hollow frustum with shell thickness [m]
                         
                    dBi_fill = (dBi-dAi)*(l_fill/l) + dAi   # interpolated inner diameter of frustum that ballast is filled to [m] 
                    v_fill, hc_fill = FrustumVCV(dAi, dBi_fill, l_fill)         # volume and center of volume of solid inner frustum that ballast occupies [m^3] [m]
                    m_fill = v_fill*rho_fill                # mass of the ballast in the submember [kg]
                    
                    # <<< The ballast is calculated as if it starts at the same end as the shell, however, if the end of the sub-member has an end cap,
                    # then the ballast sits on top of the end cap. Depending on the thickness of the end cap, this can affect m_fill, hc_fill, and MoI_fill >>>>>
                    
                    mass = m_shell + m_fill                 # total mass of the submember [kg]
                    hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass       # total center of mass of the submember from the submember's rA location [m]
                    
                    center = rA + (self.q*hc)               # total center of mass of the member from the PRP [m]
                    
                    # MOMENT OF INERTIA
                    I_rad_end_outer, I_ax_outer = FrustumMOI(dA, dB, l, rho_shell)          # radial and axial MoI about the end of the solid outer frustum [kg-m^2]
                    I_rad_end_inner, I_ax_inner = FrustumMOI(dAi, dBi, l, rho_shell)        # radial and axial MoI about the end of the imaginary solid inner frustum [kg-m^2]
                    I_rad_end_shell = I_rad_end_outer-I_rad_end_inner                       # radial MoI about the end of the frustum shell through superposition [kg-m^2]
                    I_ax_shell = I_ax_outer - I_ax_inner                                    # axial MoI of the shell through superposition [kg-m^2]
                    
                    I_rad_end_fill, I_ax_fill = FrustumMOI(dAi, dBi_fill, l_fill, rho_fill) # radial and axial MoI about the end of the solid inner ballast frustum [kg-m^2]
                    
                    I_rad_end = I_rad_end_shell + I_rad_end_fill                            # radial MoI about the end of the submember [kg-m^2]
                    I_rad = I_rad_end - mass*hc**2                                          # radial MoI about the CoG of the submember through the parallel axis theorem [kg-m^2]
                    
                    I_ax = I_ax_shell + I_ax_fill                                           # axial MoI of the submember about the total CoG (= about the end also bc axial)
    
                    Ixx = I_rad                             # circular, so the radial MoI is about the x and y axes
                    Iyy = I_rad                             # circular, so the radial MoI is about the x and y axes
                    Izz = I_ax                              # circular, so the axial MoI is about the z axis
                    
                
                elif self.shape=='rectangular':
                    # MASS AND CENTER OF GRAVITY
                    slA = self.sl[i-1]                          # outer side lengths of the lower node, of length 2 [m]
                    slB = self.sl[i]                            # outer side lengths of the upper node, of length 2 [m]
                    slAi = self.sl[i-1] - 2*self.t[i-1]         # inner side lengths of the lower node, of length 2 [m]
                    slBi = self.sl[i] - 2*self.t[i]             # inner side lengths of the upper node, of length 2 [m]
                    
                    V_outer, hco = FrustumVCV(slA, slB, l)      # volume and center of volume of solid frustum with outer side lengths [m^3] [m]
                    V_inner, hci = FrustumVCV(slAi, slBi, l)    # volume and center of volume of solid frustum with inner side lengths [m^3] [m]
                    v_shell = V_outer-V_inner                   # volume of hollow frustum with shell thickness [m^3]
                    m_shell = v_shell*rho_shell                 # mass of hollow frustum [kg]
                    
                    hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)  # center of volume of the hollow frustum with shell thickness [m]
                                        
                    slBi_fill = (slBi-slAi)*(l_fill/l) + slAi   # interpolated side lengths of frustum that ballast is filled to [m]
                    v_fill, hc_fill = FrustumVCV(slAi, slBi_fill, l_fill)   # volume and center of volume of inner frustum that ballast occupies [m^3]
                    m_fill = v_fill*rho_fill                    # mass of ballast in the submember [kg]
                    
                    mass = m_shell + m_fill                     # total mass of the submember [kg]
                    hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass       # total center of mass of the submember from the submember's rA location [m]
                    
                    center = rA + (self.q*hc)                   # total center of mass of the member from the PRP [m]
                    
                    # MOMENT OF INERTIA
                    # MoI about each axis at the bottom end node of the solid outer truncated pyramid [kg-m^2]
                    Ixx_end_outer, Iyy_end_outer, Izz_end_outer = RectangularFrustumMOI(slA[0], slA[1], slB[0], slB[1], l, rho_shell)
                    # MoI about each axis at the bottom end node of the solid imaginary inner truncated pyramid [kg-m^2]
                    Ixx_end_inner, Iyy_end_inner, Izz_end_inner = RectangularFrustumMOI(slAi[0], slAi[1], slBi[0], slBi[1], l, rho_shell)
                    # MoI about each axis at the bottom end node of the shell using superposition [kg-m^2]
                    Ixx_end_shell = Ixx_end_outer - Ixx_end_inner
                    Iyy_end_shell = Iyy_end_outer - Iyy_end_inner
                    Izz_end_shell = Izz_end_outer - Izz_end_inner
                    
                    # MoI about each axis at the bottom end node of the solid inner ballast truncated pyramid [kg-m^2]
                    Ixx_end_fill, Iyy_end_fill, Izz_end_fill = RectangularFrustumMOI(slAi[0], slAi[1], slBi_fill[0], slBi_fill[1], l_fill, rho_fill)
                    
                    # total MoI of each axis at the center of gravity of the member using the parallel axis theorem [kg-m^2]
                    Ixx_end = Ixx_end_shell + Ixx_end_fill
                    Ixx = Ixx_end - mass*hc**2
                    Iyy_end = Iyy_end_shell + Iyy_end_fill
                    Iyy = Iyy_end - mass*hc**2
                    
                    Izz_end = Izz_end_shell + Izz_end_fill
                    Izz = Izz_end       # the total MoI of the member about the z-axis is the same at any point along the z-axis



            # add/append terms
            mass_center += mass*center                  # total sum of mass the center of mass of the member [kg-m]
            mshell += m_shell                           # total mass of the shell material only of the member [kg]
            mfill.append(m_fill)                        # list of ballast masses in each submember [kg]
            pfill.append(rho_fill)                     # list of ballast densities in each submember [kg]

            # create a local submember mass matrix
            Mmat = np.diag([mass, mass, mass, 0, 0, 0]) # submember's mass matrix without MoI tensor
            # create the local submember MoI tensor in the correct directions
            I = np.diag([Ixx, Iyy, Izz])                # MoI matrix about the member's local CG. 0's on off diagonals because of symmetry
            T = self.R.T                                # transformation matrix to unrotate the member's local axes. Transposed because rotating axes.
            I_rot = np.matmul(T.T, np.matmul(I,T))      # MoI about the member's local CG with axes in same direction as global axes. [I'] = [T][I][T]^T -> [T]^T[I'][T] = [I]

            Mmat[3:,3:] = I_rot     # mass and inertia matrix about the submember's CG in unrotated, but translated local frame

            # translate this submember's local inertia matrix to the PRP and add it to the total member's M_struc matrix
            self.M_struc += translateMatrix6to6DOF(center, Mmat) # mass matrix of the member about the PRP


            # end of submember for loop


        # END CAPS/BULKHEADS
        # --------- Add the inertia properties of any end caps ---------
        self.m_cap_list = []
        # Loop through each cap or bulkhead
        for i in range(len(self.cap_stations)):

            L = self.cap_stations[i]        # The station position along the member where there is a cap or bulkhead
            h = self.cap_t[i]               # The thickness, or height of the cap or bulkhead [m]
            rho_cap = self.rho_shell        # set the cap density to the density of the member for now [kg/m^3]

            if self.shape=='circular':
                d_hole = self.cap_d_in[i]   # The diameter of the missing hole in the middle, if any [m]
                d = self.d-2*self.t         # The list of inner diameters along the member [m]

                if L==self.stations[0]:  # if the cap is on the bottom end of the member
                    dA = d[0]
                    dB = np.interp(L+h, self.stations, d)
                    dAi = d_hole
                    dBi = dB*(dAi/dA)       # keep the same proportion in d_hole from bottom to top
                elif L==self.stations[-1]:    # if the cap is on the top end of the member
                    dA = np.interp(L-h, self.stations, d)
                    dB = d[-1]
                    dBi = d_hole
                    dAi = dA*(dBi/dB)
                elif (L > self.stations[0] and L < self.stations[0] + h) or (L < self.stations[-1] and L > self.stations[-1] - h):
                    # there could be another case where 0 < L < h or self.l-h < L < self.l
                    # this would cause the inner member to stick out beyond the end point based on the following else calcs
                    # not including this for now since the modeler should be aware to not do this
                    raise ValueError('This setup cannot be handled by getIneria yet')
                elif i < len(self.cap_stations)-1 and L==self.cap_stations[i+1]: # if there's a discontinuity in the member and l=0
                    dA = np.interp(L-h, self.stations, d)   # make an end cap going down from the lower member
                    dB = d[i]
                    dBi = d_hole
                    dAi = dA*(dBi/dB)
                elif i > 0 and L==self.cap_stations[i-1]:   # and make an end cap going up from the upper member
                    dA = d[i]
                    dB = np.interp(L+h, self.stations, d)
                    dAi = d_hole
                    dBi = dB*(dAi/dA)
                else:
                    dA = np.interp(L-h/2, self.stations, d)
                    dB = np.interp(L+h/2, self.stations, d)
                    dM = np.interp(L, self.stations, d)         # find the diameter at the middle, where L is referencing
                    dMi = d_hole
                    dAi = dA*(dMi/dM)
                    dBi = dB*(dMi/dM)

                # run inertial calculations for circular caps/bulkheads
                V_outer, hco = FrustumVCV(dA, dB, h)
                V_inner, hci = FrustumVCV(dAi, dBi, h)
                v_cap = V_outer-V_inner
                m_cap = v_cap*rho_cap    # assume it's made out of the same material as the shell for now (can add in cap density input later if needed)
                hc_cap = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)

                pos_cap = self.rA + self.q*L                    # position of the referenced cap station from the PRP
                if L==self.stations[0]:         # if it's a bottom end cap, the position is at the bottom of the end cap
                    center_cap = pos_cap + self.q*hc_cap            # and the CG of the cap is at hc from the bottom, so this is the simple case
                elif L==self.stations[-1]:      # if it's a top end cap, the position is at the top of the end cap
                    center_cap = pos_cap - self.q*(h - hc_cap)      # and the CG of the cap goes from the top, to h below the top, to hc above h below the top (wording...sorry)
                else:                           # if it's a middle bulkhead, the position is at the middle of the bulkhead
                    center_cap = pos_cap - self.q*((h/2) - hc_cap)  # so the CG goes from the middle of the bulkhead, down h/2, then up hc

                I_rad_end_outer, I_ax_outer = FrustumMOI(dA, dB, h, rho_cap)
                I_rad_end_inner, I_ax_inner = FrustumMOI(dAi, dBi, h, rho_cap)
                I_rad_end = I_rad_end_outer-I_rad_end_inner
                I_rad = I_rad_end - m_cap*hc_cap**2
                I_ax = I_ax_outer - I_ax_inner

                Ixx = I_rad
                Iyy = I_rad
                Izz = I_ax



            elif self.shape=='rectangular':
                sl_hole = self.cap_d_in[i,:]
                sl = self.sl - 2*self.t

                if L==self.stations[0]:  # if the cap is on the bottom end of the member
                    slA = sl[0,:]
                    slB = np.interp(L+h, self.stations, sl)
                    slAi = sl_hole
                    slBi = slB*(slAi/slA)       # keep the same proportion in d_hole from bottom to top
                elif L==self.stations[-1]:    # if the cap is on the top end of the member
                    slA = np.interp(L-h, self.stations, sl)
                    slB = sl[-1,:]
                    slAi = slA*(slBi/slB)
                    slBi = sl_hole
                elif (L > self.stations[0] and L < self.stations[0] + h) or (L < self.stations[-1] and L > self.stations[-1] - h):
                    # there could be another case where 0 < L < h or self.l-h < L < self.l
                    # this would cause the inner member to stick out beyond the end point based on the following else calcs
                    # not including this for now since the modeler should be aware to not do this
                    raise ValueError('This setup cannot be handled by getIneria yet')
                elif i < len(self.cap_stations)-1 and L==self.cap_stations[i+1]:
                    slA = np.interp(L-h, self.stations, sl)
                    slB = sl[i]
                    slBi = sl_hole
                    slAi = slA*(slBi/slB)
                elif i > 0 and L==self.cap_stations[i-1]:
                    slA = sl[i]
                    slB = np.interp(L+h, self.stations, sl)
                    slAi = sl_hole
                    slBi = slB*(slAi/slA)
                else:
                    slA = np.interp(L-h/2, self.stations, sl)
                    slB = np.interp(L+h/2, self.stations, sl)
                    slM = np.interp(L, self.stations, sl)
                    slMi = sl_hole
                    slAi = slA*(slMi/slM)
                    slBi = slB*(slMi/slM)


                # run inertial calculations for rectangular caps/bulkheads
                V_outer, hco = FrustumVCV(slA, slB, h)
                V_inner, hci = FrustumVCV(slAi, slBi, h)
                v_cap = V_outer-V_inner
                m_cap = v_cap*rho_cap    # assume it's made out of the same material as the shell for now (can add in cap density input later if needed)
                hc_cap = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)

                pos_cap = self.rA + self.q*L                    # position of the referenced cap station from the PRP
                if L==self.stations[0]:         # if it's a bottom end cap, the position is at the bottom of the end cap
                    center_cap = pos_cap + self.q*hc_cap            # and the CG of the cap is at hc from the bottom, so this is the simple case
                elif L==self.stations[-1]:      # if it's a top end cap, the position is at the top of the end cap
                    center_cap = pos_cap - self.q*(h - hc_cap)      # and the CG of the cap goes from the top, to h below the top, to hc above h below the top (wording...sorry)
                else:                           # if it's a middle bulkhead, the position is at the middle of the bulkhead
                    center_cap = pos_cap - self.q*((h/2) - hc_cap)  # so the CG goes from the middle of the bulkhead, down h/2, then up hc

                Ixx_end_outer, Iyy_end_outer, Izz_end_outer = RectangularFrustumMOI(slA, slB, h, rho_cap)
                Ixx_end_inner, Iyy_end_inner, Izz_end_inner = RectangularFrustumMOI(slAi, slBi, h, rho_cap)
                Ixx_end = Ixx_end_outer-Ixx_end_inner
                Iyy_end = Iyy_end_outer-Iyy_end_inner
                Izz_end = Izz_end_outer-Izz_end_inner
                Ixx = Ixx_end - m_cap*hc_cap**2
                Iyy = Iyy_end - m_cap*hc_cap**2
                Izz = Izz_end


            # ----- add properties to relevant variables -----
            mass_center += m_cap*center_cap
            mshell += m_cap                # include end caps and bulkheads in the mass of the shell
            self.m_cap_list.append(m_cap)

            # create a local submember mass matrix
            Mmat = np.diag([m_cap, m_cap, m_cap, 0, 0, 0]) # submember's mass matrix without MoI tensor
            # create the local submember MoI tensor in the correct directions
            I = np.diag([Ixx, Iyy, Izz])                # MoI matrix about the member's local CG. 0's on off diagonals because of symmetry
            T = self.R.T                                # transformation matrix to unrotate the member's local axes. Transposed because rotating axes.
            I_rot = np.matmul(T.T, np.matmul(I,T))      # MoI about the member's local CG with axes in same direction as global axes. [I'] = [T][I][T]^T -> [T]^T[I'][T] = [I]

            Mmat[3:,3:] = I_rot     # mass and inertia matrix about the submember's CG in unrotated, but translated local frame

            # translate this submember's local inertia matrix to the PRP and add it to the total member's M_struc matrix
            self.M_struc += translateMatrix6to6DOF(center_cap, Mmat) # mass matrix of the member about the PRP



        mass = self.M_struc[0,0]        # total mass of the entire member [kg]
        center = mass_center/mass       # total center of mass of the entire member from the PRP [m]


        return mass, center, mshell, mfill, pfill




    def getHydrostatics(self, env):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix'''

        pi = np.pi

        # initialize some values that will be returned
        Fvec = np.zeros(6)              # this will get added to by each segment of the member
        Cmat = np.zeros([6,6])          # this will get added to by each segment of the member
        V_UW = 0                        # this will get added to by each segment of the member
        r_centerV = np.zeros(3)         # center of buoyancy times volumen total - will get added to by each segment
        # these will only get changed once, if there is a portion crossing the water plane
        AWP = 0
        IWP = 0
        xWP = 0
        yWP = 0


        # loop through each member segment, and treat each segment like how we used to treat each member
        n = len(self.stations)

        for i in range(1,n):     # starting at 1 rather than 0 because we're looking at the sections (from station i-1 to i)

            # calculate end locations for this segment only
            rA = self.rA + self.q*self.stations[i-1]
            rB = self.rA + self.q*self.stations[i  ]

            # partially submerged case
            if rA[2]*rB[2] <= 0:    # if member crosses (or touches) water plane

                # angles
                beta = np.arctan2(self.q[1],self.q[0])  # member incline heading from x axis
                phi  = np.arctan2(np.sqrt(self.q[0]**2 + self.q[1]**2), self.q[2])  # member incline angle from vertical

                # precalculate trig functions
                cosPhi=np.cos(phi)
                sinPhi=np.sin(phi)
                tanPhi=np.tan(phi)
                cosBeta=np.cos(beta)
                sinBeta=np.sin(beta)
                tanBeta=sinBeta/cosBeta

                def intrp(x, xA, xB, yA, yB):  # little basic interpolation function for 2 values rather than a vector
                    return yA + (x-xA)*(yB-yA)/(xB-xA)

                # -------------------- buoyancy and waterplane area properties ------------------------

                xWP = intrp(0, rA[2], rB[2], rA[0], rB[0])                     # x coordinate where member axis cross the waterplane [m]
                yWP = intrp(0, rA[2], rB[2], rA[1], rB[1])                     # y coordinate where member axis cross the waterplane [m]
                if self.shape=='circular':
                    dWP = intrp(0, rA[2], rB[2], self.d[i], self.d[i-1])       # diameter of member where its axis crosses the waterplane [m]
                    AWP = (np.pi/4)*dWP**2                                     # waterplane area of member [m^2]
                    IWP = (np.pi/64)*dWP**4                                    # waterplane moment of inertia [m^4] approximates as a circle
                    IxWP = IWP                                                 # MoI of circular waterplane is the same all around
                    IyWP = IWP                                                 # MoI of circular waterplane is the same all around
                elif self.shape=='rectangular':
                    slWP = intrp(0, rA[2], rB[2], self.sl[i], self.sl[i-1])    # side lengths of member where its axis crosses the waterplane [m]
                    AWP = slWP[0]*slWP[1]                                      # waterplane area of rectangular member [m^2]
                    IxWP = (1/12)*slWP[0]*slWP[1]**3                           # waterplane MoI [m^4] about the member's LOCAL x-axis, not the global x-axis
                    IyWP = (1/12)*slWP[0]**3*slWP[0]                           # waterplane MoI [m^4] about the member's LOCAL y-axis, not the global y-axis
                    I = np.diag([IxWP, IyWP, 0])                               # area moment of inertia tensor
                    T = self.R.T                                               # the transformation matrix to unrotate the member's local axes
                    I_rot = np.matmul(T.T, np.matmul(I,T))                     # area moment of inertia tensor where MoI axes are now in the same direction as PRP
                    IxWP = I_rot[0,0]
                    IyWP = I_rot[1,1]

                LWP = abs(rA[2])/cosPhi                   # get length of segment along member axis that is underwater [m]

                # Assumption: the areas and MoI of the waterplane are as if the member were completely vertical, i.e. it doesn't account for phi
                # This can be fixed later on if needed. We're using this assumption since the fix wouldn't significantly affect the outputs

                # Total enclosed underwater volume [m^3] and distance along axis from end A to center of buoyancy of member [m]
                if self.shape=='circular':
                    V_UWi, hc = FrustumVCV(self.d[i-1], dWP, LWP)
                elif self.shape=='rectangular':
                    V_UWi, hc = FrustumVCV(self.sl[i-1], slWP, LWP)

                r_center = rA + self.q*hc          # absolute coordinates of center of volume of this segment [m]


                # >>>> question: should this function be able to use displaced/rotated values? <<<<

                # ------------- get hydrostatic derivatives ----------------

                # derivatives from global to local
                dPhi_dThx  = -sinBeta                     # \frac{d\phi}{d\theta_x} = \sin\beta
                dPhi_dThy  =  cosBeta
                dFz_dz   = -env.rho*env.g*AWP /cosPhi

                # note: below calculations are based on untapered case, but
                # temporarily approximated for taper by using dWP (diameter at water plane crossing) <<< this is rough

                # buoyancy force and moment about end A
                Fz = env.rho*env.g* V_UWi
                M  = -env.rho*env.g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
                Mx = M*dPhi_dThx
                My = M*dPhi_dThy

                Fvec[2] += Fz                           # vertical buoyancy force [N]
                Fvec[3] += Mx + Fz*rA[1]                # moment about x axis [N-m]
                Fvec[4] += My - Fz*rA[0]                # moment about y axis [N-m]


                # normal approach to hydrostatic stiffness, using this temporarily until above fancier approach is verified
                Cmat[2,2] += -dFz_dz
                Cmat[2,3] += env.rho*env.g*(     -AWP*yWP    )
                Cmat[2,4] += env.rho*env.g*(      AWP*xWP    )
                Cmat[3,2] += env.rho*env.g*(     -AWP*yWP    )
                Cmat[3,3] += env.rho*env.g*(IxWP + AWP*yWP**2 )
                Cmat[3,4] += env.rho*env.g*(      AWP*xWP*yWP)
                Cmat[4,2] += env.rho*env.g*(      AWP*xWP    )
                Cmat[4,3] += env.rho*env.g*(      AWP*xWP*yWP)
                Cmat[4,4] += env.rho*env.g*(IyWP + AWP*xWP**2 )

                Cmat[3,3] += env.rho*env.g*V_UWi * r_center[2]
                Cmat[4,4] += env.rho*env.g*V_UWi * r_center[2]

                V_UW += V_UWi
                r_centerV += r_center*V_UWi


            # fully submerged case
            elif rA[2] <= 0 and rB[2] <= 0:

                # displaced volume [m^3] and distance along axis from end A to center of buoyancy of member [m]
                if self.shape=='circular':
                    V_UWi, hc = FrustumVCV(self.d[i-1], self.d[i], self.stations[i]-self.stations[i-1])
                elif self.shape=='rectangular':
                    V_UWi, hc = FrustumVCV(self.sl[i-1], self.sl[i], self.stations[i]-self.stations[i-1])

                r_center = rA + self.q*hc             # absolute coordinates of center of volume of this segment[m]

                # buoyancy force (and moment) vector
                Fvec += translateForce3to6DOF( r_center, np.array([0, 0, env.rho*env.g*V_UWi]) )

                # hydrostatic stiffness matrix (about end A)
                Cmat[3,3] += env.rho*env.g*V_UWi * r_center[2]
                Cmat[4,4] += env.rho*env.g*V_UWi * r_center[2]

                V_UW += V_UWi
                r_centerV += r_center*V_UWi

            else: # if the members are fully above the surface

                pass

        if V_UW > 0:
            r_center = r_centerV/V_UW    # calculate overall member center of buoyancy
        else:
            r_center = np.zeros(3)       # temporary fix for out-of-water members

        return Fvec, Cmat, V_UW, r_center, AWP, IWP, xWP, yWP


    def plot(self, ax):
        '''Draws the member on the passed axes'''

        # --- get coordinates of member edges in member reference frame -------------------

        m = len(self.stations)

        # lists to be filled with coordinates for plotting
        X = []
        Y = []
        Z = []

        if self.shape=="circular":   # circular member cross section
            n = 12                                          # number of sides for a circle
            for i in range(n+1):
                x = np.cos(float(i)/float(n)*2.0*np.pi)    # x coordinates of a unit circle
                y = np.sin(float(i)/float(n)*2.0*np.pi)    # y

                for j in range(m):
                    X.append(0.5*self.d[j]*x)
                    Y.append(0.5*self.d[j]*y)
                    Z.append(self.stations[j])

            coords = np.vstack([X, Y, Z])

        elif self.shape=="rectangular":    # rectangular member cross section
            n=4
            for x,y in zip([1,-1,-1,1,1], [1,1,-1,-1,1]):

                for j in range(m):
                    X.append(0.5*self.sl[j,1]*x)
                    Y.append(0.5*self.sl[j,0]*y)
                    Z.append(self.stations[j])

            coords = np.vstack([X, Y, Z])


        # ----- rotate into global frame ------------------------------
        newcoords = np.matmul(self.R, coords)

        # shift to end A location
        Xs = newcoords[0,:] + self.rA[0]
        Ys = newcoords[1,:] + self.rA[1]
        Zs = newcoords[2,:] + self.rA[2]

        # plot on the provided axes
        linebit = []  # make empty list to hold plotted lines, however many there are
        for i in range(n):  #range(int(len(Xs)/2-1)):
            #linebit.append(ax.plot(Xs[2*i:2*i+2],Ys[2*i:2*i+2],Zs[2*i:2*i+2]            , color='k'))  # side edges
            #linebit.append(ax.plot(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]],Zs[[2*i,2*i+2]]      , color='k'))  # end A edges
            #linebit.append(ax.plot(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]],Zs[[2*i+1,2*i+3]], color='k'))  # end B edges

            linebit.append(ax.plot(Xs[m*i:m*i+m],Ys[m*i:m*i+m],Zs[m*i:m*i+m]            , color='k'))  # side edges

        for j in range(m):
            linebit.append(ax.plot(Xs[j::m], Ys[j::m], Zs[j::m]            , color='k'))  # station rings

        return linebit





def FrustumVCV(dA, dB, H, rtn=0):
    '''returns the volume and center of volume of a frustum, which can be a cylinder (box), cone (pyramid), or anything in between
    Source: https://mathworld.wolfram.com/PyramidalFrustum.html '''

    if np.sum(dA)==0 and np.sum(dB)==0:
        V = 0
        hc = 0
    else:
        if np.isscalar(dA) and np.isscalar(dB): # if the inputs are scalar, meaning that it's just a diameter
            A1 = (np.pi/4)*dA**2
            A2 = (np.pi/4)*dB**2
            Amid = (np.pi/4)*dA*dB
        elif len(dA)==2 and len(dB)==2: # if the inputs are of length 2, meaning if it's two side lengths per node
            A1 = dA[0]*dA[1]
            A2 = dB[0]*dB[1]
            Amid = np.sqrt(A1*A2)
        else:
            raise ValueError('Input types not accepted')

        V = (A1 + A2 + Amid) * H/3
        hc = ((A1 + 2*Amid + 3*A2)/(A1 + Amid + A2)) * H/4

    if rtn==0:
        return V, hc
    elif rtn==1:
        return V
    elif rtn==2:
        return hc


def getVelocity(r, Xi, ws):
    '''Get node complex velocity spectrum based on platform motion's and relative position from PRP'''

    nw = len(ws)

    dr = np.zeros([3,nw], dtype=complex) # node displacement complex amplitudes
    v  = np.zeros([3,nw], dtype=complex) # velocity
    a  = np.zeros([3,nw], dtype=complex) # acceleration


    for i in range(nw):
        dr[:,i] = Xi[:3,i] + SmallRotate(r, Xi[3:,i])
        v[ :,i] = 1j*ws[i]*dr[:,i]
        a[ :,i] = 1j*ws[i]*v[ :,i]


    return dr, v, a # node dispalcement, velocity, acceleration (Each  [3 x nw])


## Get wave velocity and acceleration complex amplitudes based on wave spectrum at a given location
def getWaveKin(zeta0, w, k, h, r, nw, rho=1025.0, g=9.81):

    # inputs: wave elevation fft, wave freqs, wave numbers, depth, point position

    beta = 0  # no wave heading for now

    zeta = np.zeros(nw , dtype=complex ) # local wave elevation
    u  = np.zeros([3,nw], dtype=complex) # local wave kinematics velocity
    ud = np.zeros([3,nw], dtype=complex) # local wave kinematics acceleration
    pDyn = np.zeros(nw , dtype=complex ) # local dynamic pressure

    for i in range(nw):

        # .............................. wave elevation ...............................
        zeta[i] = zeta0[i]* np.exp( -1j*(k[i]*(np.cos(beta)*r[0] + np.sin(beta)*r[1])))  # shift each zetaC to account for location


        # ...................... wave velocities and accelerations ............................
        z = r[2]

        # only process wave kinematics if this node is submerged
        if z < 0:

            # Calculate SINH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/COSH( k*h )
            # given the wave number, k, water depth, h, and elevation z, as inputs.
            if (    k[i]   == 0.0  ):                   # When .TRUE., the shallow water formulation is ill-conditioned; thus, the known value of unity is returned.
                SINHNumOvrSIHNDen = 1.0
                breakpoint()
                COSHNumOvrSIHNDen = 99999.0
                COSHNumOvrCOSHDen = 99999.0   # <<< check
            elif ( k[i]*h >  89.4 ):                # When .TRUE., the shallow water formulation will trigger a floating point overflow error; however, with h > 14.23*wavelength (since k = 2*Pi/wavelength) we can use the numerically-stable deep water formulation instead.
                SINHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrCOSHDen = np.exp( k[i]*z ) + np.exp(-k[i]*(z + 2.0*h))
            else:                                    # 0 < k*h <= 89.4; use the shallow water formulation.
                SINHNumOvrSIHNDen = np.sinh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrSIHNDen = np.cosh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrCOSHDen = np.real( np.cosh(k[i]*(z+h)) )/np.cosh(k[i]*h)   # <<< check

            # Fourier transform of wave velocities
            u[0,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.cos(beta)
            u[1,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.sin(beta)
            u[2,i] = 1j*w[i]* zeta[i]*SINHNumOvrSIHNDen

            # Fourier transform of wave accelerations
            ud[:,i] = 1j*w[i]*u[:,i]

            # Fourier transform of dynamic pressure
            pDyn[i] = rho*g* zeta[i] * COSHNumOvrCOSHDen


    return u, ud, pDyn



# calculate wave number based on wave frequency in rad/s and depth
def waveNumber(omega, h, e=0.001):

    g = 9.81
    # omega - angular frequency of waves

    k = 0                           #initialize  k outside of loop
                                    #error tolerance
    k1 = omega*omega/g                  #deep water approx of wave number
    k2 = omega*omega/(np.tanh(k1*h)*g)      #general formula for wave number
    while np.abs(k2 - k1)/k1 > e:           #repeate until converges
        k1 = k2
        k2 = omega*omega/(np.tanh(k1*h)*g)

    k = k2

    return k


# translate point at location r based on three small angles in th
def SmallRotate(r, th):

    rt = np.zeros(3, dtype=complex)    # translated point

    rt[0] =              th[2]*r[1] - th[1]*r[2]
    rt[0] = th[2]*r[0]              - th[0]*r[2]
    rt[0] = th[1]*r[0] - th[0]*r[1]
    # shousner: @matthall, should these be rt[0], rt[1], rt[2] ?
    return rt


# given a size-3 vector, vec, return the matrix from the multiplication vec * vec.transpose
def VecVecTrans(vec):

    vvt = np.zeros([3,3])

    for i in range(3):
        for j in range(3):
            vvt[i,j]= vec[i]*vec[j]

    return vvt


# produce alternator matrix
def getH(r):

    H = np.zeros([3,3])
    H[0,1] =  r[2];
    H[1,0] = -r[2];
    H[0,2] = -r[1];
    H[2,0] =  r[1];
    H[1,2] =  r[0];
    H[2,1] = -r[0];

    return H



def translateForce3to6DOF(r, Fin):
    '''Takes in a position vector and a force vector (applied at the positon), and calculates
    the resulting 6-DOF force and moment vector.

    :param array r: x,y,z coordinates at which force is acting [m]
    :param array Fin: x,y,z components of force [N]
    :return: the resulting force and moment vector
    :rtype: array
    '''
    Fout = np.zeros(6, dtype=Fin.dtype) # initialize output vector as same dtype as input vector (to support both real and complex inputs)

    Fout[:3] = Fin

    Fout[3:] = np.cross(r, Fin)

    return Fout



# translate mass matrix to make 6DOF mass-inertia matrix
def translateMatrix3to6DOF(r, Min):
    '''Transforms a 3x3 matrix to be about a translated reference point, resulting in a 6x6 matrix.'''

    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    # note that the J term and I terms are zero in this case because the input is just a mass matrix (assumed to be about CG)

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik

    Mout = np.zeros([6,6]) #, dtype=complex)

    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min

    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[:3,3:] = np.matmul(Min, H)
    Mout[3:,:3] = Mout[:3,3:].T

    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]

    Mout[3:,3:] = np.matmul(np.matmul(H,Min), H.T)

    return Mout


def translateMatrix6to6DOF(r, Min):
    '''Transforms a 6x6 matrix to be about a translated reference point.
    r is a vector that goes from where you want the reference point to be
    to where the reference point currently is'''

    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik

    Mout = np.zeros([6,6]) #, dtype=complex)

    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min[:3,:3]

    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[:3,3:] = np.matmul(Min[:3,:3], H) + Min[:3,3:]
    Mout[3:,:3] = Mout[:3,3:].T

    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]
    Mout[3:,3:] = np.matmul(np.matmul(H,Min[:3,:3]), H.T) + np.matmul(Min[3:,:3], H) + np.matmul(H.T, Min[:3,3:]) + Min[3:,3:]

    return Mout


def JONSWAP(ws, Hs, Tp, Gamma=1.0):
    '''Returns the JONSWAP wave spectrum for the given frequencies and parameters.

    Parameters
    ----------
    ws : float | array
        wave frequencies to compute spectrum at (scalar or 1-D list/array) [rad/s]
    Hs : float
        significant wave height of spectrum [m]
    Tp : float
        peak spectral period [s]
    Gamma : float
        wave peak shape parameter []. The default value of 1.0 gives the Pierson-Moskowitz spectrum.

    Returns
    -------
    S : array
        wave power spectral density array corresponding to frequencies in ws [m^2/Hz]


    This function calculates and returns the one-sided power spectral density spectrum
    at the frequency/frequencies (ws) based on the provided significant wave height,
    peak period, and (optionally) shape parameter gamma.

    This formula for the JONSWAP spectrum is adapted from FAST v7 and based
    on what's documented in IEC 61400-3.
    '''

    # handle both scalar and array inputs
    if isinstance(ws, (list, tuple, np.ndarray)):
        ws = np.array(ws)
    else:
        ws = np.array([ws])

    # initialize output array
    S = np.zeros(len(ws))


    # the calculations
    f        = 0.5/np.pi * ws                         # wave frequencies in Hz
    fpOvrf4  = pow((Tp*f), -4.0)                      # a common term, (fp/f)^4 = (Tp*f)^(-4)
    C        = 1.0 - ( 0.287*np.log(Gamma) )          # normalizing factor
    Sigma = 0.07*(f <= 1.0/Tp) + 0.09*(f > 1.0/Tp)    # scaling factor

    Alpha = np.exp( -0.5*((f*Tp - 1.0)/Sigma)**2 )

    return  0.5/np.pi *C* 0.3125*Hs*Hs*fpOvrf4/f *np.exp( -1.25*fpOvrf4 )* Gamma**Alpha


def printMat(mat):
    '''Print a matrix'''
    for i in range(mat.shape[0]):
        print( "\t".join(["{:+8.3e}"]*mat.shape[1]).format( *mat[i,:] ))

def printVec(vec):
    '''Print a vector'''
    print( "\t".join(["{:+8.3e}"]*len(vec)).format( *vec ))


def getFromDict(dict, key, shape=0, dtype=float, default=None):
    '''
    Function to streamline getting values from design dictionary from YAML file, including error checking.

    Parameters
    ----------
    dict : dict
        the dictionary
    key : string
        the key in the dictionary
    shape : list, optional
        The desired shape of the output. If not provided, assuming scalar output. If -1, any input shape is used.
    dtype : type
        Must be a python type than can serve as a function to format the input value to the right type.
    default : number, optional
        The default value to fill in if the item isn't in the dictionary. Otherwise will raise error if the key doesn't exist.
    '''
    # in future could support nested keys   if type(key)==list: ...

    if key in dict:
        val = dict[key]                                      # get the value from the dictionary
        if shape==0:                                         # scalar input expected
            if np.isscalar(val):
                return dtype(val)
            else:
                raise ValueError(f"Value for key '{key}' is expected to be a scalar but instead is: {val}")
        elif shape==-1:                                      # any input shape accepted
            if np.isscalar(val):
                return dtype(val)
            else:
                return np.array(val, dtype=dtype)
        else:
            if np.isscalar(val):                             # if a scalar value is provided and we need to produce an array (of any shape)
                return np.tile(dtype(val), shape)

            elif np.isscalar(shape):                         # if expecting a 1D array
                if len(val) == shape:
                    return np.array([dtype(v) for v in val])
                else:
                    raise ValueError(f"Value for key '{key}' is not the expected size of {shape} and is instead: {val}")

            else:                                            # must be expecting a multi-D array
                vala = np.array(val, dtype=dtype)            # make array

                if vala.shape == shape:                      # if provided with the right shape
                    return vala
                elif len(shape) > 2:
                    raise ValueError("Function getFromDict isn't set up for shapes larger than 2 dimensions")
                elif vala.ndim==1 and len(vala)==shape[1]:   # if we expect an MxN array, and an array of size N is provided, tile it M times
                    return np.tile(vala, [shape[0], 1] )
                else:
                    raise ValueError(f"Value for key '{key}' is not a compatible size for target size of {shape} and is instead: {val}")

    else:
        if default == None:
            raise ValueError(f"Key '{key}' not found in input file...")
        else:
            if shape==0 or shape==-1:
                return default
            else:
                return np.tile(default, shape)


class Model():


    def __init__(self, design, BEM=None, nTurbines=1, w=[], depth=300):
        '''
        Empty frequency domain model initialization function

        design : dict
            Dictionary of all the design info from turbine to platform to moorings
        nTurbines
            could in future be used to set up any number of identical turbines
        '''

        self.fowtList = []
        self.coords = []

        self.nDOF = 0  # number of DOFs in system


    # ----- process turbine information -----------------------------------------
    # No processing actually needed yet - we pass the dictionary directly to RAFT.


    # ----- process platform information ----------------------------------------
    # No processing actually needed yet - we pass the dictionary directly to RAFT.


        # ----- process mooring information ----------------------------------------------

        self.ms = mp.System()

        self.ms.parseYAML(design['mooring'])
        
        
        self.potModMaster = getFromDict(design, 'potModMaster', dtype=int, default=0)
        self.dlsMax = getFromDict(design, 'dlsMax', default=5.0)
        for mi in design['platform']['members']:
            mi['dlsMax'] = self.dlsMax
            if self.potModMaster==1:
                mi['potMod'] = False
            elif self.potModMaster==2:
                mi['potMod'] = True
            
        design['turbine']['tower']['dlsMax'] = self.dlsMax
        
        self.XiStart = getFromDict(design, 'XiStart', default=0.1)
        self.nIter = getFromDict(design, 'nIter', default=15)
        

        self.depth = depth

        # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        if 'yaw stiffness' in design['turbine']:
            self.yawstiff = design['turbine']['yaw stiffness']
        else:
            self.yawstiff = 0

        # analysis frequency array
        if len(w)==0:
            w = np.arange(.05, 3, 0.05)  # angular frequencies tp analyze (rad/s)

        self.w = np.array(w)
        self.nw = len(w)  # number of frequencies

        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # set up the FOWT here  <<< only set for 1 FOWT for now <<<
        self.fowtList.append(FOWT(design, w=self.w, mpb=self.ms.BodyList[0], depth=depth))
        self.coords.append([0.0,0.0])
        self.nDOF += 6

        self.ms.BodyList[0].type = -1  # need to make sure it's set to a coupled type

        self.ms.initialize()  # reinitialize the mooring system to ensure all things are tallied properly etc.

        self.results = {}     # dictionary to hold all results from the model
        

    def addFOWT(self, fowt, xy0=[0,0]):
        '''adds an already set up FOWT to the frequency domain model solver.'''

        self.fowtList.append(fowt)
        self.coords.append(xy0)
        self.nDOF += 6

        # would potentially need to add a mooring system body for it too <<<


    def setEnv(self, Hs=8, Tp=12, spectrum='unit', V=10, beta=0, Fthrust=0):

        self.env = Env()
        self.env.Hs       = Hs
        self.env.Tp       = Tp
        self.env.spectrum = spectrum
        self.env.V        = V
        self.env.beta     = beta
        self.Fthrust      = Fthrust

        for fowt in self.fowtList:
            fowt.setEnv(Hs=Hs, Tp=Tp, V=V, spectrum=spectrum, beta=beta, Fthrust=Fthrust)


    def calcSystemProps(self):
        '''This gets the various static/constant calculations of each FOWT done.'''

        for fowt in self.fowtList:
            fowt.calcBEM()
            fowt.calcStatics()
            fowt.calcHydroConstants()
            #fowt.calcDynamicConstants()

        ## First get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        self.C_moor0 = self.ms.getCoupledStiffness(lines_only=True)                             # this method accounts for eqiuilibrium of free objects in the system
        self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)

        self.results['properties'] = {}   # signal this data is available by adding a section to the results dictionary
        
        
    
    def calcMooringAndOffsets(self):
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.  This will ultimately become a method for solving mean operating point.
        '''


        # Now find static equilibrium offsets of platform and get mooring properties about that point
        # (This assumes some loads have been applied)
        #self.ms.display=2

        self.ms.solveEquilibrium3(DOFtype="both", rmsTol=1.0E-5)     # get the system to its equilibrium
        
        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]

        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6
        
        print("Equilibrium'3' platform positions/rotations:")
        printVec(self.ms.BodyList[0].r6)

        r6eq = self.ms.BodyList[0].r6

        #self.ms.plot()

        print("Surge: {:.2f}".format(r6eq[0]))
        print("Pitch: {:.2f}".format(r6eq[4]*180/np.pi))

        C_moor = self.ms.getCoupledStiffness(lines_only=True)
        F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)    # get net forces and moments from mooring lines on Body

        # manually add yaw spring stiffness as compensation until bridle (crow foot) configuration is added
        C_moor[5,5] += self.yawstiff

        self.C_moor = C_moor
        self.F_moor = F_moor

        # store results
        self.results['means'] = {}   # signal this data is available by adding a section to the results dictionary
        self.results['means']['platform offset'  ] = r6eq
        self.results['means']['mooring force'    ] = F_moor
        #self.results['means']['fairlead tensions'] = ... # <<<
        
    
    

    def solveEigen(self):
        '''finds natural frequencies of system'''


        # total system coefficient arrays
        M_tot = np.zeros([self.nDOF,self.nDOF])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
        C_tot = np.zeros([self.nDOF,self.nDOF])       # total stiffness matrix [N/m, N, N-m]


        # add in mooring stiffness from MoorPy system
        C_tot = np.array(self.C_moor0)

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]

        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6

        # add fowt's terms to system matrices (BEM arrays are not yet included here)
        M_tot[i1:i2] += fowt.M_struc + fowt.A_hydro_morison   # mass
        C_tot[i1:i2] += fowt.C_struc + fowt.C_hydro           # stiffness

        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)
        eigenvals, eigenvectors = np.linalg.eig(np.matmul(np.linalg.inv(M_tot), C_tot))   # <<< need to sort this out so it gives desired modes, some are currently a bit messy

        # sort to normal DOF order based on which DOF is largest in each eigenvector
        ind_list = []
        for i in range(5,-1, -1):
            vec = np.abs(eigenvectors[i,:])  # look at each row (DOF) at a time (use reverse order to pick out rotational DOFs first)

            for j in range(6):               # now do another loop in case the index was claimed previously

                ind = np.argmax(vec)         # find the index of the vector with the largest value of the current DOF

                if ind in ind_list:          # if a previous vector claimed this DOF, set it to zero in this vector so that we look at the other vectors
                    vec[ind] = 0.0
                else:
                    ind_list.append(ind)     # if it hasn't been claimed before, assign this vector to the DOF
                    break

        ind_list.reverse()   # reverse the index list since we made it in reverse order

        fns = np.sqrt(eigenvals[ind_list])/2.0/np.pi   # apply sorting to eigenvalues and convert to natural frequency in Hz
        modes = eigenvectors[:,ind_list]               # apply sorting to eigenvectors

        print("natural frequencies from eigen values")
        printVec(fns)
        print("mode shapes from eigen values")
        printMat(modes)


        # alternative attempt to calculate natural frequencies based on diagonal entries (and taking pitch and roll about CG)
        if C_tot[0,0] == 0.0:
            zMoorx = 0.0
        else:
            zMoorx = C_tot[0,4]/C_tot[0,0]  # effective z elevation of mooring system reaction forces in x and y directions

        if C_tot[1,1] == 0.0:
            zMoory = 0.0
        else:
            zMoory = C_tot[1,3]/C_tot[1,1]

        zCG  = fowt.rCG_TOT[2]                    # center of mass in z
        zCMx = M_tot[0,4]/M_tot[0,0]              # effective z elevation of center of mass and added mass in x and y directions
        zCMy = M_tot[1,3]/M_tot[1,1]

        print("natural frequencies with added mass")
        fn = np.zeros(6)
        fn[0] = np.sqrt( C_tot[0,0] / M_tot[0,0] )/ 2.0/np.pi
        fn[1] = np.sqrt( C_tot[1,1] / M_tot[1,1] )/ 2.0/np.pi
        fn[2] = np.sqrt( C_tot[2,2] / M_tot[2,2] )/ 2.0/np.pi
        fn[5] = np.sqrt( C_tot[5,5] / M_tot[5,5] )/ 2.0/np.pi
        fn[3] = np.sqrt( (C_tot[3,3] + C_tot[1,1]*((zCMy-zMoory)**2 - zMoory**2) ) / (M_tot[3,3] - M_tot[1,1]*zCMy**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        fn[4] = np.sqrt( (C_tot[4,4] + C_tot[0,0]*((zCMx-zMoorx)**2 - zMoorx**2) ) / (M_tot[4,4] - M_tot[0,0]*zCMx**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        # note that the above lines use off-diagonal term rather than parallel axis theorem since rotation will not be exactly at CG due to effect of added mass
        printVec(fn)

                
        # store results
        self.results['eigen'] = {}   # signal this data is available by adding a section to the results dictionary
        self.results['eigen']['frequencies'] = fns
        self.results['eigen']['modes'      ] = modes
  

    def solveDynamics(self, tol=0.01, conv_plot=1, RAO_plot=1):
        '''After all constant parts have been computed, call this to iterate through remaining terms
        until convergence on dynamic response. Note that steady/mean quantities are excluded here.

        nIter = 2  # maximum number of iterations to allow
        '''
        
        nIter = int(self.nIter) + 1         # maybe think of a better name for the first nIter
        XiStart = self.XiStart
        
        # total system complex response amplitudes (this gets updated each iteration)
        XiLast = np.zeros([self.nDOF,self.nw], dtype=complex) + XiStart    # displacement and rotation complex amplitudes [m, rad]
        
        if conv_plot:
            fig, ax = plt.subplots(3,1,sharex=True)
            c = np.arange(nIter+1)      # adding 1 again here so that there are no RuntimeErrors
            c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))      # set up colormap to use to plot successive iteration results

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        i1 = 0                                                # range of DOFs for the current turbine
        i2 = 6

        # sum up all linear (non-varying) matrices up front
        M_lin = fowt.M_struc[:,:,None] + fowt.A_BEM + fowt.A_hydro_morison[:,:,None] # mass
        B_lin = fowt.B_struc[:,:,None] + fowt.B_BEM                                  # damping
        C_lin = fowt.C_struc   + self.C_moor        + fowt.C_hydro                   # stiffness
        F_lin =                          fowt.F_BEM + fowt.F_hydro_iner              # excitation
        
        
        # start fixed point iteration loop for dynamics   <<< would a secant method solve be possible/better? <<<
        for iiter in range(nIter):
            
            # ::: re-zero some things that will be added to :::

            # total system coefficient arrays
            M_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
            B_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total damping matrix [N-s/m, N-s, N-s-m]
            C_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total stiffness matrix [N/m, N, N-m]
            F_tot = np.zeros([self.nDOF,self.nw], dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

            Z  = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=complex)  # total system impedance matrix


            # ::: a loop could be added here for an array :::
            fowt = self.fowtList[0]
            i1 = 0                                                # range of DOFs for the current turbine
            i2 = 6

            # get linearized terms for the current turbine given latest amplitudes
            B_linearized, F_linearized = fowt.calcLinearizedTerms(XiLast)

            # calculate the response based on the latest linearized terms
            Xi = np.zeros([self.nDOF,self.nw], dtype=complex)     # displacement and rotation complex amplitudes [m, rad]

            # add fowt's terms to system matrices (BEM arrays are not yet included here)
            M_tot[:,:,:] = M_lin
            B_tot[:,:,:] = B_lin           + B_linearized[:,:,None]
            C_tot[:,:,:] = C_lin[:,:,None]
            F_tot[:  ,:] = F_lin           + F_linearized


            for ii in range(self.nw):
                # form impedance matrix
                Z[:,:,ii] = -self.w[ii]**2 * M_tot[:,:,ii] + 1j*self.w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii]
                
                # solve response (complex amplitude)
                Xi[:,ii] = np.matmul(np.linalg.inv(Z[:,:,ii]),  F_tot[:,ii] )


            if conv_plot:
                # Convergence Plotting
                # plots of surge response at each iteration for observing convergence
                ax[0].plot(self.w, np.abs(Xi[0,:]) , color=c[iiter], label=f"iteration {iiter}")
                ax[1].plot(self.w, np.real(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
                ax[2].plot(self.w, np.imag(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
    
            # check for convergence
            tolCheck = np.abs(Xi - XiLast) / ((np.abs(Xi)+tol))
            if (tolCheck < tol).all():
                print(f" Iteration {iiter}, converged, with largest tolCheck of {np.max(tolCheck)} < {tol}")
                break
            else:
                XiLast = 0.2*XiLast + 0.8*Xi    # use a mix of the old and new response amplitudes to use for the next iteration
                                                # (uses hard-coded successive under relaxation for now)
                print(f" Iteration {iiter}, still going since largest tolCheck is {np.max(tolCheck)} >= {tol}")
    
            if iiter == nIter-1:
                print("WARNING - solveDynamics iteration did not converge to the tolerance.")
        
        if conv_plot:
            # labels for convergence plots
            ax[1].legend()
            ax[0].set_ylabel("response magnitude")
            ax[1].set_ylabel("response, real")
            ax[2].set_ylabel("response, imag")
            ax[2].set_xlabel("frequency (rad/s)")
            fig.suptitle("Response convergence")


        # ------------------------------ preliminary plotting of response ---------------------------------
        
        if RAO_plot:
            # RAO plotting
            fig, ax = plt.subplots(3,1, sharex=True)
    
            fowt = self.fowtList[0]
    
            ax[0].plot(self.w, np.abs(Xi[0,:])          , 'b', label="surge")
            ax[0].plot(self.w, np.abs(Xi[1,:])          , 'g', label="sway")
            ax[0].plot(self.w, np.abs(Xi[2,:])          , 'r', label="heave")
            ax[1].plot(self.w, np.abs(Xi[3,:])*180/np.pi, 'b', label="roll")
            ax[1].plot(self.w, np.abs(Xi[4,:])*180/np.pi, 'g', label="pitch")
            ax[1].plot(self.w, np.abs(Xi[5,:])*180/np.pi, 'r', label="yaw")
            ax[2].plot(self.w, fowt.zeta,                 'k', label="wave amplitude (m)")
    
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
    
            #ax[0].set_ylim([0, 1e6])
            #ax[1].set_ylim([0, 1e9])
    
            ax[0].set_ylabel("response magnitude (m)")
            ax[1].set_ylabel("response magnitude (deg)")
            ax[2].set_ylabel("wave amplitude (m)")
            ax[2].set_xlabel("frequency (rad/s)")

        
        self.Xi = Xi

        self.results['response'] = {}   # signal this data is available by adding a section to the results dictionary

        return Xi  # currently returning the response rather than saving in the model object



    def calcOutputs(self):
        '''This is where various output quantities of interest are calculated based on the already-solved system response.'''
        
        fowt = self.fowtList[0]   # just using a single turbine for now
        
        
        # ----- system properties outputs -----------------------------
        # all values about platform reference point (z=0) unless otherwise noted
        
        if 'properties' in self.results:
        
            self.results['properties']['tower mass'] = fowt.mtower
            self.results['properties']['tower CG'] = fowt.rCG_tow
            self.results['properties']['substructure mass'] = fowt.msubstruc
            self.results['properties']['substructure CG'] = fowt.rCG_sub
            self.results['properties']['shell mass'] = fowt.mshell
            self.results['properties']['ballast mass'] = fowt.mballast
            self.results['properties']['ballast densities'] = fowt.pb
            self.results['properties']['total mass'] = fowt.M_struc[0,0]
            self.results['properties']['total CG'] = fowt.rCG_TOT
            #self.results['properties']['roll inertia at subCG'] = fowt.I44
            #self.results['properties']['pitch inertia at subCG'] = fowt.I55
            #self.results['properties']['yaw inertia at subCG'] = fowt.I66
            self.results['properties']['roll inertia at subCG'] = fowt.M_struc_subCM[3,3]
            self.results['properties']['pitch inertia at subCG'] = fowt.M_struc_subCM[4,4]
            self.results['properties']['yaw inertia at subCG'] = fowt.M_struc_subCM[5,5]
            
            self.results['properties']['Buoyancy (pgV)'] = fowt.env.rho*fowt.env.g*fowt.V
            self.results['properties']['Center of Buoyancy'] = fowt.rCB
            self.results['properties']['C stiffness matrix'] = fowt.C_hydro
            
            self.results['properties']['F_lines0'] = self.F_moor0
            self.results['properties']['C_lines0'] = self.C_moor0
                    
            # 6DOF matrices for the support structure (everything but turbine) including mass, hydrostatics, and mooring reactions
            self.results['properties']['M support structure'] = fowt.M_struc_subCM                          # mass matrix
            self.results['properties']['A support structure'] = fowt.A_hydro_morison + fowt.A_BEM[:,:,-1]   # hydrodynamic added mass (currently using highest frequency of BEM added mass)
            self.results['properties']['C support structure'] = fowt.C_struc_sub + fowt.C_hydro + self.C_moor0  # stiffness

        
        
        # ----- response outputs (always in standard units) ---------------------------------------
        
        if 'response' in self.results:
            
            RAOmag      = abs(self.Xi          /fowt.zeta)  # magnitudes of motion RAO
            
            self.results['response']['frequencies'] = self.w/2/np.pi         # Hz
            self.results['response']['wave elevation'] = fowt.zeta
            self.results['response']['Xi'         ] = self.Xi
            self.results['response']['surge RAO'  ] = RAOmag[0,:]
            self.results['response'][ 'sway RAO'  ] = RAOmag[1,:]
            self.results['response']['heave RAO'  ] = RAOmag[2,:]
            self.results['response']['pitch RAO'  ] = RAOmag[3,:]
            self.results['response'][ 'roll RAO'  ] = RAOmag[4,:]
            self.results['response'][  'yaw RAO'  ] = RAOmag[5,:]
            
            # save dynamic derived quantities
            #self.results['response']['mooring tensions'] = ...
            self.results['response']['nacelle acceleration'] = self.w**2 * (self.Xi[0] + self.Xi[4]*fowt.hHub)
        
    

        '''
         # ---------- mooring line fairlead tension RAOs and constraint implementation ----------


         for il=1:Platf.Nlines

              #aNacRAO{imeto} = -(w').^2 .* (X{imeto}(:,1) + hNac*X{imeto}(:,5));      # Nacelle Accel RAO
                #aNac2(imeto) = sum( abs(aNacRAO{imeto}).^2.*S(:,imeto) ) *(w(2)-w(1));     # RMS Nacelle Accel

            TfairRAO{imeto}(il,:) = C_lf(il,:,imeto)*rao{imeto}(:,:)';  # get fairlead tension RAO for each line (multiply by dofs)
              #RMSTfair{imeto}(il) = sqrt( sum( (abs(TfairRAO{imeto}(il,:))).^2) / length(w) );
              #figure
            #plot(w,abs(TfairRAO{imeto}(il,:)))
              #d=TfairRAO{imeto}(il,:)
              RMSTfair{imeto}(il) = sqrt( sum( (abs(TfairRAO{imeto}(il,:)).^2).*S(:,imeto)') *(w(2)-w(1)) );
              #RMSTfair
              #sumpart = sum( (abs(TfairRAO{imeto}(il,:)).^2).*S(:,imeto)')
              #dw=(w(2)-w(1))
         end

         [Tfair, il] = min( T_lf(:,imeto) );
         if Tfair - 3*RMSTfair{imeto}(il) < 0 && Xm < 1  # taut lines only
              disp([' REJECTING (mooring line goes slack)'])
              fitness = -1;
              return;  # constraint for slack line!!!
         end
         if grads
              disp(['mooring slackness: ' num2str(Tfair - 3*RMSTfair{imeto}(il))])
         end

         # ----------- dynamic pitch constraint ----------------------
         #disp('checking dynamic pitch');
         RMSpitch(imeto) = sqrt( sum( ((abs(rao{imeto}(:,5))).^2).*S(:,imeto) ) *(w(2)-w(1)) ); # fixed April 9th :(
         RMSpitchdeg = RMSpitch(imeto)*60/pi;
         if (Platf.spitch + RMSpitch(imeto))*180/pi > 10
              disp([' REJECTING (static + RMS dynamic pitch > 10)'])
              fitness = -1;
              return;
         end
         if grads
              disp(['dynamic pitch: ' num2str((Platf.spitch + RMSpitch(imeto))*180/pi)])
         end

         #figure(1)
         #plot(w,S(:,imeto))
         #hold on

         #figure()
         #plot(2*pi./w,abs(Xi{imeto}(:,5)))
         #ylabel('pitch response'); xlabel('T (s)')

         RMSsurge(imeto) = sqrt( sum( ((abs(rao{imeto}(:,1))).^2).*S(:,imeto) ) *(w(2)-w(1)) );
         RMSheave(imeto) = sqrt( sum( ((abs(rao{imeto}(:,3))).^2).*S(:,imeto) ) *(w(2)-w(1)) );
        '''

        
        
        return self.results
        

    def plot(self, hideGrid=False):
        '''plots the whole model, including FOWTs and mooring system...'''

        # for now, start the plot via the mooring system, since MoorPy doesn't yet know how to draw on other codes' plots
        self.ms.BodyList[0].setPosition(np.zeros(6))
        self.ms.initialize()
        fig, ax = self.ms.plot()
        #fig = plt.figure(figsize=(20/2.54,12/2.54))
        #ax = Axes3D(fig)

        

        # plot each FOWT
        for fowt in self.fowtList:
            fowt.plot(ax)
            
        if hideGrid:       
            ax.set_xticks([])    # Hide axes ticks
            ax.set_yticks([])
            ax.set_zticks([])     
            ax.grid(False)       # Hide grid lines
            plt.grid(b=None)
            ax.axis('off')
            plt.box(False)


class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, design, w=[], mpb=None, depth=600):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.

        Parameters
        ----------
        design : dict
            Dictionary of the design...
        w
            Array of frequencies to be used in analysis
        '''


        # basic setup
        self.nDOF = 6

        if len(w)==0:
            w = np.arange(.01, 3, 0.01)                              # angular frequencies tp analyze (rad/s)

        self.w = np.array(w)
        self.nw = len(w)                                             # number of frequencies
        self.k = np.zeros(self.nw)                                   # wave number

        self.depth = depth
             
        
        # member-based platform description
        self.memberList = []                                         # list of member objects

        for mi in design['platform']['members']:
            headings = getFromDict(mi, 'heading', shape=-1, default=0.)
            if np.isscalar(headings):
                mi['heading'] = headings
                self.memberList.append(Member(mi, self.nw))
            else:
                for heading in headings:
                    mi['heading'] = heading
                    self.memberList.append(Member(mi, self.nw))
                mi['heading'] = headings # set the headings dict value back to the yaml headings value, instead of the last one used

        self.memberList.append(Member(design['turbine']['tower'], self.nw))

        # mooring system connection
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine


        # turbine RNA description
        self.mRNA    = design['turbine']['mRNA']
        self.IxRNA   = design['turbine']['IxRNA']
        self.IrRNA   = design['turbine']['IrRNA']
        self.xCG_RNA = design['turbine']['xCG_RNA']
        self.hHub    = design['turbine']['hHub']


        # initialize BEM arrays, whether or not a BEM sovler is used
        self.A_BEM = np.zeros([6,6,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([6,6,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        self.F_BEM = np.zeros([6,  self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]



    def setEnv(self, Hs=8, Tp=12, spectrum='unit', V=10, beta=0, Fthrust=0):
        '''For now, this is where the environmental conditions acting on the FOWT are set.'''

        # ------- Wind conditions
        #Fthrust = 800e3  # peak thrust force, [N]
        #Mthrust = self.hHub*Fthrust  # overturning moment from turbine thrust force [N-m]


        self.env = Env()
        self.env.Hs       = Hs
        self.env.Tp       = Tp
        self.env.spectrum = spectrum
        self.env.V        = V
        self.env.beta     = beta

        # calculate wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
            
        # make wave spectrum
        if spectrum == 'unit':
            self.zeta = np.tile(1, self.nw)
        else:
            S = JONSWAP(self.w, Hs, Tp)        
            self.zeta = np.sqrt(S)    # wave elevation amplitudes (these are easiest to use)

        #Fthrust = 0
        #Fthrust = 800.0e3            # peak thrust force, [N]
        #Mthrust = self.hHub*Fthrust  # overturning moment from turbine thrust force [N-m]

        # add thrust force and moment to mooring system body
        self.body.f6Ext = Fthrust*np.array([np.cos(beta),np.sin(beta),0,-self.hHub*np.sin(beta), self.hHub*np.cos(beta), 0])



    def calcStatics(self):
        '''Fills in the static quantities of the FOWT and its matrices.
        Also adds some dynamic parameters that are constant, e.g. BEM coefficients and steady thrust loads.'''

        rho = self.env.rho
        g   = self.env.g

        # structure-related arrays
        self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]
        self.C_struc_sub = np.zeros([6,6])            # substructure effective stiffness matrix [N/m, N, N-m]

        # hydrostatic arrays
        self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]


        # --------------- add in linear hydrodynamic coefficients here if applicable --------------------
        #[as in load them] <<<<<<<<<<<<<<<<<<<<<

        # --------------- Get general geometry properties including hydrostatics ------------------------

        # initialize some variables for running totals
        VTOT = 0.                   # Total underwater volume of all members combined
        mTOT = 0.                   # Total mass of all members [kg]
        AWP_TOT = 0.                # Total waterplane area of all members [m^2]
        IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]
        IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]
        Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
        Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
        Sum_M_center = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)

        self.msubstruc = 0          # total mass of just the members that make up the substructure [kg]
        self.M_struc_subPRP = np.zeros([6,6])  # total mass matrix of just the substructure about the PRP
        msubstruc_sum = 0           # product of each substructure member's mass and CG, to be used to find the total substructure CG [kg-m]
        self.mshell = 0             # total mass of the shells/steel of the members in the substructure [kg]
        mballast = []               # list to store the mass of the ballast in each of the substructure members [kg]
        pballast = []               # list to store the density of ballast in each of the substructure members [kg]
        '''
        I44list = []                # list to store the I44 MoI about the PRP of each substructure member
        I55list = []                # list to store the I55 MoI about the PRP of each substructure member
        I66list = []                # list to store the I66 MoI about the PRP of each substructure member
        masslist = []               # list to store the mass of each substructure member
        '''
        # loop through each member
        for mem in self.memberList:

            # calculate member's orientation information (needed for later steps)
            mem.calcOrientation()

            # ---------------------- get member's mass and inertia properties ------------------------------
            mass, center, mshell, mfill, pfill = mem.getInertia() # calls the getInertia method to calcaulte values


            # Calculate the mass matrix of the FOWT about the PRP
            self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*mass]) )  # weight vector
            self.M_struc += mem.M_struc     # mass/inertia matrix about the PRP

            Sum_M_center += center*mass     # product sum of the mass and center of mass to find the total center of mass [kg-m]


            # Tower calculations
            if mem.type <= 1:   # <<<<<<<<<<<< maybe find a better way to do the if condition
                self.mtower = mass                  # mass of the tower [kg]
                self.rCG_tow = center               # center of mass of the tower from the PRP [m]
            # Substructure calculations
            if mem.type > 1:
                self.msubstruc += mass              # mass of the substructure
                self.M_struc_subPRP += mem.M_struc     # mass matrix of the substructure about the PRP
                msubstruc_sum += center*mass        # product sum of the substructure members and their centers of mass [kg-m]
                self.mshell += mshell               # mass of the substructure shell material [kg]
                mballast.extend(mfill)              # list of ballast masses in each substructure member (list of lists) [kg]
                pballast.extend(pfill)              # list of ballast densities in each substructure member (list of lists) [kg/m^3]
                '''
                # Store substructure moment of inertia terms
                I44list.append(mem.M_struc[3,3])
                I55list.append(mem.M_struc[4,4])
                I66list.append(mem.M_struc[5,5])
                masslist.append(mass)
                '''

            # -------------------- get each member's buoyancy/hydrostatic properties -----------------------

            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(self.env)  # call to Member method for hydrostatic calculations

            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices <<<<< needs updating (already about PRP)
            self.W_hydro += Fvec # translateForce3to6DOF( mem.rA, np.array([0,0, Fz]) )  # weight vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(mem.rA, Cmat)                       # hydrostatic stiffness matrix

            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP


        # ------------------------- include RNA properties -----------------------------

        # Here we could initialize first versions of the structure matrix components.
        # These might be iterated on later to deal with mean- or amplitude-dependent terms.
        #self.M_struc += structural.M_lin(q0,      self.turbineParams)        # Linear Mass Matrix
        #self.B_struc += structural.C_lin(q0, qd0, self.turbineParams, u0)    # Linear Damping Matrix
        #self.C_struc += structural.K_lin(q0, qd0, self.turbineParams, u0)    # Linear Stifness Matrix
        #self.W_struc += structural.B_lin(q0, qd0, self.turbineParams, u0)    # Linear RHS

        # below are temporary placeholders
        # for now, turbine RNA is specified by some simple lumped properties
        Mmat = np.diag([self.mRNA, self.mRNA, self.mRNA, self.IxRNA, self.IrRNA, self.IrRNA])            # create mass/inertia matrix
        center = np.array([self.xCG_RNA, 0, self.hHub])                                 # RNA center of mass location

        # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
        self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*self.mRNA]) )  # weight vector
        self.M_struc += translateMatrix6to6DOF(center, Mmat)                       # mass/inertia matrix
        Sum_M_center += center*self.mRNA


        # ----------- process inertia-related totals ----------------

        mTOT = self.M_struc[0,0]                        # total mass of all the members
        rCG_TOT = Sum_M_center/mTOT                     # total CG of all the members
        self.rCG_TOT = rCG_TOT

        self.rCG_sub = msubstruc_sum/self.msubstruc     # solve for just the substructure mass and CG
        
        self.M_struc_subCM = translateMatrix6to6DOF(-self.rCG_sub, self.M_struc_subPRP) # the mass matrix of the substructure about the substruc's CM
        # need to make rCG_sub negative here because tM6to6DOF takes a vector that goes from where you want the ref point to be (CM) to the currently ref point (PRP)
        
        '''
        self.I44 = 0        # moment of inertia in roll due to roll of the substructure about the substruc's CG [kg-m^2]
        self.I44B = 0       # moment of inertia in roll due to roll of the substructure about the PRP [kg-m^2]
        self.I55 = 0        # moment of inertia in pitch due to pitch of the substructure about the substruc's CG [kg-m^2]
        self.I55B = 0       # moment of inertia in pitch due to pitch of the substructure about the PRP [kg-m^2]
        self.I66 = 0        # moment of inertia in yaw due to yaw of the substructure about the substruc's centerline [kg-m^2]

        # Use the parallel axis theorem to move each substructure's MoI to the substructure's CG
        x = np.linalg.norm([self.rCG_sub[1],self.rCG_sub[2]])   # the normalized distance between the x and x' axes
        y = np.linalg.norm([self.rCG_sub[0],self.rCG_sub[2]])   # the normalized distance between the y and y' axes
        z = np.linalg.norm([self.rCG_sub[0],self.rCG_sub[1]])   # the normalized distance between the z and z' axes
        for i in range(len(I44list)):
            self.I44 += I44list[i] - masslist[i]*x**2
            self.I44B += I44list[i]
            self.I55 += I55list[i] - masslist[i]*y**2
            self.I55B += I55list[i]
            self.I66 += I66list[i] - masslist[i]*z**2
        '''
        
        # Solve for the total mass of each type of ballast in the substructure
        self.pb = []                                                # empty list to store the unique ballast densities
        for i in range(len(pballast)):
            if pballast[i] != 0:                                    # if the value in pballast is not zero
                if self.pb.count(pballast[i]) == 0:                 # and if that value is not already in pb
                    self.pb.append(pballast[i])                     # store that ballast density value

        self.mballast = np.zeros(len(self.pb))                      # make an empty mballast list with len=len(pb)
        for i in range(len(self.pb)):                               # for each ballast density
            for j in range(len(mballast)):                          # loop through each ballast mass
                if np.float(pballast[j]) == np.float(self.pb[i]):   # but only if the index of the ballast mass (density) matches the value of pb
                    self.mballast[i] += mballast[j]                 # add that ballast mass to the correct index of mballast



        # ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------

        self.V = VTOT                                   # save the total underwater volume
        rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform
        self.rCB = rCB_TOT

        if VTOT==0: # if you're only working with members above the platform, like modeling the wind turbine
            zMeta = 0
        else:
            zMeta   = rCB_TOT[2] + IWPx_TOT/VTOT  # add center of buoyancy and BM=I/v to get z elevation of metecenter [m] (have to pick one direction for IWP)

        self.C_struc[3,3] = -mTOT*g*rCG_TOT[2]
        self.C_struc[4,4] = -mTOT*g*rCG_TOT[2]
        
        self.C_struc_sub[3,3] = -self.msubstruc*g*self.rCG_sub[2]
        self.C_struc_sub[4,4] = -self.msubstruc*g*self.rCG_sub[2]

        # add relevant properties to this turbine's MoorPy Body
        self.body.m = mTOT
        self.body.v = VTOT
        self.body.rCG = rCG_TOT
        self.body.AWP = AWP_TOT
        self.body.rM = np.array([0,0,zMeta])
        # is there any risk of additional moments due to offset CB since MoorPy assumes CB at ref point? <<<



    def calcBEM(self):
        '''This generates a mesh for the platform and runs a BEM analysis on it.
        The mesh is only for non-interesecting members flagged with potMod=1.'''
        
        rho = self.env.rho
        g   = self.env.g

        # desired panel size (longitudinal and azimuthal)
        dz = 3
        da = 2
        
        # go through members to be modeled with BEM and calculated their nodes and panels lists
        nodes = []
        panels = []
        
        vertices = np.zeros([0,3])  # for GDF output

        for mem in self.memberList:
            
            if mem.potMod==True:
                pnl.meshMember(mem.stations, mem.d, mem.rA, mem.rB,
                        dz_max=dz, da_max=da, savedNodes=nodes, savedPanels=panels)
                        
                # for GDF output
                vertices_i = pnl.meshMemberForGDF(mem.stations, mem.d, mem.rA, mem.rB, dz_max=dz, da_max=da)
                vertices = np.vstack([vertices, vertices_i])              # append the member's vertices to the master list


        # only try to save a mesh and run HAMS if some members DO have potMod=True
        if len(panels) > 0:

            meshDir = 'BEM'
            
            pnl.writeMesh(nodes, panels, oDir=osp.join(meshDir,'Input')) # generate a mesh file in the HAMS .pnl format
            
            pnl.writeMeshToGDF(vertices)                                # also a GDF for visualization
            
            # TODO: maybe create a 'HAMS Project' class:
            #     - methods:
            #         - create HAMS project structure
            #         - write HAMS input files
            #         - call HAMS.exe
            #     - attributes:
            #         - addedMass, damping, fEx coefficients
            ph.create_hams_dirs(meshDir)
            
            ph.write_hydrostatic_file(meshDir)
            
            ph.write_control_file(meshDir, waterDepth=self.depth,
                                  numFreqs=-len(self.w), minFreq=self.w[0], dFreq=np.diff(self.w[:2])[0])
            
            ph.run_hams(meshDir) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            data1 = osp.join(meshDir, f'Output/Wamit_format/Buoy.1')
            data3 = osp.join(meshDir, f'Output/Wamit_format/Buoy.3')
            
            #raftDir = osp.dirname(__file__)
            #addedMass, damping = ph.read_wamit1(osp.join(raftDir, data1))
            addedMass, damping, w_HAMS = ph.read_wamit1B(data1)
            fExMod, fExPhase, fExReal, fExImag = ph.read_wamit3(data3)
                       
            #addedMass, damping = ph.read_wamit1(data1)         # original
            #addedMass, damping = ph.read_wamit1('C:\\Code\\RAFT\\raft\\data\\cylinder\\Output\\Wamit_format\\Buoy.1')
            #addedMass, damping = ph.read_wamit1('C:\\Code\\RAFT\\raft\\BEM\\Output\\Wamit_format\\Buoy.1')
            
            #fExMod, fExPhase, fExReal, fExImag = ph.read_wamit3(data3)         # original
            #fExMod, fExPhase, fExReal, fExImag = ph.read_wamit3('C:\\Code\\RAFT\\raft\\data\\cylinder\\Output\\Wamit_format\\Buoy.3')
            #fExMod, fExPhase, fExReal, fExImag = ph.read_wamit3('C:\\Code\\RAFT\\raft\\BEM\\Output\\Wamit_format\\Buoy.3')
            
            # copy results over to the FOWT's coefficient arrays
            self.A_BEM = self.env.rho * addedMass
            self.B_BEM = self.env.rho * damping
            self.X_BEM = self.env.rho * self.env.g * (fExReal + 1j*fExImag)  # linear wave excitation coefficients
            self.F_BEM = self.X_BEM * self.zeta     # wave excitation force
            self.w_BEM = w_HAMS

            # >>> do we want to seperate out infinite-frequency added mass? <<<
            
            

    def calcHydroConstants(self):
        '''This computes the linear strip-theory-hydrodynamics terms.'''

        rho = self.env.rho
        g   = self.env.g

        # --------------------- get constant hydrodynamic values along each member -----------------------------

        self.A_hydro_morison = np.zeros([6,6])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]
        self.F_hydro_iner    = np.zeros([6,self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]

        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # get wave kinematics spectra given a certain wave spectrum and location
                    mem.u[il,:,:], mem.ud[il,:,:], mem.pDyn[il,:] = getWaveKin(self.zeta, self.w, self.k, self.depth, mem.r[il,:], self.nw)

                    # only compute inertial loads and added mass for members that aren't modeled with potential flow
                    if mem.potMod==False:

                        # interpolate coefficients for the current strip
                        Ca_q   = np.interp( mem.ls[il], mem.stations, mem.Ca_q  )
                        Ca_p1  = np.interp( mem.ls[il], mem.stations, mem.Ca_p1 )
                        Ca_p2  = np.interp( mem.ls[il], mem.stations, mem.Ca_p2 )
                        Ca_End = np.interp( mem.ls[il], mem.stations, mem.Ca_End)


                        # ----- compute side effects ---------------------------------------------------------

                        if circ:
                            v_i = 0.25*np.pi*mem.ds[il]**2*mem.dls[il]
                        else:
                            v_i = mem.ds[il,0]*mem.ds[il,1]*mem.dls[il]  # member volume assigned to this node
                            
                        if mem.r[il,2] + 0.5*mem.dls[il] > 0:    # if member extends out of water              # <<< may want a better appraoch for this...
                            v_i = v_i * (0.5*mem.dls[il] - mem.r[il,2]) / mem.dls[il]  # scale volume by the portion that is under water
                            
                         
                        # added mass
                        Amat = rho*v_i *( Ca_q*mem.qMat + Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat )  # local added mass matrix

                        self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:], Amat)    # add to global added mass matrix for Morison members
                        
                        # inertial excitation - Froude-Krylov  (axial term explicitly excluded here - we aren't dealing with chains)
                        Imat = rho*v_i *(  (1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat ) # local inertial excitation matrix
                        #Imat = rho*v_i *( (1.+Ca_q)*mem.qMat + (1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat ) # local inertial excitation matrix

                        for i in range(self.nw):                                             # for each wave frequency...

                            mem.F_exc_iner[il,:,i] = np.matmul(Imat, mem.ud[il,:,i])         # add to global excitation vector (frequency dependent)

                            self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], mem.F_exc_iner[il,:,i])  # add to global excitation vector (frequency dependent)


                        # ----- add axial/end effects for added mass, and excitation including dynamic pressure ------
                        # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                        if circ:
                            v_i = np.pi/12.0 * abs((mem.ds[il]+mem.drs[il])**3 - (mem.ds[il]-mem.drs[il])**3)  # volume assigned to this end surface
                            a_i = np.pi*mem.ds[il] * mem.drs[il]   # signed end area (positive facing down) = mean diameter of strip * radius change of strip
                        else:
                            v_i = np.pi/12.0 * ((np.mean(mem.ds[il]+mem.drs[il]))**3 - (np.mean(mem.ds[il]-mem.drs[il]))**3)    # so far just using sphere eqn and taking mean of side lengths as d
                            a_i = (mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1])
                            # >>> should support different coefficients or reference volumes for rectangular cross sections <<<

                        # added mass
                        AmatE = rho*v_i * Ca_End*mem.qMat                             # local added mass matrix

                        self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:],AmatE) # add to global added mass matrix for Morison members
                        
                        # inertial excitation
                        ImatE = rho*v_i * (1+Ca_End)*mem.qMat                         # local inertial excitation matrix

                        for i in range(self.nw):                                         # for each wave frequency...

                            F_exc_iner_temp = np.matmul(ImatE, mem.ud[il,:,i])            # local inertial excitation force complex amplitude in x,y,z

                            F_exc_iner_temp += mem.pDyn[il,i]*a_i *mem.q                 # add dynamic pressure - positive with q if end A - determined by sign of a_i

                            mem.F_exc_iner[il,:,i] += F_exc_iner_temp                    # add to stored member force vector

                            self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_iner_temp) # add to global excitation vector (frequency dependent)



    def calcLinearizedTerms(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent linearized coefficients.

        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes [m, rad]

        '''

        rho = self.env.rho
        g   = self.env.g

        # The linearized coefficients to be calculated

        B_hydro_drag = np.zeros([6,6])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]


        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # node displacement, velocity, and acceleration (each [3 x nw])
                drnode, vnode, anode = getVelocity(mem.r[il,:], Xi, self.w)      # get node complex velocity spectrum based on platform motion's and relative position from PRP


                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # interpolate coefficients for the current strip
                    Cd_q   = np.interp( mem.ls[il], mem.stations, mem.Ca_q  )
                    Cd_p1  = np.interp( mem.ls[il], mem.stations, mem.Ca_p1 )
                    Cd_p2  = np.interp( mem.ls[il], mem.stations, mem.Ca_p2 )
                    Cd_End = np.interp( mem.ls[il], mem.stations, mem.Ca_End)


                    # ----- compute side effects ------------------------

                    # member acting area assigned to this node in each direction
                    a_i_q  = np.pi*mem.ds[il]*mem.dls[il]  if circ else  2*(mem.ds[il,0]+mem.ds[il,0])*mem.dls[il]
                    a_i_p1 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,0]      *mem.dls[il]
                    a_i_p2 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,1]      *mem.dls[il]

                    # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
                    vrel = mem.u[il,:] - vnode

                    # break out velocity components in each direction relative to member orientation [nw]
                    vrel_q  = vrel*mem.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                    vrel_p1 = vrel*mem.p1[:,None]
                    vrel_p2 = vrel*mem.p2[:,None]

                    # get RMS of relative velocity component magnitudes (real-valued)
                    vRMS_q  = np.linalg.norm( np.abs(vrel_q ) )                          # equivalent to np.sqrt( np.sum( np.abs(vrel_q )**2) /nw)
                    vRMS_p1 = np.linalg.norm( np.abs(vrel_p1) )
                    vRMS_p2 = np.linalg.norm( np.abs(vrel_p2) )

                    # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
                    Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * a_i_q  * Cd_q
                    Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * a_i_p1 * Cd_p1
                    Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * a_i_p2 * Cd_p2

                    Bmat = Bprime_q*mem.qMat + Bprime_p1*mem.p1Mat + Bprime_p2*mem.p2Mat # damping matrix for the node based on linearized drag coefficients

                    B_hydro_drag += translateMatrix3to6DOF(mem.r[il,:], Bmat)            # add to global damping matrix for Morison members

                    for i in range(self.nw):

                        mem.F_exc_drag[il,:,i] = np.matmul(Bmat, mem.u[il,:,i])          # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]

                        F_hydro_drag[:,i] += translateForce3to6DOF( mem.r[il,:], mem.F_exc_drag[il,:,i])   # add to global excitation vector (frequency dependent)


                    # ----- add end/axial effects for added mass, and excitation including dynamic pressure ------
                    # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))

                    Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*a_i*Cd_End

                    Bmat = Bprime_End*mem.qMat                                       #

                    B_hydro_drag += translateMatrix3to6DOF(mem.r[il,:], Bmat)        # add to global damping matrix for Morison members

                    for i in range(self.nw):                                         # for each wave frequency...

                        F_exc_drag_temp = np.matmul(Bmat, mem.u[il,:,i])             # local drag excitation force complex amplitude in x,y,z

                        mem.F_exc_drag[il,:,i] += F_exc_drag_temp                    # add to stored member force vector

                        F_hydro_drag[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_drag_temp) # add to global excitation vector (frequency dependent)


        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag, F_hydro_drag


    def plot(self, ax):
        '''plots the FOWT...'''


        # loop through each member and plot it
        for mem in self.memberList:

            mem.calcOrientation()  # temporary

            mem.plot(ax)

        # in future should consider ability to animate mode shapes and also to animate response at each frequency
        # including hydro excitation vectors stored in each member
