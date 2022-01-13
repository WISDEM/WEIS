# RAFT's support structure member class

import numpy as np

from raft.helpers import *

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

        self.potMod = getFromDict(mi, 'potMod', dtype=bool, default=False)     # hard coding BEM analysis enabled for now <<<< need to move this to the member YAML input instead <<<
        

        # heading feature for rotation members about the z axis (used for rotated patterns)
        self.headings = getFromDict(mi, 'headings', shape=-1, default=0.0)
        self.heading = getFromDict(mi, 'heading', default=0.0)            # rotation about z axis to apply to the member [deg]
        if self.heading != 0.0:
            c = np.cos(np.deg2rad(self.heading))
            s = np.sin(np.deg2rad(self.heading))
            rotMat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.rA = np.matmul(rotMat, self.rA)
            self.rB = np.matmul(rotMat, self.rB)


        rAB = self.rB-self.rA                                        # The relative coordinates of upper node from lower node [m]
        self.l = np.linalg.norm(rAB)                                 # member length [m]
    

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
        
        if isinstance(self.l_fill, np.ndarray):
            if len(self.l_fill) != n-1 or len(self.rho_fill) != n-1:
                raise ValueError(f'The number of stations ({n}) should always be 1 greater than the number of ballast sections, l_fill ({len(self.l_fill)}) and rho_fill ({len(self.rho_fill)})')
            
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
            self.cap_t        = getFromDict(mi, 'cap_t'   , shape=cap_stations.shape[0])   # thicknesses [m]
            self.cap_d_in     = getFromDict(mi, 'cap_d_in', shape=cap_stations.shape[0])   # inner diameter (if it isn't a solid plate) [m]
            self.cap_stations = (cap_stations - A[0])/(A[-1] - A[0])*self.l             # calculate station positions along the member axis from 0 to l [m]
            

        # Drag coefficients
        self.Cd_q   = getFromDict(mi, 'Cd_q' , shape=n, default=0.0 )     # axial drag coefficient
        if not np.isscalar(mi['Cd']) and len(mi['Cd'])==2:  # special case for rectangular members with directional coefficients
            self.Cd_p1 = np.tile(float(mi['Cd'][0]), [n])
            self.Cd_p2 = np.tile(float(mi['Cd'][1]), [n])
        else:
            self.Cd_p1  = getFromDict(mi, 'Cd'   , shape=n, default=0.6 )     # transverse1 drag coefficient
            self.Cd_p2  = getFromDict(mi, 'Cd'   , shape=n, default=0.6 )     # transverse2 drag coefficient
        self.Cd_End = getFromDict(mi, 'CdEnd', shape=n, default=0.6 )     # end drag coefficient
        # Added mass coefficients
        self.Ca_q   = getFromDict(mi, 'Ca_q' , shape=n, default=0.0 )     # axial added mass coefficient
        if not np.isscalar(mi['Ca']) and len(mi['Ca'])==2:  # special case for rectangular members with directional coefficients
            self.Ca_p1 = np.tile(float(mi['Ca'][0]), [n])
            self.Ca_p2 = np.tile(float(mi['Ca'][1]), [n])
        else:
            self.Ca_p1  = getFromDict(mi, 'Ca'   , shape=n, default=0.97)     # transverse1 added mass coefficient
            self.Ca_p2  = getFromDict(mi, 'Ca'   , shape=n, default=0.97)     # transverse2 added mass coefficient
        self.Ca_End = getFromDict(mi, 'CaEnd', shape=n, default=0.6 )     # end added mass coefficient


        # discretize into strips with a node at the midpoint of each strip (flat surfaces have dl=0)
        dorsl  = list(self.d) if self.shape=='circular' else list(self.sl)   # get a variable that is either diameter or side length pair
        dlsMax = mi['dlsMax']
        
        # start things off with the strip for end A
        ls     = [0.0]                 # list of lengths along member axis where a node is located <<< should these be midpoints instead of ends???
        dls    = [0.0]                 # lumped node lengths (end nodes have half the segment length)
        ds     = [0.5*dorsl[0]]       # mean diameter or side length pair of each strip
        drs    = [0.5*dorsl[0]]       # change in radius (or side half-length pair) over each strip (from node i-1 to node i)

        for i in range(1,n):

            lstrip = self.stations[i]-self.stations[i-1]             # the axial length of the strip

            if lstrip > 0.0:
                ns= int(np.ceil( (lstrip) / dlsMax ))             # number of strips to split this segment into
                dlstrip = lstrip/ns
                m   = 0.5*(dorsl[i] - dorsl[i-1])/lstrip          # taper ratio
                ls  += [self.stations[i-1] + dlstrip*(0.5+j) for j in range(ns)] # add node locations
                dls += [dlstrip]*ns
                ds  += [dorsl[i-1] + dlstrip*2*m*(0.5+j) for j in range(ns)]
                drs += [dlstrip*m]*ns
                
            elif lstrip == 0.0:                                      # flat plate case (ends, and any flat transitions), a single strip for this section
                dlstrip = 0
                ls  += [self.stations[i-1]]                          # add node location
                dls += [dlstrip]
                ds  += [0.5*(dorsl[i-1] + dorsl[i])]               # set diameter as midpoint diameter
                drs += [0.5*(dorsl[i] - dorsl[i-1])]

            # finish things off with the strip for end B
            dlstrip = 0
            ls  += [self.stations[-1]]         
            dls += [0.0]
            ds  += [0.5*dorsl[-1]]
            drs += [-0.5*dorsl[-1]]
        
        # >>> may want to have a way to not have an end strip for members that intersect things <<<
        
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
        self.F_exc_a   = np.zeros([self.ns,3,nw], dtype=complex)            #  component due to wave acceleration
        self.F_exc_p   = np.zeros([self.ns,3,nw], dtype=complex)            #  component due to dynamic pressure
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


        p1 = np.matmul( R, [1,0,0] )               # unit vector that is in the 'beta' plane if gamma is zero
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
        
        mass_center = 0                                 # total sum of mass the center of mass of the member [kg-m]
        mshell = 0                                      # total mass of the shell material only of the member [kg]
        self.vfill = []                                 # list of ballast volumes in each submember [m^3] - stored in the object for later access
        mfill = []                                      # list of ballast masses in each submember [kg]
        pfill = []                                      # list of ballast densities in each submember [kg]
        self.M_struc = np.zeros([6,6])                  # member mass/inertia matrix [kg, kg-m, kg-m^2]

        # loop through each sub-member
        for i in range(1,len(self.stations)):                            # start at 1 rather than 0 because we're looking at the sections (from station i-1 to i)

            # initialize common variables
            rA = self.rA + self.q*self.stations[i-1]    # lower node position of the submember [m]
            l = self.stations[i]-self.stations[i-1]     # length of the submember [m]
            if l==0.0:
                mass = 0
                center = np.zeros(3)
                m_shell = 0
                v_fill = 0
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
            self.vfill.append(v_fill)                        # list of ballast volumes in each submember [m^3]
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
            self.M_struc += translateMatrix6to6DOF(Mmat, center) # mass matrix of the member about the PRP


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
            self.M_struc += translateMatrix6to6DOF(Mmat, center_cap) # mass matrix of the member about the PRP


        mass = self.M_struc[0,0]        # total mass of the entire member [kg]
        center = mass_center/mass       # total center of mass of the entire member from the PRP [m]


        return mass, center, mshell, mfill, pfill




    def getHydrostatics(self, rho, g):
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
                dFz_dz   = -rho*g*AWP /cosPhi

                # note: below calculations are based on untapered case, but
                # temporarily approximated for taper by using dWP (diameter at water plane crossing) <<< this is rough

                # buoyancy force and moment about end A
                Fz = rho*g* V_UWi
                M  = -rho*g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
                Mx = M*dPhi_dThx
                My = M*dPhi_dThy

                Fvec[2] += Fz                           # vertical buoyancy force [N]
                Fvec[3] += Mx + Fz*rA[1]                # moment about x axis [N-m]
                Fvec[4] += My - Fz*rA[0]                # moment about y axis [N-m]


                # normal approach to hydrostatic stiffness, using this temporarily until above fancier approach is verified
                Cmat[2,2] += -dFz_dz
                Cmat[2,3] += rho*g*(     -AWP*yWP    )
                Cmat[2,4] += rho*g*(      AWP*xWP    )
                Cmat[3,2] += rho*g*(     -AWP*yWP    )
                Cmat[3,3] += rho*g*(IxWP + AWP*yWP**2 )
                Cmat[3,4] += rho*g*(      AWP*xWP*yWP)
                Cmat[4,2] += rho*g*(      AWP*xWP    )
                Cmat[4,3] += rho*g*(      AWP*xWP*yWP)
                Cmat[4,4] += rho*g*(IyWP + AWP*xWP**2 )

                Cmat[3,3] += rho*g*V_UWi * r_center[2]
                Cmat[4,4] += rho*g*V_UWi * r_center[2]

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
                Fvec += translateForce3to6DOF(np.array([0, 0, rho*g*V_UWi]), r_center)

                # hydrostatic stiffness matrix (about end A)
                Cmat[3,3] += rho*g*V_UWi * r_center[2]
                Cmat[4,4] += rho*g*V_UWi * r_center[2]

                V_UW += V_UWi
                r_centerV += r_center*V_UWi

            else: # if the members are fully above the surface

                pass

        if V_UW > 0:
            r_center = r_centerV/V_UW    # calculate overall member center of buoyancy
        else:
            r_center = np.zeros(3)       # temporary fix for out-of-water members
        
        return Fvec, Cmat, V_UW, r_center, AWP, IWP, xWP, yWP


    def plot(self, ax, r_ptfm=[0,0,0], R_ptfm=[], color='k', nodes=0):
        '''Draws the member on the passed axes, and optional platform offset and rotation matrix'''

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


        # ----- move to global frame ------------------------------
        newcoords = np.matmul(self.R, coords)          # relative orientation in platform

        newcoords = newcoords + self.rA[:,None]        # shift to end A location, still relative to platform
        
        if len(R_ptfm) > 0:
            newcoords = np.matmul(R_ptfm, newcoords)   # account for offset platform orientation

        # apply platform translational offset
        Xs = newcoords[0,:] + r_ptfm[0]
        Ys = newcoords[1,:] + r_ptfm[1]
        Zs = newcoords[2,:] + r_ptfm[2]
        
        # plot on the provided axes
        linebit = []  # make empty list to hold plotted lines, however many there are
        for i in range(n):  #range(int(len(Xs)/2-1)):
            #linebit.append(ax.plot(Xs[2*i:2*i+2],Ys[2*i:2*i+2],Zs[2*i:2*i+2]            , color='k'))  # side edges
            #linebit.append(ax.plot(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]],Zs[[2*i,2*i+2]]      , color='k'))  # end A edges
            #linebit.append(ax.plot(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]],Zs[[2*i+1,2*i+3]], color='k'))  # end B edges

            linebit.append(ax.plot(Xs[m*i:m*i+m],Ys[m*i:m*i+m],Zs[m*i:m*i+m]            , color=color, lw=0.5, zorder=2))  # side edges

        for j in range(m):
            linebit.append(ax.plot(Xs[j::m], Ys[j::m], Zs[j::m]            , color=color, lw=0.5, zorder=2))  # station rings


        # plot nodes if asked
        if nodes > 0:
            ax.scatter(self.r[:,0], self.r[:,1], self.r[:,2])

        return linebit


