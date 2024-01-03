# RAFT's floating wind turbine class

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, interp2d, griddata

import raft.member2pnl as pnl
from raft.helpers import *
from raft.raft_member import Member
from raft.raft_rotor import Rotor
import moorpy as mp

# deleted call to ccblade in this file, since it is called in raft_rotor
# also ignoring changes to solveEquilibrium3 in raft_model and the re-addition of n=len(stations) in raft_member, based on raft_patch



class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, design, w, mpb, depth=600, x_ref=0, y_ref=0, heading_adjust=0):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.

        Parameters
        ----------
        design : dict
            Dictionary of the design - must include sections for site, turbine, platform, and mooring
        w
            Array of frequencies to be used in analysis (rad/s)
        mpb
            A MoorPy Body object that represents this FOWT in MoorPy
        mooring : dict
            A dictionary describing the mooring system that is specific to this FOWT (to be stored inside this object)
        depth
            Water depth, positive-down. (m)
        x_ref : float
            Reference x location of this fowt in the array [m]
        y_ref : float
            Reference y location of this fowt in the array [m]
        heading_adjust : float
            Rotation to the heading of the platform and mooring system to be applied [deg]
        '''

        print("Making FOWT")

        # from Joep: for muliple HAMS threads and multiple wave headings >>>
        #self.numThreads = int(getFromDict(design['settings'],'numThreads', default=8)) # number of threads to run HAMS
        #self.headsStored = np.empty(1)  
        
        # basic setup
        self.nDOF = 6
        self.nw = len(w)                                          # number of frequencies
        self.Xi0 = np.zeros( self.nDOF)                           # mean offsets of platform from its reference point [m, rad]
        self.Xi  = np.zeros([self.nDOF, self.nw], dtype=complex)  # complex response amplitudes as a function of frequency  [m, rad]
        
        # position in the array
        self.x_ref = x_ref      # reference x position of the FOWT in the array [m]
        self.y_ref = y_ref      # reference y position of the FOWT in the array [m]
        self.r6 = np.zeros(6)   # mean position/orientation in absolute/array coordinates [m,rad]
        
        # count number of platform members
        self.nplatmems = 0
        for platmem in design['platform']['members']:
            if 'heading' in platmem:
                #print(platmem['heading'])
                self.nplatmems += len(platmem['heading'])
            else:
                self.nplatmems += 1
        
        # count numbers of rotors and their towers and copy some info
        if 'turbine' in design:
        
            self.nrotors = getFromDict(design['turbine'], 'nrotors', dtype=int, shape=0, default=1)
            if self.nrotors==1: design['turbine']['nrotors'] = 1
            
            if 'tower' in design['turbine']:
                if isinstance(design['turbine']['tower'], dict):                            # if a single tower is specified (rather than a list)
                    design['turbine']['tower'] = [design['turbine']['tower']]*self.nrotors  # replicate the tower info for each rotor
            
                self.ntowers = len(design['turbine']['tower'])
            
            else:
                self.ntowers = 0

            # copy over site info into turbine dictionary
            design['turbine']['rho_air'       ] = getFromDict(design['site'], 'rho_air', shape=0, default=1.225)
            design['turbine']['mu_air'        ] = getFromDict(design['site'], 'mu_air', shape=0, default=1.81e-05)
            design['turbine']['shearExp_air'  ] = getFromDict(design['site'], 'shearExp_air', shape=0, default=0.12)
            design['turbine']['rho_water'     ] = getFromDict(design['site'], 'rho_water', shape=0, default=1025.0)
            design['turbine']['mu_water'      ] = getFromDict(design['site'], 'mu_water', shape=0, default=1.0e-03)
            design['turbine']['shearExp_water'] = getFromDict(design['site'], 'shearExp_water', shape=0, default=0.12)
            
            # turbine RNA description 
            self.mRNA    = getFromDict(design['turbine'], 'mRNA'   , shape=self.nrotors)
            self.IxRNA   = getFromDict(design['turbine'], 'IxRNA'  , shape=self.nrotors)
            self.IrRNA   = getFromDict(design['turbine'], 'IrRNA'  , shape=self.nrotors)
            self.xCG_RNA = getFromDict(design['turbine'], 'xCG_RNA', shape=self.nrotors)
            self.hHub    = getFromDict(design['turbine'], 'hHub'   , shape=self.nrotors)
        
        else: # no turbines/rotors case
        
            self.nrotors = 0  # this will ensure RAFT doesn't set up or look for any Rotor objects
            self.ntowers = 0
            
            # note: RNA descriptions are not defined because there is no turbine


        self.rotorList = []

        self.depth = depth

        self.w = np.array(w)
        self.dw = w[1]-w[0]         # frequency increment [rad/s]    
        
        self.k = np.array([waveNumber(w, self.depth) for w in self.w])  # wave number [m/rad]

        self.rho_water = getFromDict(design['site'], 'rho_water', default=1025.0)
        self.g         = getFromDict(design['site'], 'g'        , default=9.81)
        self.shearExp_water = getFromDict(design['site'], 'shearExp_water', default=0.12)
        
        # <<<<<<<<<<< Should it be something like self.dlsMax?
        #design['turbine']['tower']['dlsMax'] = getFromDict(design['turbine']['tower'], 'dlsMax', shape=-1, default=5.0)

        potModMaster = getFromDict(design['platform'], 'potModMaster', dtype=int, default=0)
        self.potModMaster = potModMaster
        dlsMax       = getFromDict(design['platform'], 'dlsMax'      , default=5.0)
        min_freq_BEM = getFromDict(design['platform'], 'min_freq_BEM', default=self.dw/2/np.pi)
        self.dw_BEM  = 2.0*np.pi*min_freq_BEM
        self.dz_BEM  = getFromDict(design['platform'], 'dz_BEM', default=3.0)
        self.da_BEM  = getFromDict(design['platform'], 'da_BEM', default=2.0)


        #self.aeroServoMod = np.atleast_1d(getFromDict(design['turbine'], 'aeroServoMod', default=1))  # flag for aeroservodynamics (0=none, 1=aero only, 2=aero and control)


        # member-based platform description
        self.memberList = []                                         # list of member objects

        for mi in design['platform']['members']:

            # prepare member info
            if potModMaster==1:
                mi['potMod'] = False
            elif potModMaster==2:
                mi['potMod'] = True
                
            if 'dlsMax' not in mi:    # apply the global dlsMax setting to any member that doesn't have its own setting
                mi['dlsMax'] = dlsMax

            headings = getFromDict(mi, 'heading', shape=-1, default=0.)
            mi['headings'] = headings   # differentiating the list of headings/copies of a member from the individual member's heading
            
            # create member object
            if np.isscalar(headings):
                self.memberList.append(Member(mi, self.nw, heading=headings+heading_adjust))
            else:
                for heading in headings:
                    self.memberList.append(Member(mi, self.nw, heading=heading+heading_adjust))
        
        
        # add tower(s) to member list if applicable
        for ti in range(self.ntowers):
            #tower['dlsMax'] = design['turbine']['tower']['dlsMax']     # for when we want to make a list of tower 'members' and have general inputs like dlsMax first
            self.memberList.append(Member(design['turbine']['tower'][ti], self.nw))
        #TODO: consider putting the tower somewhere else rather than in end of memberList <<<

        # array-level mooring system connection
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine

        # this FOWT's own MoorPy system (may not be used)
        if design['mooring']:

            self.ms = mp.System()
            self.ms.parseYAML(design['mooring'])
            
            # ensure proper setup with one coupled Body tied to this FOWT
            if len(self.ms.bodyList) == 0:
                self.ms.addBody(-1, [0,0,0,0,0,0]) # create a new body if needed
                for point in self.ms.pointList:
                    if point.type == -1:  # attached any coupled points to the body
                        self.ms.bodyList[0].attachPoint(point.number, point.r)
                        point.type = 1  # now indicate point is fixed (to the body)
                        
            elif len(self.ms.bodyList) == 1:
                self.ms.bodyList[0].type = -1  # ensure it's set to coupled type
            else:
                raise Exception("More than one body detected in FOWT mooring system.")
                
            # move mooring system according to the FOWT's reference position
            self.ms.transform(trans=[x_ref, y_ref], rot=heading_adjust)  
            self.ms.initialize()

        else:
            self.ms = None
        
        self.F_moor0 = np.zeros(6)     # mean mooring forces in a given scenario
        self.C_moor = np.zeros([6,6])  # mooring stiffness matrix in a given scenario

        if 'yaw_stiffness' in design['platform']:
            self.yawstiff = design['platform']['yaw_stiffness']       # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        else:
            self.yawstiff = 0

        # build Rotor list
        for ir in range(self.nrotors):
            self.rotorList.append(Rotor(design['turbine'], self.w, ir))
            #self.rotorCoords.append(design['turbine']['rotorCoords'])
        
        # initialize mean force arrays to zero, so the model can work before we calculate excitation
        self.f_aero0 = np.zeros([6, self.nrotors])
        # mean weight and hydro force arrays are set elsewhere. In future hydro could include current.
        self.D_hydro = np.zeros(6)      # initialize mean drag force from current - acts as a "static" force, like f_aero0

        # flag to signal whether any BEM hydro modeling is done
        self.potMod = any([member['potMod']==True for member in design['platform']['members'] ])

        # initialize BEM arrays, whether or not a BEM solver is used
        # __, __, nWaveHeadings = getUniqueCaseHeadings(design['cases']['keys'], design['cases']['data'])  # identify how many wave headings need to be preprocessed
        self.A_BEM = np.zeros([6,6,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([6,6,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        #self.X_BEM = np.zeros([self.nWaves, 6, self.nw], dtype=complex)               # linaer wave excitation force/moment coefficients vector [N, N-m]
        #self.F_BEM = np.zeros([self.nWaves, 6, self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]
        # <<< TODO: Set maximum of amount of headingsm for X_BEM and F_BEM and use interpolation of hydrodynamics. Or not?

        # Add a flag to either not compute 2nd order hydro; read QTF; or compute it with a slender-body approximation
        # self.qtf = np.zeros([nw1, nw2, nheads, self.nDOF], dtype=complex)
        if 'qtfPath' in design['platform']:
            self.secondOrderWaveMod = 1     # <<< temporary, until we implement a flag like this in the inputs
            self.qtfPath = design['platform']['qtfPath']
            self.readQTF(self.qtfPath)
            self.mu_2nd = self.w - self.w[0] # The frequency array for difference-frequency second-order is the same as self.w, but starting at 0    
        else:
            self.secondOrderWaveMod = 0    # <<< temporary


    def setPosition(self, r6):
        '''Updates the FOWT's (mean) position including all components.
        
        r6 : float array
            6 DOF absolute position of FOWT [m, rad]
        '''
        
        # if offset provided, set things according to those positions, otherwise zero it
        #if r6:
        self.r6 = r6
        #else:
        #    self.r6 = np.array([x_ref, y_ref, 0, 0, 0, 0])
        self.Xi0 = self.r6 - np.array([self.x_ref, self.y_ref, 0, 0, 0, 0])
        
        # calculate and save a rotation/orientation matrix
        self.Rmat = rotationMatrix(*self.r6[3:])  # rotation matrix for fowt orientation
        
        # set the positions of the FOWT's members, rotors, and MoorPy system
        if self.ms:
            self.ms.bodyList[0].setPosition(self.r6)
        
        for rot in self.rotorList:
            rot.setPosition(r6=self.r6)
            
        for mem in self.memberList:
            mem.setPosition(r6=self.r6)
        
        # solve the mooring system equilibrium of this FOWT's own MoorPy system
        if self.ms:
            self.ms.solveEquilibrium()
            self.C_moor = self.ms.getCoupledStiffnessA()
            self.F_moor0 = self.ms.bodyList[0].getForces(lines_only=True)
        

    def calcStatics(self):
        '''Computes the static properties of the FOWT in terms of mass and hydrostatic-related
        matrices and mean force vectors about the FOWT PRP in unrotated directions, based on
        the mean offsets of the FOWT, Xi0.
        '''

        rho = self.rho_water
        g   = self.g

        # structure-related arrays
        self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]
        
        # hydrostatic arrays
        self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]  <<<<< not used yet


        # --------------- add in linear hydrodynamic coefficients here if applicable --------------------
        #[as in load them] <<<<<<<<<<<<<<<<<<<<<

        # --------------- Get general geometry properties including hydrostatics ------------------------

        # initialize some variables for running totals
        VTOT = 0.                   # Total underwater volume of all members combined
        m_all = 0.                   # Total mass of all members [kg]
        AWP_TOT = 0.                # Total waterplane area of all members [m^2]
        IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]
        IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]
        Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
        Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
        m_center_sum = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)

        self.m_sub = 0              # total mass of just the members that make up the substructure [kg]
        self.C_struc_sub = np.zeros([6,6])  # substructure effective stiffness matrix [N/m, N, N-m]
        self.M_struc_sub = np.zeros([6,6])  # total mass matrix of just the substructure about the PRP
        m_sub_sum = 0               # product of each substructure member's mass and CG, to be used to find the total substructure CG [kg-m]
        self.m_shell = 0             # total mass of the shells/steel of the members in the substructure [kg]
        mballast = []               # list to store the mass of the ballast in each of the substructure members [kg]
        pballast = []               # list to store the density of ballast in each of the substructure members [kg]
        '''
        I44list = []                # list to store the I44 MoI about the PRP of each substructure member
        I55list = []                # list to store the I55 MoI about the PRP of each substructure member
        I66list = []                # list to store the I66 MoI about the PRP of each substructure member
        masslist = []               # list to store the mass of each substructure member
        '''
        self.mtower = np.zeros(self.ntowers)    # assume that the whole tower will always be one member
        self.rCG_tow = []

        # loop through each member
        for i,mem in enumerate(self.memberList):

            # calculate member's orientation information (stored in the member and used in later steps)
            mem.setPosition(r6=self.r6)  # <<< is this redundant, assume fowt.setPosition has been called?
            
            # note: quantities in the following section are relative to the PRP (but with global direcions)

            # ---------------------- get member's mass and inertia properties ------------------------------
            # get member mass and inertia info (including mem.M_struc) <<< still split between converting to PRP in or out of these functions
            mass, center, m_shell, mfill, pfill = mem.getInertia(rPRP=self.r6[:3]) 

            # Calculate the mass matrix of the FOWT about the PRP
            self.W_struc += translateForce3to6DOF( np.array([0,0, -g*mass]), center )  # weight vector
            self.M_struc += mem.M_struc     # mass/inertia matrix about the PRP
            
            m_center_sum += center*mass     # product sum of the mass and center of mass to find the total center of mass [kg-m]

            # Tower calculations
            if mem.type <= 1:   # <<<<<<<<<<<< maybe find a better way to do the if condition
                self.mtower[i-self.nplatmems] = mass                  # mass of the tower [kg]
                self.rCG_tow.append(center)               # center of mass of the tower from the PRP [m]
            # Substructure calculations
            if mem.type > 1:
                self.m_sub += mass              # mass of the substructure
                self.M_struc_sub += mem.M_struc     # mass matrix of the substructure about the PRP
                m_sub_sum += center*mass        # product sum of the substructure members and their centers of mass [kg-m]
                self.m_shell += m_shell               # mass of the substructure shell material [kg]
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

            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(rho=self.rho_water, g=self.g, rPRP=self.r6[:3])
            
            # add to fowt's mean force vector and stiffness matrix
            self.W_hydro += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix
       
            # convert other metrics to also be about the PRP (platform reference point)
            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP
            
        
        # ------------- include buoyancy effects of underwater rotors -------------
        # loop through each blade member to calculate rotor buoyancy forces (for underwater turbines)
        for i in range(self.nrotors):       # for each rotor in the fowt

            rotor = self.rotorList[i]

            if rotor.Zhub < 0:      # only do this for underwater rotors
                
                for j in range(int(rotor.nBlades)):     # for each blade on the rotor

                    # ensure the blade headings are equally spaced apart (so that the specific heading values are arbitrary)
                    if all(np.mod(np.diff(rotor.headings, append=rotor.headings[0]),360) != np.mod(np.diff(rotor.headings, append=rotor.headings[0])[0], 360)):
                        raise ValueError("The headings of the blades need to be equally spaced apart")

                    for k,afmem in enumerate(rotor.bladeMemberList):    # for each airfoil member in the bladeMemberList

                        # store original positions of the airfoil member about the original rotor axis
                        rA_OG = afmem.rA0
                        rB_OG = afmem.rB0
                        rOG = np.vstack([rA_OG, rB_OG])

                        # set the heading, or azimuth angle, of the blade member
                        afmem.heading = rotor.headings[j]

                        # find the end nodes of the blade member about the global coordinates (e.g,, if rA_OG = [0,0,0] and rB_OG=[0,1,0], and heading=90, then rA=[0,0,0] and rB=[0,0,1])
                        r_new = rotor.getBladeMemberPositions(rotor.headings[j], rOG)

                        # save these end node positions in the blade member
                        afmem.rA0 = r_new[0,:]
                        afmem.rB0 = r_new[1,:]

                        # save the positions of the nodes for each blade
                        rotor.nodes[j,k,:] = afmem.rA0
                        if k==len(rotor.bladeMemberList)-1:     # if it's the last blade member, save it's rB position to the last position in the nodes array
                            rotor.nodes[j,k+1,:] = afmem.rB0

                        # find the actual orientation vectors of the blade member
                        afmem.setPosition()

                        # calculate the mass and inertial properties of the blade members
                        #mass, center, m_shell, mfill, pfill = afmem.getInertia()
                        # >>>>>> can be used later if actual rectangular mass properties are desired other than mRNA <<<<<<<<

                        # calculate hydrostatic properties of the blade (sub)member and add them to the system matrices
                        Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = afmem.getHydrostatics(rho=self.rho_water, g=self.g, rPRP=self.r6[:3])
                        
                        # outputs of getHydrostatics should already be about the PRP
                        self.W_hydro += Fvec # buoyancy vector
                        self.C_hydro += Cmat # hydrostatic stiffness matrix

                        VTOT    += V_UW    # add to total underwater volume of all members combined
                        AWP_TOT += AWP
                        IWPx_TOT += IWP + AWP*yWP**2
                        IWPy_TOT += IWP + AWP*xWP**2
                        Sum_V_rCB   += r_CB*V_UW
                        Sum_AWP_rWP += np.array([xWP, yWP])*AWP

                        # reset original rA and rB values of the airfoil member
                        afmem.rA0 = rA_OG
                        afmem.rB0 = rB_OG
                        afmem.setPosition()
                        
                        # Note: it might be possible to streamline the above using new capabilities in setPosition (but not sure).



        # ------------------------- include RNA properties -----------------------------

        # Here we could initialize first versions of the structure matrix components.
        # These might be iterated on later to deal with mean- or amplitude-dependent terms.
        #self.M_struc += structural.M_lin(q0,      self.turbineParams)        # Linear Mass Matrix
        #self.B_struc += structural.C_lin(q0, qd0, self.turbineParams, u0)    # Linear Damping Matrix
        #self.C_struc += structural.K_lin(q0, qd0, self.turbineParams, u0)    # Linear Stifness Matrix
        #self.W_struc += structural.B_lin(q0, qd0, self.turbineParams, u0)    # Linear RHS

        # below are temporary placeholders
        # for now, turbine RNA is specified by some simple lumped properties

        for i in range(self.nrotors):
            Mmat = np.diag([self.mRNA[i], self.mRNA[i], self.mRNA[i], self.IxRNA[i], self.IrRNA[i], self.IrRNA[i]])            # create mass/inertia matrix
            center = np.array([self.xCG_RNA[i], self.rotorList[i].rHub[1], self.hHub[i]])   # RNA center of mass location <<< need to overhaul this!

            # compute rotated RNA mass properties considering current fowt pose <<< need to double check these
            Mmat = rotateMatrix6(Mmat, self.Rmat)       # adjust the inertia matrix to align with global directions
            center = np.matmul(self.Rmat, center)      # adjust center offset from PRP along global directions

            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
            self.W_struc += translateForce3to6DOF(np.array([0,0, -g*self.mRNA[i]]), center )   # weight vector
            self.M_struc += translateMatrix6to6DOF(Mmat, center)                            # mass/inertia matrix
            m_center_sum += center*self.mRNA[i]

        # ----------- process inertia-related totals ----------------

        m_all = self.M_struc[0,0]             # total mass of all the members
        rCG_all = m_center_sum/m_all          # total CG of all the members
        
        self.rCG = rCG_all
        self.rCG_sub = m_sub_sum/self.m_sub   # solve for just the substructure mass and CG
        
        
  
        # get principal moments of inertia (about CG) 
        # note: these are likely only useful in the unrotated frame, i.e.
        # the first time calcStatics is called.
        
        # the mass matrix of the substructure about the substruc's CM
        M_sub = translateMatrix6to6DOF(self.M_struc_sub, -self.rCG_sub)  # -rCG_sub due to convention of the function
        
        # overall structure mass matrix about its CM
        M_all = translateMatrix6to6DOF(self.M_struc, -self.rCG)

        # could check that off-diagonals are approximately zero as an error check
        
        
        # Solve for the total mass of each type of ballast in the substructure
        self.pb = []                                                # empty list to store the unique ballast densities
        for i in range(len(pballast)):
            if pballast[i] != 0:                                    # if the value in pballast is not zero
                if self.pb.count(pballast[i]) == 0:                 # and if that value is not already in pb
                    self.pb.append(pballast[i])                     # store that ballast density value

        self.m_ballast = np.zeros(len(self.pb))                      # make an empty m_ballast list with len=len(pb)
        for i in range(len(self.pb)):                               # for each ballast density
            for j in range(len(mballast)):                          # loop through each ballast mass
                if float(pballast[j]) == float(self.pb[i]):   # but only if the index of the ballast mass (density) matches the value of pb
                    self.m_ballast[i] += mballast[j]                 # add that ballast mass to the correct index of mballast



        # ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------
                               # save the total underwater volume
        rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform
        
        if VTOT==0: # if you're only working with members above the platform, like modeling the wind turbine
            zMeta = 0
        else:
            zMeta   = rCB_TOT[2] + IWPx_TOT/VTOT  # add center of buoyancy and BM=I/v to get z elevation of metecenter [m] (have to pick one direction for IWP)

        self.C_struc[3,3] = -m_all*g*rCG_all[2]
        self.C_struc[4,4] = -m_all*g*rCG_all[2]
        
        self.C_struc_sub[3,3] = -self.m_sub*g*self.rCG_sub[2]
        self.C_struc_sub[4,4] = -self.m_sub*g*self.rCG_sub[2]

        # add relevant properties to this turbine's MoorPy Body
        # >>> should double check proper handling of mean weight and buoyancy forces throughout model <<<
        if self.body:   # note: this is likely unused now <<<
            self.body.m = m_all
            self.body.v = VTOT
            self.body.rCG = rCG_all
            self.body.AWP = AWP_TOT
            self.body.rM = np.array([rCB_TOT[0], rCB_TOT[1], zMeta])    # now includes x and y coordinates for center of buoyancy
        #is there any risk of additional moments due to offset CB since MoorPy assumes CB at ref point? <<<
        self.rCB = rCB_TOT
        self.m = m_all
        self.V = VTOT
        self.AWP = AWP_TOT
        self.rM = np.array([rCB_TOT[0], rCB_TOT[1], zMeta])
      
        # save things in a dictionary now        
        self.props = {}
        self.props['m']     = self.m 
        self.props['m_sub'] = self.m_sub 
        self.props['v'] = self.V
        self.props['rCG']     = self.rCG
        self.props['rCG_sub'] = self.rCG_sub
        self.props['rCB'] = self.rCB
        self.props['AWP'] = self.AWP
        self.props['rM'] = self.rM
        self.props['Ixx'] = M_all[3,3]  # principale moments of inertia of entire structure
        self.props['Iyy'] = M_all[4,4]
        self.props['Izz'] = M_all[5,5]
        self.props['Ixx_sub'] = M_sub[3,3]  # principale moments of inertia of substructure
        self.props['Iyy_sub'] = M_sub[4,4]
        self.props['Izz_sub'] = M_sub[5,5]


    def calcBEM(self, dw=0, wMax=0, wInf=10.0, dz=0, da=0, headings=[0], meshDir=os.path.join(os.getcwd(),'BEM')):
        '''This generates a mesh for the platform and runs a BEM analysis on it
        using pyHAMS. It can also write adjusted .1 and .3 output files suitable
        for use with OpenFAST.
        The mesh is only made for non-interesecting members flagged with potMod=1.
        Note that this method does not consider mean offsets when calculating coefficients.
        
        PARAMETERS
        ----------
        dw : float
            Optional specification of custom frequency increment (rad/s).
        wMax : float
            Optional specification of maximum frequency for BEM analysis (rad/s). Will only be
            used if it is greater than the maximum frequency used in RAFT.
        wInf : float
            Optional specification of large frequency to use as approximation for infinite
            frequency in pyHAMS analysis (rad/s).
        dz : float
            desired longitudinal panel size for potential flow BEM analysis (m)
        da : float
            desired azimuthal panel size for potential flow BEM analysis (m)
        headings : list of floats
            incident wave headings to be considered, default [0] (deg). <<<<<<<< capability still needs to be added, after adjustments to pyHAMS
        '''
        
        # go through members to be modeled with BEM and calculated their nodes and panels lists
        nodes = []
        panels = []

        vertices = np.zeros([0,3])  # for GDF output

        dz = self.dz_BEM if dz==0 else dz  # allow override if provided
        da = self.da_BEM if da==0 else da

        for mem in self.memberList:

            if mem.potMod==True:
                pnl.meshMember(mem.stations, mem.d, mem.rA, mem.rB,
                        dz_max=dz, da_max=da, savedNodes=nodes, savedPanels=panels)

                # for GDF output
                vertices_i = pnl.meshMemberForGDF(mem.stations, mem.d, mem.rA, mem.rB, dz_max=dz, da_max=da)
                vertices = np.vstack([vertices, vertices_i])              # append the member's vertices to the master list


        # only try to save a mesh and run HAMS if some members DO have potMod=True
        if len(panels) > 0:
            
            import pyhams.pyhams as ph  # import PyHAMS only if we're going to use it

            pnl.writeMesh(nodes, panels, oDir=os.path.join(meshDir,'Input')) # generate a mesh file in the HAMS .pnl format

            #pnl.writeMeshToGDF(vertices)                # also a GDF for visualization

            ph.create_hams_dirs(meshDir)                #

            # HAMS needs a hydrostatics file, it's unused for .1 and .3,
            # but HAMS write the .hst file that OpenFAST uses
            ph.write_hydrostatic_file(meshDir, kHydro=self.C_hydro)

            # prepare frequency settings for HAMS
            dw_HAMS = self.dw_BEM if dw==0 else dw     # frequency increment - allow override if provided

            wMax_HAMS = max(wMax, max(self.w))         # make sure the HAMS runs includes both RAFT and export frequency extents

            nw_HAMS = int(np.ceil(wMax_HAMS/dw_HAMS))  # ensure the upper frequency of the HAMS analysis is large enough

            ph.write_control_file(meshDir, waterDepth=self.depth, incFLim=1, iFType=3, oFType=4,   # inputs are in rad/s, outputs in s
                                  numFreqs=-nw_HAMS, minFreq=dw_HAMS, dFreq=dw_HAMS,
                                  numHeadings=len(headings), headingList=headings) #, numThreads=self.numThreads) <<<
            
            # Note about zero/infinite frequencies from WAMIT-formatted output files (as per WAMIT v7 manual): 
            # The limiting values of the added-mass coefficients may be evaluated for zero or infinite
            # period by specifying the values PER= 0:0 and PER< 0:0, respectively.  These special values are always
            # associated with the wave period, irrespective of the value of IPERIN and the corresponding
            # interpretation of the positive elements of the array PER


            # execute the HAMS analysis
            ph.run_hams(meshDir)

            # read the HAMS WAMIT-style output files
            addedMass, damping, w1 = ph.read_wamit1(os.path.join(meshDir,'Output','Wamit_format','Buoy.1'), TFlag=True)  # first two entries in frequency dimension are expected to be zero-frequency then infinite frequency
            M, P, R, I, w3, heads  = ph.read_wamit3(os.path.join(meshDir,'Output','Wamit_format','Buoy.3'), TFlag=True)   
            
            # process headings and sort frequencies
            self.BEM_headings = np.array(heads)%(360)  # save headings in range of 0-360 [deg]            # interpole to the frequencies RAFT is using
            '''
            isort1 = np.argsort(w1)  # indices that will sort the radiation results
            isort3 = np.argsort(w3)  # indices that will sort the diffraction results
            w1        = np.take(w1, isort1)
            w3        = np.take(w3, isort3)
            addedMass = np.take(addedMass, isort1, axis=2)
            damping   = np.take(damping  , isort1, axis=2)
            M         = np.take(M        , isort3, axis=2)
            P         = np.take(P        , isort3, axis=2)
            R         = np.take(R        , isort3, axis=2)
            I         = np.take(I        , isort3, axis=2)
            '''
            # interpolate to RAFT model frequencies
            # zero frequency values are being stacked on to give smooth results if the requested frequency is below what's available from HAMS
            addedMassInterp = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([addedMass[:,:,2:], addedMass[:,:,0]]), assume_sorted=False, axis=2)(self.w)
            dampingInterp   = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([  damping[:,:,2:], np.zeros([6,6]) ]), assume_sorted=False, axis=2)(self.w)
            fExRealInterp   = interp1d(np.hstack([w3,      0.0]), np.dstack([       R, np.zeros([len(heads),6]) ]), assume_sorted=False, axis=2)(self.w)
            fExImagInterp   = interp1d(np.hstack([w3,      0.0]), np.dstack([       I, np.zeros([len(heads),6]) ]), assume_sorted=False, axis=2)(self.w)
            # note: fEx tensors are sized according to [nHeadings, 6 DOFs, frequencies]
            
            # copy results over to the FOWT's coefficient arrays
            self.A_BEM = self.rho_water * addedMassInterp
            self.B_BEM = self.rho_water * dampingInterp                                 
            X_BEM_temp = self.rho_water * self.g * (fExRealInterp + 1j*fExImagInterp)
            
            
            # transform excitation coefficients so that DOFs are always relative to incident wave heading rather than global frame
            self.X_BEM = np.zeros_like(X_BEM_temp)
            
            for ih in range(len(heads)):
                sin_heading = np.sin(np.radians(heads[ih]))
                cos_heading = np.cos(np.radians(heads[ih]))
                
                self.X_BEM[ih,0,:] =  cos_heading * X_BEM_temp[ih,0,:] + sin_heading * X_BEM_temp[ih,1,:]
                self.X_BEM[ih,1,:] = -sin_heading * X_BEM_temp[ih,0,:] + cos_heading * X_BEM_temp[ih,1,:]
                self.X_BEM[ih,2,:] = X_BEM_temp[ih,2,:]
                self.X_BEM[ih,3,:] =  cos_heading * X_BEM_temp[ih,3,:] + sin_heading * X_BEM_temp[ih,4,:]
                self.X_BEM[ih,4,:] = -sin_heading * X_BEM_temp[ih,3,:] + cos_heading * X_BEM_temp[ih,4,:]
                self.X_BEM[ih,5,:] = X_BEM_temp[ih,5,:]
                
            
            # HAMS results error checks  >>> any more we should have? <<<
            if np.isnan(self.A_BEM).any():
                #print("NaN values detected in HAMS calculations for added mass. Check the geometry.")
                #breakpoint()
                raise Exception("NaN values detected in HAMS calculations for added mass. Check the geometry.")
            if np.isnan(self.B_BEM).any():
                #print("NaN values detected in HAMS calculations for damping. Check the geometry.")
                #breakpoint()
                raise Exception("NaN values detected in HAMS calculations for damping. Check the geometry.")
            if np.isnan(self.X_BEM).any():
                #print("NaN values detected in HAMS calculations for excitation. Check the geometry.")
                #breakpoint()
                raise Exception("NaN values detected in HAMS calculations for excitation. Check the geometry.")

            # TODO: add support for multiple wave headings <<<
            # note: RAFT will only be using finite-frequency potential flow coefficients


    def calcTurbineConstants(self, case, ptfm_pitch=0):
        '''This computes turbine linear terms (excluding hydrodynamic added 
        mass and inertial excitation, which are handled by getHydroConstants).
        
        case
            dictionary of case information
        ptfm_pitch
            mean pitch angle of the platform [rad]

        '''

        #self.rotor.runCCBlade(case['wind_speed'], ptfm_pitch=ptfm_pitch, yaw_misalign=case['yaw_misalign'])
        #print(case)
        turbine_heading = getFromDict(case, 'turbine_heading', shape=0, dtype = float, default=0.0)  # [deg]
        turbine_status  = getFromDict(case, 'turbine_status', shape=0, dtype=str, default='operating')

        # initialize arrays (can remain zero if aerodynamics are disabled)
        self.A_aero  = np.zeros([6,6,self.nw,self.nrotors])                      # frequency-dependent aero-servo added mass matrix
        self.B_aero  = np.zeros([6,6,self.nw,self.nrotors])                      # frequency-dependent aero-servo damping matrix
        self.f_aero  = np.zeros([6,  self.nw,self.nrotors], dtype=complex)       # dynamice excitation force and moment amplitude spectra
        self.f_aero0 = np.zeros([6,          self.nrotors])                      # mean aerodynamic forces and moments
        #todo: reorder above so that w is last index <<<
        
        self.B_gyro  = np.zeros([6,6,self.nrotors])  # rotor gyroscopic damping matrix
        
        if turbine_status == 'operating':
            
            for ir in range(self.nrotors):
                # only compute the aerodynamics if enabled and windspeed is nonzero

                if self.rotorList[ir].Zhub < 0:
                    current = True
                    speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
                else:
                    current = False
                    speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
                
                if self.rotorList[ir].aeroServoMod > 0 and speed > 0.0:
                
                    # Get mean aero forces and fore-aft coefficients 
                    # Note: these are about hub coordinate in global orientation.
                    f_aero0, f_aero, a_aero, b_aero = self.rotorList[ir].calcAero(case, current=current)  # get values about hub
                    
                    # hub reference frame relative to PRP <<<<<<<<<<<<<<<<<
                    rHub = self.rotorList[ir].rHub   # TODO: change the loop to use enumerate to streamline this section, and add mean position support <<<
                    
                    # convert coefficients to platform reference frame and populate tensor slice for this rotor
                    for iw in range(self.nw):
                        self.A_aero[:,:,iw,ir] = translateMatrix6to6DOF(a_aero[:,:,iw], rHub)
                        self.B_aero[:,:,iw,ir] = translateMatrix6to6DOF(b_aero[:,:,iw], rHub)
                    
                    # convert forces to platform reference frame
                    self.f_aero0[:,ir] = transformForce(f_aero0, offset=rHub) # mean forces and moments
                    
                    for iw in range(self.nw):
                        self.f_aero[:,iw,ir] = transformForce(f_aero[:,iw], offset=rHub) # excitation
                    
                    # calculate cavitation of the rotor (platform motions should already be accounted for in the CCBlade object after running calcAero)
                    # if rotor is submerged...
                    # cav = self.rotorList[ir].calcCavitation(case)  # TO-DO: wire this to be a result/output, then uncomment <<<
                    
                    # ----- calculate rotor gyroscopic effects -----
                    # rotor speed [rpm]
                    Omega_rpm = np.interp(speed, self.rotorList[ir].Uhub, self.rotorList[ir].Omega_rpm)
                    
                    rotMat = RotFrm2Vect(np.array([1,0,0]), self.rotorList[ir].q) # rotation matrix from global to rotor axis
                    Omega_rotor = np.matmul(rotMat, [Omega_rpm*2*np.pi/60, 0, 0]) # rotor angular velocity vector
                    
                    # rotating inertia vector [kg-m^2 * rad/s = N-m-s]
                    IO_rotor = self.rotorList[ir].I_drivetrain * Omega_rotor
                    
                    GyroDampingMatrix = getH(IO_rotor)
                    
                    self.B_gyro[3:, 3:, ir] = GyroDampingMatrix  
                    
                    # note, gyroscopic effect is purely rotational so no translation adjustment needed

        else:
            print(f"Warning: turbine status is '{turbine_status}' so rotor fluid loads are neglected.")


    def calcHydroConstants(self):
        '''Compute FOWT hydrodynamic added mass matrix and member-level
        inertial excitation coefficients.'''


        rho = self.rho_water
        g   = self.g
        
        # --------------------- get constant hydrodynamic values along each member -----------------------------

        self.A_hydro_morison = np.zeros([6,6])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]

        # loop through each member

        for i,mem in enumerate(self.memberList):
            '''

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections
            
            # find the proper member node positions (mem.r) if this memberList is a bladeMemberList for underwater rotors. Otherwise, skip
            if mem.type==3 and Rotor is not None:
                
                # save specific rotor variables and original positions of the blade member nodes
                rotor = Rotor
                rOG = [mem.rA0, mem.rB0]

                # (without making any changes to the rest of the hydrodynamics calculations below)
                # the input memberList is the rotor.bladeMemberList multiplied by nBlades (e.g. if len(rotor.bladeMemberList)=10, then this len(memberList)=30 for 3 blade rotor)
                # adjust the heading/orientation of the each section of blade members by the corresponding heading in rotor.headings based on where in the list you are
                j = i // int(len(memberList)/rotor.nBlades)     # i (from 0 to 30) // 10, which results in either j = 1, 2, or 3, for each blade heading

                # find the end nodes of the blade member about the global coordinates (e.g,, if rA_OG = [0,0,0] and rB_OG=[0,1,0], and heading=90, then rA=[0,0,0] and rB=[0,0,1])
                rUpdated = rotor.getBladeMemberPositions(rotor.headings[j], rOG)
                mem.rA0 = rUpdated[0]
                mem.rB0 = rUpdated[-1]
                
                # apply a blade pitch angle to every blade member if applicable
                mem.gamma = mem.gamma + dgamma

                # run calcOrientation to ensure the p1 and p2 axes of the members are oriented the way they should be
                mem.setPosition()

                # >>>>>>> TODO (for added mass of underwater rotors) <<<<<<<<<<
                # - Do we need to adjust any other methods in raft_fowt to account for added mass?
                # - Do we need to make any changes to the pyHAMS code to account for rotor added mass?
                # - How should we adjust the inclusion of member end effects of added mass knowing blade members are attached on ends to each other?
            

            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:
                    
                    # only compute inertial loads and added mass for members that aren't modeled with potential flow
                    if mem.potMod==False:

                        # interpolate coefficients for the current strip
                        Ca_q   = np.interp( mem.ls[il], mem.stations, mem.Ca_q  )
                        Ca_p1  = np.interp( mem.ls[il], mem.stations, mem.Ca_p1 )
                        Ca_p2  = np.interp( mem.ls[il], mem.stations, mem.Ca_p2 )
                        Ca_End = np.interp( mem.ls[il], mem.stations, mem.Ca_End)


                        # ----- compute side effects (transverse only) -----

                        if circ:
                            v_i = 0.25*np.pi*mem.ds[il]**2*mem.dls[il]
                        else:
                            v_i = mem.ds[il,0]*mem.ds[il,1]*mem.dls[il]  # member volume assigned to this node

                        if mem.r[il,2] + 0.5*mem.dls[il] > 0:    # if member extends out of water              # <<< may want a better appraoch for this...
                            v_i = v_i * (0.5*mem.dls[il] - mem.r[il,2]) / mem.dls[il]  # scale volume by the portion that is under water
                        
                        # added mass (axial term explicitly excluded here - we aren't dealing with chains)
                        Amat_sides = rho*v_i *( Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat )  # local added mass matrix
                        
                        # inertial excitation - Froude-Krylov  (axial term explicitly excluded here - we aren't dealing with chains)
                        Imat_sides = rho*v_i *(  (1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat ) # local inertial excitation matrix (note: the 1 is the Cp, dynamic pressure, term)
                        

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
                        Amat_end = rho*v_i * Ca_End*mem.qMat                             # local added mass matrix
                        
                        # inertial excitation
                        Imat_end = rho*v_i * Ca_End*mem.qMat                         # local inertial excitation matrix (note, there is no 1 added to Ca_End because dynamic pressure is handled separately) 
                        

                        # ----- sum up side and end added mass and inertial excitation coefficient matrices ------
                        mem.Amat[il,:,:] = Amat_sides + Amat_end
                        mem.Imat[il,:,:] = Imat_sides + Imat_end
                        mem.a_i[il] = a_i
                        
                        
                        # add to global added mass matrix for Morison members, which considers the mean offsets and is relative to the RPP with global orientation
                        self.A_hydro_morison += translateMatrix3to6DOF(mem.Amat[il,:,:], mem.r[il,:] - self.r6[:3])    
            

            # reset the member.r variable if underwater blade members were used
            if mem.type==3 and Rotor is not None:
                mem.rA0 = rOG[0]
                mem.rB0 = rOG[-1]
            '''
            # get member added mass matrix about PRP (also saves each member's inertial excitation coefficients)
            A_hydro_i = mem.calcHydroConstants(r_ref=self.r6[:3])  
            
            self.A_hydro_morison += A_hydro_i  # add to FOWT added mass matrix
    
    
        # ----- Get hydrodynamic contributions from any underwater rotors ------
        for i, rot in enumerate(self.rotorList):
            
            # compute rotor hydro added mass/inertia properties
            A_hydro_i, I_hydro_i = rot.calcHydroConstants(rho=self.rho_water, g=self.g)
            
            # make relative to PRP and add to system added mass matrix
            self.A_hydro_morison += translateMatrix6to6DOF(A_hydro_i, rot.r3-self.r6[:3]) 
    

    def calcHydroExcitation(self, case, memberList=[], dgamma=0):
        '''This computes the wave kinematics and linear excitation for a given case.
        It calculates and F_BEM and F_hydro_iner, each with dimensions [wave headings, DOFs, frequencies].
        '''
        
        # ----- set up sea state -----
        
        # JvS made support for a second set of case wave info for a different heading. Instead, to generalize,
        # we will allow the case wave entries to be lists, with size nHeadings.
        
        if np.isscalar(case['wave_heading']):  # deal with the typical case of just one set of waves specified
            self.nWaves = 1
        else:
            self.nWaves = len(case['wave_heading'])
        
        # ensure our inputs are all arrays and of size nWaves (any scalar inputs will be tiled)
        case['wave_heading'] = getFromDict(case, 'wave_heading'  , shape=self.nWaves, dtype=float, default=0)
        case['wave_spectrum']= getFromDict(case, 'wave_spectrum' , shape=self.nWaves, dtype=str, default='JONSWAP')
        case['wave_period']  = getFromDict(case, 'wave_period'   , shape=self.nWaves, dtype=float)
        case['wave_height']  = getFromDict(case, 'wave_height'   , shape=self.nWaves, dtype=float)
        case['wave_gamma']   = getFromDict(case, 'wave_gamma'    , shape=self.nWaves, dtype=float, default=0)
        
        
        self.beta = case['wave_heading']   # [rad] array of wave headings
        self.zeta = np.zeros([self.nWaves,self.nw], dtype=complex)
        
        # make wave spectrum for each heading
        # We are actually losing the phase by computing the 2nd order hydrodynamic forces this way, as the amplitudes will always be real
        self.Fhydro_2nd = np.zeros([self.nWaves, self.nDOF, self.nw], dtype=complex) 
        self.Fhydro_2nd_mean = np.zeros([self.nWaves, self.nDOF])
        for ih in range(self.nWaves):
            if case['wave_spectrum'][ih] == 'unit':
                #self.zeta[ih,:] = np.tile(1, self.nw)
                S         = np.tile(1, self.nw)
                self.zeta[ih,:] = np.sqrt(2*S*self.dw)
            elif case['wave_spectrum'][ih] == 'JONSWAP':
                S = JONSWAP(self.w, case['wave_height'][ih], case['wave_period'][ih], Gamma=case['wave_gamma'][ih])        
                self.zeta[ih,:] = np.sqrt(2*S*self.dw)    # wave elevation amplitudes (these are easiest to use)
            elif case['wave_spectrum'][ih] in ['none','still']:
                self.zeta[ih,:] = np.zeros(self.nw)        
                S = np.zeros(self.nw)        
            else:
                raise ValueError(f"Wave spectrum input '{case['wave_spectrum'][ih]}' not recognized.")
            
            if self.secondOrderWaveMod == 1:
                self.Fhydro_2nd_mean[ih, :], self.Fhydro_2nd[ih, :, :] = self.calcHydroForce_2ndOrd(case['wave_heading'][ih], S)
            

        #print(f"significant wave height:  {4*np.sqrt(np.sum(S)*self.dw):5.2f} = {4*getRMS(self.zeta, self.dw):5.2f}") # << temporary <<<

        # TODO: consider current and viscous drift <<<
        
        # resize members' wave kinematics arrays for this case's sea states
        for i,mem in enumerate(memberList):
            mem.u    = np.zeros([self.nWaves, mem.ns, 3, self.nw], dtype=complex)
            mem.ud   = np.zeros([self.nWaves, mem.ns, 3, self.nw], dtype=complex)
            mem.pDyn = np.zeros([self.nWaves, mem.ns,    self.nw], dtype=complex)
        
        # also set up wave kinematics arrays for the rotors
        for i,rot in enumerate(self.rotorList):
            rot.u    = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.ud   = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.pDyn = np.zeros([self.nWaves,    self.nw], dtype=complex)
            
        # ----- calculate potential-flow wave excitation force -----

        self.F_BEM = np.zeros([self.nWaves,6,self.nw], dtype=complex)
        self.F_hydro_iner = np.zeros([self.nWaves, 6, self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]

        # BEM-based wave excitation force on platform for each wave heading (will be zero if HAMS isn't run). This includes heading interpolation.
        if self.potMod:

            for ih in range(self.nWaves):
                
                # phase offset due to FOWT position in array
                phase_offset = np.exp(-1j*self.k* ( self.x_ref*np.cos(case['wave_heading'][ih]) 
                                                  + self.y_ref*np.sin(case['wave_heading'][ih]) ) )

                beta = np.degrees(self.beta[ih])%360  # heading in range of 0-360 [deg]
                
                headings = self.BEM_headings      # the headings of the available BEM data [deg]
                nhs = len(headings)
                # find interpolation indices and factors, of format y* = y[iout[0]]*fout[0] + y[iout[1]]*fout[1]
                
                # this code needs checking!
                if (beta <= headings[0]):         # when wave heading (beta) is before first BEM heading
                    hlast = headings[-1] - 360    # make the last BEM heading negative so that it's before beta
                    i1 = nhs-1
                    i2 = 0
                    f2 = (beta - hlast)/( headings[0] - hlast )
                
                elif (beta >= headings[nhs-1]):   # when wave heading (beta) is after last BEM heading
                    hfirst = headings[0] + 360    # make the fisrt BEM heading positive so that it's after beta
                    i1 = nhs-1  
                    i2 = 0
                    f2 = (beta - headings[-1])/( hfirst - headings[-1] )
                
                else:                             # normal case
                    for i in range(nhs-1):
                        if (headings[i+1] > beta):
                            i1 = i
                            i2 = i+1 
                            f2 = (beta - headings[i] )/( headings[i+1] - headings[i] )
                            break
                
                f1 = 1.0 - f2
                      
                # interpolate excitation coefficients
                X_prime = self.X_BEM[i1,:,:]*f1 + self.X_BEM[i2,:,:]*f2  # excitation coefficients (relative to wave direction)            
                
                #self.beta[ih], self.BEM_headings, self.X_BEM[, period=360) * 

                # rotate back to global frame
                sin_beta = np.sin(self.beta[ih])   #  check that I'm right in assuming HAMS outputs degrees
                cos_beta = np.cos(self.beta[ih])
                
                self.X_BEM[ih,0,:] = X_prime[0,:]*cos_beta - X_prime[1,:]*sin_beta
                self.X_BEM[ih,1,:] = X_prime[0,:]*sin_beta + X_prime[1,:]*cos_beta
                self.X_BEM[ih,2,:] = X_prime[2,:]
                self.X_BEM[ih,3,:] = X_prime[3,:]*cos_beta - X_prime[4,:]*sin_beta
                self.X_BEM[ih,4,:] = X_prime[3,:]*sin_beta + X_prime[4,:]*cos_beta
                self.X_BEM[ih,5,:] = X_prime[5,:]
                
                # multiply excitation coefficients by wave elevation to get excitation forces and moments for this wave heading
                self.F_BEM[ih,:,:] = self.X_BEM[ih,:,:] * self.zeta[ih,:] * phase_offset


        # ----- strip-theory wave excitation force -----
        # loop through each member to compute strip-theory contributions
        for i,mem in enumerate(memberList):
            
            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections


            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # get wave kinematics spectra given a certain wave spectrum and location
                    # for each wave direction, calculate this node's frequency-dependent wave velocity, acceleration, and dynamic presure
                    for ih in range(self.nWaves):
                        mem.u[ih,il,:,:], mem.ud[ih,il,:,:], mem.pDyn[ih,il,:] = getWaveKin(self.zeta[ih,:], self.beta[ih], 
                                                                                 self.w, self.k, self.depth, mem.r[il,:], self.nw)
                    # Note: the above wave kinematics account for phasing due to the FOWT's mean offset position in the array
                    
                    # now sum across wave headings and store these values - not accurate for multidirectional, 
                    # but versatile if we want to visualize something. <<< not currently used
                    #mem.u[ il,:,:] = np.sum(u   , axis=0)
                    #mem.ud[il,:,:] = np.sum(ud  , axis=0)
                    #mem.pDyn[il,:] = np.sum(pDyn, axis=0)     


                    # calculate the linear excitation forces on this node for each wave heading and frequency
                    for ih in range(self.nWaves):
                        for i in range(self.nw):

                            # add dynamic pressure - positive with q if end A - determined by sign of a_i
                            F_exc_iner_temp = np.matmul(mem.Imat[il,:,:], mem.ud[ih,il,:,i]) + mem.pDyn[ih,il,i]*mem.a_i[il]*mem.q 
                            
                            # add the excitation complex amplitude for this heading and frequency to the global excitation vector
                            self.F_hydro_iner[ih,:,i] += translateForce3to6DOF(F_exc_iner_temp, mem.r[il,:] - self.r6[:3]) # (about PRP)


        # ----- inertial excitation on rotor(s) -----
        
        for i, rot in enumerate(self.rotorList):
            if rot.Zhub < 0:  # if submerged

                # get wave kinematics spectra given a certain wave spectrum and location
                # for each wave direction, calculate the hub wave kinematics
                for ih in range(self.nWaves):
                    rot.u[ih,:,:], rot.ud[ih,:,:], rot.pDyn[ih,:] = getWaveKin(self.zeta[ih,:], self.beta[ih], 
                                                                    self.w, self.k, self.depth, rot.r3, self.nw)
                # Note: the above wave kinematics account for phasing due to the FOWT's mean offset position in the array
                
                # Rotate rotor inertial excitation matrix to align with global
                #angles = self.r6[3:] + np.array([0, np.atan2(rot.axis[2], rot.axis[0]),
                #                                    np.atan2(rot.axis[1], rot.axis[0])])
                #Rmat = rotationMatrix(*angles)  # rotation matrix for rotor+fowt orientation
                Rmat = RotFrm2Vect( np.array([1,0,0]), rot.q)
                I_hydro = rotateMatrix6(rot.I_hydro, Rmat)  # Should check. Only first 3 columns of I_hydro have meaning.

                # compute force across frequency range
                for i in range(self.nw):
                    f3 = np.matmul( I_hydro[:3,:3], rot.ud[ih,:,i] )  # Forces due to acceleration
                    f6 = translateForce3to6DOF(f3, rot.r3 - self.r6[:3])  # Translate to about PRP (induces some moments)
                    f6[3:] += np.matmul( I_hydro[3:,:3], rot.ud[ih,:,i] )  # Add moments due to acceleration

                    self.F_hydro_iner[ih,:,i] += f6
                

    def calcHydroLinearization(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent 
        linearized coefficients, including the system linearized drag damping matrix. For the 
        drag-based excitation, call calcDragExcitation after this method.
        
        Currently hard-coded to only consider the first seastate/heading.

        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes [m, rad]

        '''

        rho = self.rho_water
        g   = self.g

        # The linearized coefficients to be calculated

        B_hydro_drag = np.zeros([6,6])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]

        ih = 0  # we will only consider the first sea state in this linearization process

        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # get node complex velocity spectrum based on platform motion's and relative position from PRP
                # node displacement, velocity, and acceleration (each [3 x nw])
                drnode, vnode, anode = getKinematics(mem.r[il,:], Xi, self.w)

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # interpolate coefficients for the current strip
                    Cd_q   = np.interp( mem.ls[il], mem.stations, mem.Cd_q  )
                    Cd_p1  = np.interp( mem.ls[il], mem.stations, mem.Cd_p1 )
                    Cd_p2  = np.interp( mem.ls[il], mem.stations, mem.Cd_p2 )
                    Cd_End = np.interp( mem.ls[il], mem.stations, mem.Cd_End)


                    # ----- compute side effects ------------------------

                    # member acting area assigned to this node in each direction
                    a_i_q  = np.pi*mem.ds[il]*mem.dls[il]  if circ else  2*(mem.ds[il,0]+mem.ds[il,0])*mem.dls[il]
                    a_i_p1 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,0]      *mem.dls[il]
                    a_i_p2 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,1]      *mem.dls[il]

                    # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
                    vrel = mem.u[ih,il,:] - vnode

                    # break out velocity components in each direction relative to member orientation [nw]
                    vrel_q  = np.sum(vrel*mem.q[ :,None], axis=0)*mem.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                    vrel_p1 = np.sum(vrel*mem.p1[:,None], axis=0)*mem.p1[:,None]
                    vrel_p2 = np.sum(vrel*mem.p2[:,None], axis=0)*mem.p2[:,None]
                    
                    # get RMS of relative velocity component magnitudes (real-valued)
                    vRMS_q  = getRMS(vrel_q)
                    vRMS_p1 = getRMS(vrel_p1)
                    vRMS_p2 = getRMS(vrel_p2)
                    
                    # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
                    Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * a_i_q  * Cd_q
                    Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * a_i_p1 * Cd_p1
                    Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * a_i_p2 * Cd_p2

                    # form damping matrix for the node based on linearized drag coefficients
                    Bmat_sides = Bprime_q*mem.qMat + Bprime_p1*mem.p1Mat + Bprime_p2*mem.p2Mat 


                    # ----- add end/axial effects for added mass, drag, and excitation including dynamic pressure ------
                    # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))

                    Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*a_i*Cd_End
                    #print(f"  {a_i:5.2f}  {vRMS_q:5.2f}  {Bprime_End:5.2f}")

                    # form damping matrix for the node based on linearized drag coefficients
                    Bmat_end = Bprime_End*mem.qMat                                       #


                    # ----- sum up side and end damping matrices ------
                    
                    mem.Bmat[il,:,:] = Bmat_sides + Bmat_end   # store in Member object to be called later to get drag excitation for each wave heading

                    B_hydro_drag += translateMatrix3to6DOF(mem.Bmat[il,:,:], mem.r[il,:] - self.r6[:3])   # add to global damping matrix for Morison members


                    # ----- calculate wave drag excitation (this may be recalculated later) -----

                    for i in range(self.nw):

                        mem.F_exc_drag[il,:,i] = np.matmul(mem.Bmat[il,:,:], mem.u[ih,il,:,i])   # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]

                        F_hydro_drag[:,i] += translateForce3to6DOF(mem.F_exc_drag[il,:,i], mem.r[il,:] - self.r6[:3])   # add to global excitation vector (frequency dependent)
                    
                    '''
                    for i in range(self.nw):                                         # for each wave frequency...

                        F_exc_drag_temp = np.matmul(Bmat, mem.u[ih,il,:,i])             # local drag excitation force complex amplitude in x,y,z

                        mem.F_exc_drag[il,:,i] += F_exc_drag_temp                    # add to stored member force vector

                        F_hydro_drag[:,i] += translateForce3to6DOF(F_exc_drag_temp, mem.r[il,:]) # add to global excitation vector (frequency dependent)
                    '''

        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics or for visualization
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag
    
    
    
    def calcDragExcitation(self, ih):
        '''Calculated linearized viscous drag excitation for a given sea state (wave heading). calcLinearizedTerms should be called first.

        ih : int
            index of wave case being evaluated here

        '''

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]
     
        for mem in self.memberList:   # loop through each member
            for il in range(mem.ns):  # loop through each node of the member
                if mem.r[il,2] < 0:   # only process hydrodynamics if this node is submerged
                    for i in range(self.nw):
                        
                        # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]
                        mem.F_exc_drag[il,:,i] = np.matmul(mem.Bmat[il,:,:], mem.u[ih,il,:,i])   
                        
                        # add to global excitation vector (frequency dependent)
                        F_hydro_drag[:,i] += translateForce3to6DOF(mem.F_exc_drag[il,:,i], mem.r[il,:] - self.r6[:3])   

        self.F_hydro_drag = F_hydro_drag

        return F_hydro_drag
    
    
    
    def calcCurrentLoads(self, case):
        '''method to calculate the "static" current loads on each member and save as a current force
        Uses a simple power law relationship to calculate the current velocity as a function of member node depth'''

        rho = self.rho_water
        g   = self.g

        D_hydro = np.zeros(6)      # create variable to hold the total drag force

        # extract current variables out of the case dictionary
        speed = getFromDict(case, 'current_speed', shape=0, default=0.0)
        heading = getFromDict(case, 'current_heading', shape=0, default=0)

        
        Zref = 0.0  # reference z elevation for current profile (at the sea surface by default) (reference height set to submerged rotor hub depth if rotor is submerged)
        for ti in range(self.nrotors):
            if self.rotorList[ti].Zhub < 0:     # If there is a submerged rotor,
                Zref = self.rotorList[ti].Zhub  # use it for the reference current height.

        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # calculate current velocity as a function of node depth [x,y,z] (assumes no vertical current velocity)
                    v = speed * (((self.depth + Zref) - abs(mem.r[il,2]))/(self.depth + Zref))**self.shearExp_water
                    vcur = np.array([v*np.cos(np.deg2rad(heading)), v*np.sin(np.deg2rad(heading)), 0])

                    # interpolate coefficients for the current strip
                    Cd_q   = np.interp( mem.ls[il], mem.stations, mem.Cd_q  )
                    Cd_p1  = np.interp( mem.ls[il], mem.stations, mem.Cd_p1 )
                    Cd_p2  = np.interp( mem.ls[il], mem.stations, mem.Cd_p2 )
                    Cd_End = np.interp( mem.ls[il], mem.stations, mem.Cd_End)

                    # current (relative) velocity over node (no complex numbers bc not function of frequency)
                    vrel = np.array(vcur)
                    # break out velocity components in each direction relative to member orientation
                    vrel_q  = np.sum(vrel*mem.q[:] )*mem.q[:] 
                    vrel_p1 = np.sum(vrel*mem.p1[:])*mem.p1[:]
                    vrel_p2 = np.sum(vrel*mem.p2[:])*mem.p2[:]
                    
                    # ----- compute side effects ------------------------

                    # member acting area assigned to this node in each direction
                    a_i_q  = np.pi*mem.ds[il]*mem.dls[il]  if circ else  2*(mem.ds[il,0]+mem.ds[il,0])*mem.dls[il]
                    a_i_p1 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,0]      *mem.dls[il]
                    a_i_p2 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,1]      *mem.dls[il]

                    # calculate drag force wrt to each orientation using simple Morison's drag equation
                    Dq = 0.5 * rho * a_i_q * Cd_q * np.linalg.norm(vrel_q) * vrel_q
                    Dp1 = 0.5 * rho * a_i_p1 * Cd_p1 * np.linalg.norm(vrel_p1) * vrel_p1
                    Dp2 = 0.5 * rho * a_i_p2 * Cd_p2 * np.linalg.norm(vrel_p2) * vrel_p2
                    
                    # ----- end/axial effects drag ------

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i_End = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i_End = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))
                    
                    Dq_End = 0.5 * rho * a_i_End * Cd_End * np.linalg.norm(vrel_q) * vrel_q

                    # ----- sum forces and add to total mean drag load about PRP ------
                    D = Dq + Dp1 + Dp2 + Dq_End     # sum drag forces at node in member's local orientation frame

                    D_hydro += translateForce3to6DOF(D, mem.r[il,:] - self.r6[:3])  # sum as forces and moments about PRP
                    
        self.D_hydro = D_hydro  # save hydro drag forces/moments to FOWT for later access

        return D_hydro


    def readQTF(self, flPath):
        data = np.loadtxt(flPath)
        ULEN = 1 # For now, assume that ULEN = 1
        data[:,0:2] = 2.*np.pi/data[:,0:2] # Input is assumed to be as a function of wave period

        # Consider only unidirectional QTFs
        if not (data[:,2] == data[:,3]).all():
            raise ValueError("Only unidirectional QTFs are supported for now.")
        self.heads_2nd = np.sort(np.unique(data[:,2]))
        nheads = len(self.heads_2nd)

        # Both frequency vectors should contain the same frequencies,
        # but they are not necessarily the same as self.w 
        self.w1_2nd = np.unique(data[:,0])  
        self.w2_2nd = np.unique(data[:,1])
        nw1 = len(self.w1_2nd)
        nw2 = len(self.w2_2nd)
        if not (self.w1_2nd==self.w2_2nd).all():
            raise ValueError("Both frequency columns in the input QTF must contain the same values.")
        
        self.qtf = np.zeros([nw1, nw2, nheads, self.nDOF], dtype=complex)                
        for row in data:
            indw1, = np.where(self.w1_2nd==row[0]) # index for first frequency
            indw2, = np.where(self.w2_2nd==row[1]) # index for second frequency
            indhead, = np.where(self.heads_2nd==row[2]) # index for heading
            indDOF = round(row[4]-1) # index for degree of freedom. Needs to be an int. -1 is due to being from 1 to 6 in the input file

            # Factor for dimensionalization (except for wave amplitudes)
            factor = self.rho_water * self.g * ULEN
            if indDOF >= 3:
                factor *= ULEN

            self.qtf[indw1[0], indw2[0], indhead[0], indDOF] = factor*(row[7]+1j*row[8])

            # Fill the other half of the matrix (which is equal to the conjugate of its symmetric element)
            if not indw1[0] == indw2[0]:
                self.qtf[indw2[0], indw1[0], indhead[0], indDOF] = factor*(row[7]-1j*row[8])
        

        # plt.matshow(np.squeeze(np.abs(self.qtf[:,:,0,0])))    


    # Change that for a spectrum
    def calcHydroForce_2ndOrd(self, beta, S0):
        f = np.zeros([self.nDOF, self.nw], dtype=complex)
        nw1 = len(self.w1_2nd)

        # Resample wave spectrum (the input is assumed to be in rad/s)
        S = np.interp(self.w1_2nd, self.w, S0, left=0, right=0) 

        # Interpolate for the wave incidence        
        if beta < self.heads_2nd[0]:
            print(f"Warning in calcHydroForce_2ndOrd: angle {beta} is less than the minimum incidence angle in the QTF. An incidence of {self.heads_2nd[0]} will be considered for 2nd order loads.")

        if beta > self.heads_2nd[-1]:
            print(f"Warning in calcHydroForce_2ndOrd: angle {beta} is more than the maximum incidence angle in the QTF. An incidence of {self.heads_2nd[-1]} will be considered for 2nd order loads.")

        if len(self.heads_2nd)==1:
            qtf_interpBeta = self.qtf[:,:,0,:]
        
        else:
            qtf_interpBeta = interp1d(self.heads_2nd, self.qtf, assume_sorted=True, axis=2, bounds_error=False, fill_value=(self.qtf[:,:,0,:], self.qtf[:,:,-1,:]))(beta)
        
        # The number of difference frequencies is the same as the number of frequencies, but starting from frequency mu=0.        
        mu = self.w1_2nd - self.w1_2nd[0]
        Sf = np.zeros([self.nDOF, nw1]) # Second-order force spectrum
        Sf_interp = np.zeros([self.nDOF, self.nw])
        for idof in range(0,self.nDOF):                        
            for imu in range(0, nw1): # Loop the difference frequencies
                Saux = np.zeros(nw1)
                Saux[0:nw1-imu] = S[imu:] # Auxiliar wave spectrum that is dislocated in frequency. See the definition of second-order force spectrum
                Qaux = np.zeros(nw1, dtype=complex)

                Qaux[0:nw1-imu] = np.diag(np.squeeze(qtf_interpBeta[:,:,idof]), -imu) # Sum only the lower half of the QTF
                Sf[idof, imu] = 8 * np.sum(S*Saux*np.abs(Qaux)**2) * (self.w1_2nd[1]-self.w1_2nd[0])
                        

            # Interpolate the force spectrum to the same resolution as the original wave spectrum            
            Sf_interp[idof, :] = np.interp(self.mu_2nd, mu, Sf[idof,:], left=0, right=0)        

        # # TODO: Those lines were here for debugging. Need to delete them after this feature is fully implemented.
        f = np.sqrt(2*Sf_interp*self.dw)
        f_mean = np.zeros([self.nDOF])
        f_mean[:] = f[:,0]
        f[:, 0:-1] = f[:, 1:] # Displace f by one frequency so that it aligns with the frequency vector
        f[:, -1] = 0

        # with open('examples/Sf_2nd.txt', 'w') as file:
        #     mu_interp = self.w - self.w[0]
        #     for w, Srow in zip(mu_interp, Sf_interp.T):
        #         file.write(f'{w:.5f} {Srow[0]:.5f} {Srow[1]:.5f} {Srow[2]:.5f} {Srow[3]:.5f} {Srow[4]:.5f} {Srow[5]:.5f}\n')

        # with open('examples/f_2nd.txt', 'w') as file:
        #     for w, frow in zip(mu_interp, f.T):
        #         file.write(f'{w:.5f} {frow[0]:.5f} {frow[1]:.5f} {frow[2]:.5f} {frow[3]:.5f} {frow[4]:.5f} {frow[5]:.5f}\n')
        return f_mean, f


    def saveTurbineOutputs(self, results, case):
        '''Calculate and store output metrics of the FOWT response at the current load case.
        Note that the FOWT offset, motions, load case info, etc. are taken from what is stored
        in the FOWT object. I.e. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   
        Load cases may have multiple sources of excitation (such as wind, wave, and another wave
        at a different heading). Results are computed by RMS summing across these excitation sources.
        '''
        
        self.Xi0 = self.r6 - np.array([self.x_ref, self.y_ref,0,0,0,0])  # FOWT's mean offset vector [m, rad]

        # platform motions
        results['surge_avg'] = self.Xi0[0]
        results['surge_std'] = getRMS(self.Xi[:,0,:]) 
        results['surge_max'] = self.Xi0[0] + 3*results['surge_std']
        results['surge_PSD'] = getPSD(self.Xi[:,0,:], self.dw)
        
        results['sway_avg'] = self.Xi0[1]
        results['sway_std'] = getRMS(self.Xi[:,1,:])
        results['sway_max'] = self.Xi0[1] + 3*results['sway_std']
        results['sway_PSD'] = getPSD(self.Xi[:,1,:], self.dw)
        
        results['heave_avg'] = self.Xi0[2]
        results['heave_std'] = getRMS(self.Xi[:,2,:])
        results['heave_max'] = self.Xi0[2] + 3*results['heave_std']
        results['heave_PSD'] = getPSD(self.Xi[:,2,:], self.dw)
        
        roll_deg = rad2deg(self.Xi[:,3,:])
        results['roll_avg'] = rad2deg(self.Xi0[3])
        results['roll_std'] = getRMS(roll_deg)
        results['roll_max']  = rad2deg(self.Xi0[3]) + 3*results['roll_std']
        results['roll_PSD'] = getPSD(roll_deg, self.dw)
        
        pitch_deg = rad2deg(self.Xi[:,4,:])
        results['pitch_avg'] = rad2deg(self.Xi0[4])
        results['pitch_std'] = getRMS(pitch_deg)
        results['pitch_max'] = rad2deg(self.Xi0[4]) + 3*results['pitch_std']
        results['pitch_PSD'] = getPSD(pitch_deg, self.dw)
        
        yaw_deg = rad2deg(self.Xi[:,5,:])
        results['yaw_avg'] = rad2deg(self.Xi0[5])
        results['yaw_std'] = getRMS(yaw_deg)
        results['yaw_max'] = rad2deg(self.Xi0[5]) + 3*results['yaw_std']
        results['yaw_PSD'] = getPSD(yaw_deg, self.dw)
        
        # ----- turbine-level mooring outputs (similar code as array-level) -----
        if self.ms:
            nLines = len(self.ms.lineList)
            T_moor_amps = np.zeros([self.nWaves+1, 2*nLines, self.nw], dtype=complex)  # mooring tension amplitudes for each excitation source and line end
            C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True, tensions=True) # get stiffness matrix and tension jacobian matrix
            T_moor = self.ms.getTensions()  # get line end mean tensions
            
            for ih in range(self.nWaves+1):
                for iw in range(self.nw):
                    T_moor_amps[ih,:,iw] = np.matmul(J_moor, self.Xi[ih,:,iw])   # FFT of mooring tensions
        
            results['Tmoor_avg'] = T_moor
            results['Tmoor_std'] = []
            results['Tmoor_max'] = []
            results['Tmoor_PSD'] =  np.zeros([ self.nw, 2*nLines])
            for iT in range(2*nLines):
                TRMS = getRMS(T_moor_amps[:,iT,:]) # estimated mooring line RMS tension [N]
                results['Tmoor_std'].append(TRMS)
                results['Tmoor_max'].append( T_moor[iT] + 3*TRMS)
                results['Tmoor_PSD'][:,iT] = (getPSD(T_moor_amps[:,iT,:], self.w[0])) # PSD in N^2/(rad/s)
            
            # log the maximum line tensions predicted by RAFT for MoorPy use
            # self.ms.saveMaxTensions(results['Tmoor_max']) 
        
        # hub fore-aft displacement amplitude and acceleration (used as an approximation in a number of outputs)
        XiHub = np.zeros([self.Xi.shape[0], self.nrotors, self.nw], dtype=complex)
        results['AxRNA_std'] = []
        results['AxRNA_PSD'] = np.zeros([self.nw, self.nrotors]) 
        results['AxRNA_avg'] = []
        results['AxRNA_max'] = []
        
        for ir in range(self.nrotors):
            XiHub[:,ir,:] = self.Xi[:,0,:] + self.hHub[ir]*self.Xi[:,4,:]  
        
            # nacelle acceleration
            results['AxRNA_std'].append(getRMS(XiHub[:,ir,:]*self.w**2))
            results['AxRNA_PSD'][:,ir] = (getPSD(XiHub[:,ir,:]*self.w**2, self.dw))
            results['AxRNA_avg'].append(abs(np.sin(self.Xi0[4])*9.81)) # @Matt check this! 
            results['AxRNA_max'].append(results['AxRNA_avg'][ir]+3*results['AxRNA_std'][ir])
            
        # tower base bending moment  >>> should three-dimensionalize this <<<
        m_turbine = np.zeros(len(self.mtower))
        zCG_turbine = np.zeros_like(m_turbine)
        zBase = np.zeros_like(m_turbine)
        hArm = np.zeros_like(m_turbine)
        
        aCG_turbine = np.zeros_like(XiHub, dtype=complex)
        ICG_turbine = np.zeros_like(m_turbine)
        M_I            = np.zeros_like(XiHub)
        M_w            = np.zeros_like(XiHub)
        M_X_aero       = np.zeros_like(XiHub)
        dynamic_moment = np.zeros_like(XiHub)
        dynamic_moment_RMS = np.zeros(self.nrotors)

        results['Mbase_avg'] = []
        results['Mbase_std'] = []
        results['Mbase_PSD'] = np.zeros([self.nw, self.nrotors])
        results['Mbase_max'] = []
        for ir in range(self.nrotors):
            # mass and moment arm >>> should three-dimensionalize <<<
            m_turbine[ir] = self.mtower[ir] + self.mRNA[ir]                        # total masses of each turbine
            zCG_turbine[ir] = (self.rCG_tow[ir][2]*self.mtower[ir]                 # CoG of each turbine
                                + self.hHub[ir]*self.mRNA[ir])/m_turbine[ir]
            zBase[ir] = self.memberList[self.nplatmems + ir].rA[2]                  # tower base elevation [m]
            hArm[ir] = zCG_turbine[ir] - zBase[ir]                                 # vertical distance from tower base to turbine CG [m]

            aCG_turbine[:,ir,:] = -self.w**2 *( self.Xi[:,0,:] + zCG_turbine[ir]*self.Xi[:,4,:] )  # fore-aft acceleration of turbine CG

            # turbine pitch moment of inertia about CG [kg-m^2]
            ICG_turbine[ir] = (translateMatrix6to6DOF(self.memberList[self.nplatmems+ir].M_struc, [0,0,-zCG_turbine[ir]])[4,4] # tower MOI about turbine CG
                        + self.mRNA[ir]*(self.hHub[ir]-zCG_turbine[ir])**2 + self.IrRNA[ir]   )                          # RNA MOI with parallel axis theorem
            # moment components and summation (all complex amplitudes)
            M_I[:,ir,:] = -m_turbine[ir]*aCG_turbine[:,ir,:]*hArm[ir] - ICG_turbine[ir]*(-self.w**2 *self.Xi[:,4,:] ) # tower base inertial reaction moment
            M_w[:,ir,:] =  m_turbine[ir]*self.g * hArm[ir]*self.Xi[:,4]                                    # tower base weight moment
            
            M_F_aero = 0.0 # <<<<self.f_aero[0,:]*(self.hHub - zBase)  # tower base moment from turbulent wind excitation  <<<<<<<<<<<<<
            
            M_X_aero[:,ir,:] = -(-self.w**2 *self.A_aero[0,0,:,ir]                                 # tower base aero reaction moment
                        + 1j*self.w *self.B_aero[0,0,:,ir] )*(self.hHub[ir] - zBase[ir])**2 *self.Xi[:,4,:]        
            dynamic_moment[:,ir,:] = M_I[:,ir,:] + M_w[:,ir,:] + M_F_aero + M_X_aero[:,ir,:]       # total tower base fore-aft bending moment [N-m]
            dynamic_moment_RMS[ir] = getRMS(dynamic_moment[:,ir,:])

            # fill in metrics
            results['Mbase_avg'].append(m_turbine[ir]*self.g * hArm[ir]*np.sin(self.Xi0[4]) + transformForce(self.f_aero0[:,ir], offset=[0,0,-hArm[ir]])[4] )# mean moment from weight and thrust
            results['Mbase_std'].append(dynamic_moment_RMS[ir])
            results['Mbase_PSD'][:,ir] = (getPSD(dynamic_moment[:,ir,:], self.dw))
            results['Mbase_max'].append(results['Mbase_avg'][ir]+3*results['Mbase_std'][ir])
            #results['Mbase_DEL'][iCase]
        
        # wave PSD for reference
        results['wave_PSD'] = getPSD(self.zeta, self.dw)        # wave elevation spectrum

        '''
        # TEMPORARY CHECK>>>
        import matplotlib.pyplot as plt
        plt.close('all')
        fig , ax  = plt.subplots(4, 1, sharex=True)
        ax[0].plot(abs(M_I     ))
        ax[1].plot(abs(M_w     ))
        ax[2].plot(abs(M_F_aero))
        ax[3].plot(abs(M_X_aero))

        ax[0].set_ylabel('M_I     ')
        ax[1].set_ylabel('M_w     ')
        ax[2].set_ylabel('M_F_aero')
        ax[3].set_ylabel('M_X_aero')
        plt.show()
        breakpoint()
        print(endnow)
        '''

        # initialize complex amplitudes for rotor response
        phi_w    = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # rotor azimuth deviation 
        omega_w  = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # rotor speed deviation
        torque_w = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # generator torque
        bPitch_w = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # blade pitch


        results['omega_avg'] = []
        results['omega_std'] = []
        results['omega_max'] = []
        results['omega_PSD'] = []
        results['torque_avg'] = []
        results['torque_std'] = []
        results['torque_PSD'] = []
        results['power_avg'] = []
        results['bPitch_avg'] = []
        results['bPitch_std'] = []
        results['bPitch_PSD'] = []
        
        
        for ir in range(self.nrotors):
        
            # get inflow speed for wind or current turbine
            if self.rotorList[ir].Zhub < 0:
                speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
            else:
                speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
        
            # rotor-related outputs are only available if aerodynamics modeling is enabled
            if self.rotorList[ir].aeroServoMod > 1 and speed > 0.0:
            
                # compute spectra of rotor azimuth variation, rotor speed, generator torque, and blade pitch
                for ih in range(self.nWaves):
                    phi_w[ih,ir,:] = self.rotorList[ir].C * XiHub[ih,ir,:]
                
                phi_w[-1,ir,:] = self.rotorList[ir].C * (XiHub[-1,ir,:] - self.rotorList[ir].V_w / (1j *self.w))
                
                # TODO
                omega_w[ :,ir,:] =  1j*self.w * phi_w[:,ir,:]
                torque_w[:,ir,:] = (1j*self.w * self.rotorList[ir].kp_tau  + self.rotorList[ir].ki_tau ) * phi_w[:,ir,:]
                bPitch_w[:,ir,:] = (1j*self.w * self.rotorList[ir].kp_beta + self.rotorList[ir].ki_beta) * phi_w[:,ir,:]
                
                # rotor speed (rpm) 
                results['omega_avg'].append(self.rotorList[ir].Omega_case)
                results['omega_std'].append(radps2rpm(getRMS(omega_w[:,ir,:])))
                results['omega_max'].append(results['omega_avg'][ir] + 2 * results['omega_std'][ir]) # this and other _max values will be based on std (avg + 2 or 3 * std)   (95% or 99% max)
                results['omega_PSD'].append(radps2rpm(1)**2 * getPSD(omega_w[:,ir,:], self.dw))
                
                # generator torque (Nm)
                results['torque_avg'].append(self.rotorList[ir].aero_torque / self.rotorList[ir].Ng)        # Nm
                results['torque_std'].append(getRMS(torque_w[:,ir,:]))
                results['torque_PSD'].append(getPSD(torque_w[:,ir,:], self.dw))
                # results['torque_max'][iCase]    # skip, nonlinear
                
                # rotor power (W)
                results['power_avg'].append(self.rotorList[ir].aero_power) # compute from cc-blade coeffs
                # results['power_std'][iCase]     # nonlinear near rated, covered by torque_ and omega_std
                # results['power_max'][iCase]     # skip, nonlinear
                
                # collective blade pitch (deg)
                results['bPitch_avg'].append(self.rotorList[ir].pitch_case)
                results['bPitch_std'].append(rad2deg(getRMS(bPitch_w[:,ir,:])))
                results['bPitch_PSD'].append(rad2deg(1)**2 *getPSD(bPitch_w[:,ir,:], self.dw))
                # results['bPitch_max'][iCase]    # skip, not something we'd consider in design
                
                # wind PSD for reference
                results['wind_PSD'] = getPSD(self.rotorList[ir].V_w, self.dw)   # <<< need to confirm


        '''
        Outputs from OpenFAST to consider covering:

        # Rotor power outputs
        self.add_output('V_out', val=np.zeros(n_ws_dlc11), units='m/s', desc='wind speed vector from the OF simulations')
        self.add_output('P_out', val=np.zeros(n_ws_dlc11), units='W', desc='rotor electrical power')
        self.add_output('Cp_out', val=np.zeros(n_ws_dlc11), desc='rotor aero power coefficient')
        self.add_output('Omega_out', val=np.zeros(n_ws_dlc11), units='rpm', desc='rotation speeds to run')
        self.add_output('pitch_out', val=np.zeros(n_ws_dlc11), units='deg', desc='pitch angles to run')
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production reconstructed from the openfast simulations')

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')

        self.add_output('loads_r',      val=np.zeros(n_span), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega',  val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch',  val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')

        # Control outputs
        self.add_output('rotor_overspeed', val=0.0, desc='Maximum percent overspeed of the rotor during an OpenFAST simulation')  # is this over a set of sims?

        # Blade outputs
        self.add_output('max_TipDxc', val=0.0, units='m', desc='Maximum of channel TipDxc, i.e. out of plane tip deflection. For upwind rotors, the max value is tower the tower')
        self.add_output('max_RootMyb', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root flapwise moment')
        self.add_output('max_RootMyc', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root out of plane moment')
        self.add_output('max_RootMzb', val=0.0, units='kN*m', desc='Maximum of the signals RootMzb1, RootMzb2, ... across all n blades representing the maximum blade root torsional moment')
        self.add_output('DEL_RootMyb', val=0.0, units='kN*m', desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('max_aoa', val=np.zeros(n_span), units='deg', desc='maxima of the angles of attack distributed along blade span')
        self.add_output('std_aoa', val=np.zeros(n_span), units='deg', desc='standard deviation of the angles of attack distributed along blade span')
        self.add_output('mean_aoa', val=np.zeros(n_span), units='deg', desc='mean of the angles of attack distributed along blade span')
        # Blade loads corresponding to maximum blade tip deflection
        self.add_output('blade_maxTD_Mx', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned x-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_My', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned y-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_Fz', val=np.zeros(n_span), units='kN', desc='distributed force in blade-aligned z-direction corresponding to maximum blade tip deflection')

        # Hub outputs
        self.add_output('hub_Fxyz', val=np.zeros(3), units='kN', desc = 'Maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz', val=np.zeros(3), units='kN*m', desc = 'Maximum hub moments in the non rotating frame')

        # Tower outputs
        self.add_output('max_TwrBsMyt',val=0.0, units='kN*m', desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt',val=0.0, units='kN*m', desc='damage equivalent load of tower base bending moment in fore-aft direction')
        self.add_output('tower_maxMy_Fx', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned x-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fy', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned y-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fz', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned z-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mx', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_My', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mz', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        '''

    def plot(self, ax, color=None, nodes=0, plot_rotor=True, station_plot=[], airfoils=False, zorder=2):
        '''plots the FOWT...'''

        R = rotationMatrix(self.r6[3], self.r6[4], self.r6[5])  # note: eventually Rotor could handle orientation internally <<<

        if self.ms:
            self.ms.plot(ax=ax, color=color)
        
        if color==None:
            color='k'
        
        if plot_rotor:
            for rotor in self.rotorList:
                coords = np.array([rotor.coords[0], rotor.coords[1], 0]) + self.r6[:3]
                rotor.plot(ax, r_ptfm=coords, R_ptfm=R, color=color, airfoils=airfoils, zorder=zorder)
                #rotor.plot(ax, r_ptfm=self.body.r6[:3], R_ptfm=self.body.R, color=color)
                '''
                for afmem in rotor.bladeMemberList:
                    afmem.calcOrientation()
                    afmem.plot(ax, r_ptfm=self.body.r6[:3], R_ptfm=self.body.R, color=color, nodes=nodes, station_plot=station_plot)
                '''


        # loop through each member and plot it
        for mem in self.memberList:

            mem.setPosition()  # offsets/rotations could be done in this function rather than in mem.plot <<<

            mem.plot(ax, r_ptfm=self.r6[:3], R_ptfm=R, color=color, 
                     nodes=nodes, station_plot=station_plot, zorder=zorder)

        # in future should consider ability to animate mode shapes and also to animate response at each frequency
        # including hydro excitation vectors stored in each member


    def plot2d(self, ax, color=None, station_plot=[], Xuvec=[1,0,0], Yuvec=[0,0,1]):
        '''plots the FOWT in 2d - only the platform and moorings so far...'''

        R = rotationMatrix(self.r6[3], self.r6[4], self.r6[5])  # note: eventually Rotor could handle orientation internally <<<

        if self.ms:
            self.ms.plot2d(ax=ax, color=color, Xuvec=Xuvec, Yuvec=Yuvec)

        if color==None:
            color='k'
            
        # loop through each member and plot it
        for mem in self.memberList:
            mem.setPosition()
            mem.plot(ax, r_ptfm=self.r6[:3], R_ptfm=R, color=color, plot2d=True, Xuvec=Xuvec, Yuvec=Yuvec)
