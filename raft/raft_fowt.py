# RAFT's floating wind turbine class

import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

import pyhams.pyhams     as ph
import raft.member2pnl as pnl
from raft.helpers import *
import raft.raft_member as mem

from importlib import reload
Member = reload(mem).Member

try:
    import ccblade         # clone ccblade
except:
    import wisdem.ccblade  # via conda install wisdem




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
