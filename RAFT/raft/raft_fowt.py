# RAFT's floating wind turbine class

import os
import shutil
import numpy as np
from scipy.interpolate import interp1d

import pyhams.pyhams     as ph
import raft.member2pnl as pnl
from raft.helpers import *
from raft.raft_member import Member
from raft.raft_rotor import Rotor

import multiprocessing as mp

# deleted call to ccblade in this file, since it is called in raft_rotor
# also ignoring changes to solveEquilibrium3 in raft_model and the re-addition of n=len(stations) in raft_member, based on raft_patch



class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, design, w, mpb, depth=600):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.

        Parameters
        ----------
        design : dict
            Dictionary of the design...
        w
            Array of frequencies to be used in analysis (rad/s)
        mpb
            A MoorPy Body object that represents this FOWT in MoorPy
        depth
            Water depth, positive-down. (m)
        '''

        # basic setup
        self.nDOF = 6
        self.Xi0 = np.zeros(6)    # mean offsets of platform, initialized at zero  [m, rad]

        self.depth = depth

        self.w = np.array(w)
        self.nw = len(w)            # number of frequencies
        self.dw = w[1]-w[0]         # frequency increment [rad/s]    
        
        self.k = np.array([waveNumber(w, self.depth) for w in self.w])  # wave number [m/rad]

        
        self.rho_water = getFromDict(design['site'], 'rho_water', default=1025.0)
        self.g         = getFromDict(design['site'], 'g'        , default=9.81)
             
            
        design['turbine']['tower']['dlsMax'] = getFromDict(design['turbine']['tower'], 'dlsMax', default=5.0)
             
             
        potModMaster = getFromDict(design['platform'], 'potModMaster', dtype=int, default=0)
        dlsMax       = getFromDict(design['platform'], 'dlsMax'      , default=5.0)
        min_freq_BEM = getFromDict(design['platform'], 'min_freq_BEM', default=self.dw/2/np.pi)
        self.dw_BEM  = 2.0*np.pi*min_freq_BEM
        self.dz_BEM  = getFromDict(design['platform'], 'dz_BEM', default=3.0)
        self.da_BEM  = getFromDict(design['platform'], 'da_BEM', default=2.0)
        
        
        self.aeroServoMod = getFromDict(design['turbine'], 'aeroServoMod', default=1)  # flag for aeroservodynamics (0=none, 1=aero only, 2=aero and control)
        
        
        # member-based platform description
        self.memberList = []                                         # list of member objects

        for mi in design['platform']['members']:

            if potModMaster==1:
                mi['potMod'] = False
            elif potModMaster==2:
                mi['potMod'] = True
                
            mi['dlsMax'] = dlsMax

            headings = getFromDict(mi, 'heading', shape=-1, default=0.)
            mi['headings'] = headings   # differentiating the list of headings/copies of a member from the individual member's heading
            if np.isscalar(headings):
                mi['heading'] = headings
                self.memberList.append(Member(mi, self.nw))
            else:
                for heading in headings:
                    mi['heading'] = heading
                    self.memberList.append(Member(mi, self.nw))
                mi['heading'] = headings # set the headings dict value back to the yaml headings value, instead of the last one used

        self.memberList.append(Member(design['turbine']['tower'], self.nw))
        #TODO: consider putting the tower somewhere else rather than in end of memberList <<<

        # mooring system connection
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine

        if 'yaw_stiffness' in design['platform']:
            self.yawstiff = design['platform']['yaw_stiffness']       # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        else:
            self.yawstiff = 0

        # Turbine rotor
        design['turbine']['rho_air' ] = design['site']['rho_air']
        design['turbine']['mu_air'  ] = design['site']['mu_air']
        design['turbine']['shearExp'] = design['site']['shearExp']
        
        self.rotor = Rotor(design['turbine'], self.w)

        # turbine RNA description
        self.mRNA    = design['turbine']['mRNA']
        self.IxRNA   = design['turbine']['IxRNA']
        self.IrRNA   = design['turbine']['IrRNA']
        self.xCG_RNA = design['turbine']['xCG_RNA']
        self.hHub    = design['turbine']['hHub']
        
        # initialize mean force arrays to zero, so the model can work before we calculate excitation
        self.F_aero0 = np.zeros(6)
        # mean weight and hydro force arrays are set elsewhere. In future hydro could include current.

        # initialize BEM arrays, whether or not a BEM sovler is used
        self.A_BEM = np.zeros([6,6,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([6,6,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        self.X_BEM = np.zeros([6,  self.nw], dtype=complex)               # linaer wave excitation force/moment coefficients vector [N, N-m]
        self.F_BEM = np.zeros([6,  self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]


    def calcStatics(self):
        '''Fills in the static quantities of the FOWT and its matrices.
        Also adds some dynamic parameters that are constant, e.g. BEM coefficients and steady thrust loads.'''

        rho = self.rho_water
        g   = self.g

        # structure-related arrays
        self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]
        self.C_struc_sub = np.zeros([6,6])            # substructure effective stiffness matrix [N/m, N, N-m]

        # hydrostatic arrays
        self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]  <<<<< not used yet


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
            self.W_struc += translateForce3to6DOF( np.array([0,0, -g*mass]), center )  # weight vector
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

            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(self.rho_water, self.g)  # call to Member method for hydrostatic calculations
            
            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices <<<<< needs updating (already about PRP)
            self.W_hydro += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix

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
        self.W_struc += translateForce3to6DOF(np.array([0,0, -g*self.mRNA]), center )   # weight vector
        self.M_struc += translateMatrix6to6DOF(Mmat, center)                            # mass/inertia matrix
        Sum_M_center += center*self.mRNA


        # ----------- process inertia-related totals ----------------

        mTOT = self.M_struc[0,0]                        # total mass of all the members
        rCG_TOT = Sum_M_center/mTOT                     # total CG of all the members
        self.rCG_TOT = rCG_TOT

        self.rCG_sub = msubstruc_sum/self.msubstruc     # solve for just the substructure mass and CG
        
        self.M_struc_subCM = translateMatrix6to6DOF(self.M_struc_subPRP, -self.rCG_sub) # the mass matrix of the substructure about the substruc's CM
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
        # >>> should double check proper handling of mean weight and buoyancy forces throughout model <<<
        self.body.m = mTOT
        self.body.v = VTOT
        self.body.rCG = rCG_TOT
        self.body.AWP = AWP_TOT
        self.body.rM = np.array([0,0,zMeta])
        # is there any risk of additional moments due to offset CB since MoorPy assumes CB at ref point? <<<



    def calcBEM(self, dw=0, wMax=0, wInf=10.0, dz=0, da=0, meshDir=os.path.join(os.getcwd(),'BEM')):
        '''This generates a mesh for the platform and runs a BEM analysis on it
        using pyHAMS. It can also write adjusted .1 and .3 output files suitable
        for use with OpenFAST.
        The mesh is only made for non-interesecting members flagged with potMod=1.
        
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
                                  numFreqs=-nw_HAMS, minFreq=dw_HAMS, dFreq=dw_HAMS)
                                  
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
            
            # interpole to the frequencies RAFT is using
            addedMassInterp = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([addedMass[:,:,2:], addedMass[:,:,0]]), assume_sorted=False, axis=2)(self.w)
            dampingInterp   = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([  damping[:,:,2:], np.zeros([6,6]) ]), assume_sorted=False, axis=2)(self.w)
            fExRealInterp   = interp1d(w3,   R      , assume_sorted=False        )(self.w)
            fExImagInterp   = interp1d(w3,   I      , assume_sorted=False        )(self.w)
            
            # copy results over to the FOWT's coefficient arrays
            self.A_BEM = self.rho_water * addedMassInterp
            self.B_BEM = self.rho_water * dampingInterp                                 
            self.X_BEM = self.rho_water * self.g * (fExRealInterp + 1j*fExImagInterp)
                        
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
        '''This computes turbine linear terms
        
        case
            dictionary of case information
        ptfm_pitch
            mean pitch angle of the platform [rad]
        
        '''
        
        #self.rotor.runCCBlade(case['wind_speed'], ptfm_pitch=ptfm_pitch, yaw_misalign=case['yaw_misalign'])
        
        # initialize arrays (can remain zero if aerodynamics are disabled)
        self.A_aero  = np.zeros([6,6,self.nw])                      # frequency-dependent aero-servo added mass matrix    
        self.B_aero  = np.zeros([6,6,self.nw])                      # frequency-dependent aero-servo damping matrix
        self.F_aero  = np.zeros([6, self.nw], dtype=complex)        # dynamice excitation force and moment amplitude spectra
        self.F_aero0 = np.zeros([6])                                # mean aerodynamic forces and moments
        
        # only compute the aerodynamics if enabled and windspeed is nonzero
        if self.aeroServoMod > 0 and case['wind_speed'] > 0.0:
        
            F_aero0, f_aero, a_aero, b_aero = self.rotor.calcAeroServoContributions(case, ptfm_pitch=ptfm_pitch)  # get values about hub
            
            # hub reference frame relative to PRP <<<<<<<<<<<<<<<<<
            rHub = np.array([0, 0, self.hHub])
            #rotMatHub = rotationMatrix(0, 0.01, 0)
            
            # convert coefficients to platform reference frame
            for i in range(self.nw):
                self.A_aero[:,:,i] = translateMatrix3to6DOF( np.diag([a_aero[i], 0, 0]),  rHub)
                self.B_aero[:,:,i] = translateMatrix3to6DOF( np.diag([b_aero[i], 0, 0]),  rHub)
            #self.C_aero = translateMatrix6to6DOF( rotateMatrix6(C_aero, rotMatHub),  rHub)
            
            # convert forces to platform reference frame
            self.F_aero0 = transformForce(F_aero0, offset=rHub)         # mean forces and moments
            for iw in range(self.nw):
                #self.F_aero[:,iw] = transformForce(F_aero[:,iw], offset=rHub, orientation=rotMatHub)
                self.F_aero[:,iw] = translateForce3to6DOF(np.array([f_aero[iw], 0, 0]), rHub)
        

    def calcHydroConstants(self, case):
        '''This computes the linear strip-theory-hydrodynamics terms, including wave excitation for a specific case.'''

        # set up sea state
        
        self.beta = case['wave_heading']

        # make wave spectrum
        if case['wave_spectrum'] == 'unit':
            self.zeta = np.tile(1, self.nw)
            S         = np.tile(1, self.nw)
        elif case['wave_spectrum'] == 'JONSWAP':
            S = JONSWAP(self.w, case['wave_height'], case['wave_period'])        
            self.zeta = np.sqrt(S)    # wave elevation amplitudes (these are easiest to use)
        elif case['wave_spectrum'] in ['none','still']:
            self.zeta = np.zeros(self.nw)        
            S = np.zeros(self.nw)        
        else:
            raise ValueError(f"Wave spectrum input '{case['wave_spectrum']}' not recognized.")
        
        rho = self.rho_water
        g   = self.g
        
        #print(f"significant wave height:  {4*np.sqrt(np.sum(S)*self.dw):5.2f} = {4*getRMS(self.zeta, self.dw):5.2f}") # << temporary <<<

        # TODO: consider current and viscous drift

        # ----- calculate potential-flow wave excitation force -----

        self.F_BEM = self.X_BEM * self.zeta     # wave excitation force (will be zero if HAMS wasn't run)
            
        # --------------------- get constant hydrodynamic values along each member -----------------------------

        self.A_hydro_morison = np.zeros([6,6])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]
        self.F_hydro_iner    = np.zeros([6,self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]

        # loop through each member
        for mem in self.memberList:
        
            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections
#            print(mem.name)

            # loop through each node of the member
            for il in range(mem.ns):
#                print(il)

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:
#                    print("underwater")

                    # get wave kinematics spectra given a certain wave spectrum and location
                    mem.u[il,:,:], mem.ud[il,:,:], mem.pDyn[il,:] = getWaveKin(self.zeta, self.beta, self.w, self.k, self.depth, mem.r[il,:], self.nw)

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
                            
                         
                        # added mass (axial term explicitly excluded here - we aren't dealing with chains)
                        Amat = rho*v_i *( Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat )  # local added mass matrix
                        #Amat = rho*v_i *( Ca_q*mem.qMat + Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat )  # local added mass matrix
#                        print(f"Member side added mass diagonals are {Amat[0,0]:6.2e} {Amat[1,1]:6.2e} {Amat[2,2]:6.2e}")

                        self.A_hydro_morison += translateMatrix3to6DOF(Amat, mem.r[il,:])    # add to global added mass matrix for Morison members
                        
                        # inertial excitation - Froude-Krylov  (axial term explicitly excluded here - we aren't dealing with chains)
                        Imat = rho*v_i *(  (1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat ) # local inertial excitation matrix (note: the 1 is the Cp, dynamic pressure, term)
                        #Imat = rho*v_i *( (1.+Ca_q)*mem.qMat + (1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat ) # local inertial excitation matrix

                        for i in range(self.nw):                                             # for each wave frequency...

                            mem.F_exc_iner[il,:,i] = np.matmul(Imat, mem.ud[il,:,i])         # add to global excitation vector (frequency dependent)

                            self.F_hydro_iner[:,i] += translateForce3to6DOF(mem.F_exc_iner[il,:,i], mem.r[il,:])  # add to global excitation vector (frequency dependent)


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
#                        print(f"Member END  added mass diagonals are {AmatE[0,0]:6.2e} {AmatE[1,1]:6.2e} {AmatE[2,2]:6.2e}")
                        
                        self.A_hydro_morison += translateMatrix3to6DOF(AmatE, mem.r[il,:]) # add to global added mass matrix for Morison members
                        
                        # inertial excitation
                        ImatE = rho*v_i * Ca_End*mem.qMat                         # local inertial excitation matrix (note, there is no 1 added to Ca_End because dynamic pressure is handled separately) 
                        #ImatE = rho*v_i * (1+Ca_End)*mem.qMat                         # local inertial excitation matrix

                        for i in range(self.nw):                                         # for each wave frequency...

                            #F_exc_iner_temp = np.matmul(ImatE, mem.ud[il,:,i])            # local inertial excitation force complex amplitude in x,y,z
                            mem.F_exc_a[il,:,i] = np.matmul(ImatE, mem.ud[il,:,i])            # local inertial excitation force complex amplitude in x,y,z

                            # >>> may want to add a separate dynamic pressure input <<<
                            mem.F_exc_p[il,:,i] = mem.pDyn[il,i]*a_i *mem.q                 # add dynamic pressure - positive with q if end A - determined by sign of a_i
                            #F_exc_iner_temp += mem.pDyn[il,i]*a_i *mem.q                 # add dynamic pressure - positive with q if end A - determined by sign of a_i

                            F_exc_iner_temp = mem.F_exc_a[il,:,i] + mem.F_exc_p[il,:,i] 
                            mem.F_exc_iner[il,:,i] += F_exc_iner_temp                    # add to stored member force vector

                            self.F_hydro_iner[:,i] += translateForce3to6DOF(F_exc_iner_temp, mem.r[il,:]) # add to global excitation vector (frequency dependent)



    def calcLinearizedTerms(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent linearized coefficients.

        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes [m, rad]

        '''

        rho = self.rho_water
        g   = self.g

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
                    vrel = mem.u[il,:] - vnode

                    # break out velocity components in each direction relative to member orientation [nw]
                    vrel_q  = vrel*mem.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                    vrel_p1 = vrel*mem.p1[:,None]
                    vrel_p2 = vrel*mem.p2[:,None]

                    # get RMS of relative velocity component magnitudes (real-valued)
                    vRMS_q  = getRMS(vrel_q , self.dw)
                    vRMS_p1 = getRMS(vrel_p1, self.dw)
                    vRMS_p2 = getRMS(vrel_p2, self.dw)
                    
                    #print(f"  {vRMS_q:5.2f}  {vRMS_p1:5.2f}  {vRMS_p2:5.2f}")

                    # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
                    Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * a_i_q  * Cd_q
                    Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * a_i_p1 * Cd_p1
                    Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * a_i_p2 * Cd_p2

                    Bmat = Bprime_q*mem.qMat + Bprime_p1*mem.p1Mat + Bprime_p2*mem.p2Mat # damping matrix for the node based on linearized drag coefficients

                    B_hydro_drag += translateMatrix3to6DOF(Bmat, mem.r[il,:])            # add to global damping matrix for Morison members

                    for i in range(self.nw):

                        mem.F_exc_drag[il,:,i] = np.matmul(Bmat, mem.u[il,:,i])          # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]

                        F_hydro_drag[:,i] += translateForce3to6DOF(mem.F_exc_drag[il,:,i], mem.r[il,:])   # add to global excitation vector (frequency dependent)


                    # ----- add end/axial effects for added mass, and excitation including dynamic pressure ------
                    # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))

                    Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*a_i*Cd_End
                    #print(f"  {a_i:5.2f}  {vRMS_q:5.2f}  {Bprime_End:5.2f}")

                    Bmat = Bprime_End*mem.qMat                                       #

                    B_hydro_drag += translateMatrix3to6DOF(Bmat, mem.r[il,:])        # add to global damping matrix for Morison members

                    for i in range(self.nw):                                         # for each wave frequency...

                        F_exc_drag_temp = np.matmul(Bmat, mem.u[il,:,i])             # local drag excitation force complex amplitude in x,y,z

                        mem.F_exc_drag[il,:,i] += F_exc_drag_temp                    # add to stored member force vector

                        F_hydro_drag[:,i] += translateForce3to6DOF(F_exc_drag_temp, mem.r[il,:]) # add to global excitation vector (frequency dependent)


        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag, F_hydro_drag


    def saveTurbineOutputs(self, results, case, iCase, Xi0, Xi):

        # platform motions
        results['surge_avg'][iCase] = Xi0[0]
        results['surge_std'][iCase] = getRMS(Xi[0,:], self.dw)
        results['surge_max'][iCase] = Xi0[0] + 3*results['surge_std'][iCase]
        results['surge_PSD'][iCase,:] = getPSD(Xi[0,:])
        
        results['sway_avg'][iCase] = Xi0[1]
        results['sway_std'][iCase] = getRMS(Xi[1,:], self.dw)
        results['sway_max'][iCase] = Xi0[1] + 3*results['heave_std'][iCase]
        results['sway_PSD'][iCase,:] = getPSD(Xi[1,:])
        
        results['heave_avg'][iCase] = Xi0[2]
        results['heave_std'][iCase] = getRMS(Xi[2,:], self.dw)
        results['heave_max'][iCase] = Xi0[2] + 3*results['heave_std'][iCase]
        results['heave_PSD'][iCase,:] = getPSD(Xi[2,:])
        
        roll_deg = rad2deg(Xi[3,:])
        results['roll_avg'][iCase] = rad2deg(Xi0[3])
        results['roll_std'][iCase] = getRMS(roll_deg, self.dw)
        results['roll_max'][iCase] = rad2deg(Xi0[3]) + 3*results['roll_std'][iCase]
        results['roll_PSD'][iCase,:] = getPSD(roll_deg)
        
        pitch_deg = rad2deg(Xi[4,:])
        results['pitch_avg'][iCase] = rad2deg(Xi0[4])
        results['pitch_std'][iCase] = getRMS(pitch_deg, self.dw)
        results['pitch_max'][iCase] = rad2deg(Xi0[4]) + 3*results['pitch_std'][iCase]
        results['pitch_PSD'][iCase,:] = getPSD(pitch_deg)
        
        yaw_deg = rad2deg(Xi[5,:])
        results['yaw_avg'][iCase] = rad2deg(Xi0[5])
        results['yaw_std'][iCase] = getRMS(yaw_deg, self.dw)
        results['yaw_max'][iCase] = rad2deg(Xi0[5]) + 3*results['yaw_std'][iCase]
        results['yaw_PSD'][iCase,:] = getPSD(yaw_deg)
        
        XiHub = Xi[0,:] + self.hHub*Xi[4,:]  # hub fore-aft displacement amplitude (used as an approximation in a number of outputs)
        
        # nacelle acceleration
        results['AxRNA_std'][iCase] = getRMS(XiHub*self.w**2, self.dw)
        results['AxRNA_PSD'][iCase,:] = getPSD(XiHub*self.w**2)
        
        # tower base bending moment
        m_turbine   = self.mtower + self.mRNA                                       # turbine total mass
        zCG_turbine = (self.rCG_tow[2]*self.mtower + self.hHub*self.mRNA)/m_turbine # turbine center of gravity
        zBase = self.memberList[-1].rA[2]                                           # tower base elevation [m]
        hArm = zCG_turbine - zBase                                                  # vertical distance from tower base to turbine CG [m]
        aCG_turbine = -self.w**2 *( Xi[0,:] + zCG_turbine*Xi[4,:] )                 # fore-aft acceleration of turbine CG
        # turbine pitch moment of inertia about CG [kg-m^2]
        ICG_turbine = (translateMatrix6to6DOF(self.memberList[-1].M_struc, [0,0,-zCG_turbine])[4,4]     # tower MOI about turbine CG
                       + self.mRNA*(self.hHub-zCG_turbine)**2 + self.IrRNA   )                          # RNA MOI with parallel axis theorem
        # moment components and summation (all complex amplitudes)
        M_I      = -m_turbine*aCG_turbine*hArm - ICG_turbine*(-self.w**2 *Xi[4,:] ) # tower base inertial reaction moment
        M_w      = m_turbine*self.g * hArm*Xi[4]                                    # tower base weight moment
        M_F_aero = 0.0 # <<<<self.F_aero[0,:]*(self.hHub - zBase)                   # tower base moment from turbulent wind excitation    
        M_X_aero = -(-self.w**2 *self.A_aero[0,0,:]                                 # tower base aero reaction moment
                     + 1j*self.w *self.B_aero[0,0,:] )*(self.hHub - zBase)**2 *Xi[4,:]        
        dynamic_moment = M_I + M_w + M_F_aero + M_X_aero                            # total tower base fore-aft bending moment [N-m]
        dynamic_moment_RMS = getRMS(dynamic_moment, self.dw)
        # fill in metrics
        results['Mbase_avg'][iCase] = m_turbine*self.g * hArm*np.sin(Xi0[4]) + transformForce(self.F_aero0, offset=[0,0,-hArm])[4] # mean moment from weight and thrust
        results['Mbase_std'][iCase] = dynamic_moment_RMS
        results['Mbase_max'][iCase] = results['Mbase_avg'][iCase] + 3 * results['Mbase_std'][iCase]
        results['Mbase_PSD'][iCase,:] = getPSD(dynamic_moment)
        #results['Mbase_max'][iCase]
        #results['Mbase_DEL'][iCase]
        
        
        # wave PSD for reference
        results['wave_PSD'][iCase,:] = getPSD(self.zeta)        # wave elevation spectrum
        
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

        # rotor-related outputs are only available if aerodynamics modeling is enabled
        if self.aeroServoMod > 1 and case['wind_speed'] > 0.0:
            # rotor speed (rpm)
            # spectra
            phi_w   = self.rotor.C * (XiHub - self.rotor.V_w / (1j *self.w))
            omega_w =  (1j *self.w) * phi_w

            results['omega_avg'][iCase]     = self.rotor.Omega_case
            results['omega_std'][iCase]     = radps2rpm(getRMS(omega_w, self.dw))
            results['omega_max'][iCase]     = results['omega_avg'][iCase] + 2 * results['omega_std'][iCase] # this and other _max values will be based on std (avg + 2 or 3 * std)   (95% or 99% max)
            results['omega_PSD'][iCase]     = radps2rpm(1)**2 * getPSD(omega_w)
            
            # generator torque (Nm)
            torque_w = (1j * self.w * self.rotor.kp_tau + self.rotor.ki_tau) * phi_w

            results['torque_avg'][iCase]    = self.rotor.aero_torque / self.rotor.Ng        # Nm
            results['torque_std'][iCase]    = getRMS(torque_w, self.dw)
            results['torque_PSD'][iCase]    = getPSD(torque_w)
            # results['torque_max'][iCase]    # skip, nonlinear
            
            
            # rotor power (W)
            results['power_avg'][iCase]    = self.rotor.aero_power # compute from cc-blade coeffs
            # results['power_std'][iCase]     # nonlinear near rated, covered by torque_ and omega_std
            # results['power_max'][iCase]     # skip, nonlinear

            
            # collective blade pitch (deg)
            bPitch_w = (1j * self.w * self.rotor.kp_beta + self.rotor.ki_beta) * phi_w

            results['bPitch_avg'][iCase]    = self.rotor.pitch_case
            results['bPitch_std'][iCase]    = rad2deg(getRMS(bPitch_w, self.dw))
            results['bPitch_PSD'][iCase]    = rad2deg(1)**2 *getPSD(bPitch_w)
            # results['bPitch_max'][iCase]    # skip, not something we'd consider in design
            
            
            # wind PSD for reference
            results['wind_PSD'][iCase,:] = getPSD(self.rotor.V_w)   # <<< need to confirm


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

    def plot(self, ax, color='k', nodes=0):
        '''plots the FOWT...'''

        self.rotor.plot(ax, r_ptfm=self.body.r6[:3], R_ptfm=self.body.R, color=color)

        # loop through each member and plot it
        for mem in self.memberList:

            mem.calcOrientation()  # temporary

            mem.plot(ax, r_ptfm=self.body.r6[:3], R_ptfm=self.body.R, color=color, nodes=nodes)

        # in future should consider ability to animate mode shapes and also to animate response at each frequency
        # including hydro excitation vectors stored in each member
