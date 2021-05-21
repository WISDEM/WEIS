# RAFT's main model class

import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import moorpy as mp
import pyhams.pyhams     as ph
import raft.member2pnl as pnl
import raft.raft_fowt  as fowt
from raft.helpers import *

#import F6T1RNA as structural    # import turbine structural model functions

# reload the libraries each time in case we make any changes
from importlib import reload
mp     = reload(mp)
ph     = reload(ph)
pnl    = reload(pnl)
FOWT   = reload(fowt).FOWT


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


        # analysis frequency array
        if len(w)==0:
            w = np.arange(.05, 3, 0.05)  # angular frequencies tp analyze (rad/s)

        self.w = np.array(w)
        self.nw = len(w)  # number of frequencies

        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # set up the FOWT here  <<< only set for 1 FOWT for now <<<
        self.fowtList.append(FOWT(design, w=self.w, mpb=self.ms.bodyList[0], depth=depth))
        self.coords.append([0.0,0.0])
        self.nDOF += 6

        self.ms.bodyList[0].type = -1  # need to make sure it's set to a coupled type

        try:
            self.ms.initialize()  # reinitialize the mooring system to ensure all things are tallied properly etc.
        except Exception as e:
            raise RuntimeError('An error occured when initializing the mooring system: '+e.message)
        
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

        # First get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        try: 
            self.C_moor0 = self.ms.getCoupledStiffness(lines_only=True)                             # this method accounts for eqiuilibrium of free objects in the system
            self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)
        except Exception as e:
            raise RuntimeError('An error occured when getting linearized mooring properties in undisplaced state: '+e.message)

        self.results['properties'] = {}   # signal this data is available by adding a section to the results dictionary
        
        
    
    def calcMooringAndOffsets(self):
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.  This will ultimately become a method for solving mean operating point.
        '''


        # Now find static equilibrium offsets of platform and get mooring properties about that point
        # (This assumes some loads have been applied)
        #self.ms.display=2

        try:
            self.ms.solveEquilibrium3(DOFtype="both", tol=-0.01) #, rmsTol=1.0E-5)     # get the system to its equilibrium
        except Exception as e:     #mp.MoorPyError
            print('An error occured when solving system equilibrium: '+e.message)
            #raise RuntimeError('An error occured when solving unloaded equilibrium: '+error.message)
            
        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        
        print("Equilibrium'3' platform positions/rotations:")
        printVec(self.ms.bodyList[0].r6)

        r6eq = self.ms.bodyList[0].r6

        #self.ms.plot()

        print("Surge: {:.2f}".format(r6eq[0]))
        print("Pitch: {:.2f}".format(r6eq[4]*180/np.pi))

        try:
            C_moor = self.ms.getCoupledStiffness(lines_only=True)
            F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)    # get net forces and moments from mooring lines on Body
        except Exception as e:
            raise RuntimeError('An error occured when getting linearized mooring properties in offset state: '+e.message)
            
        # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
        C_moor[5,5] += fowt.yawstiff

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
        C_tot += np.array(self.C_moor0)

        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]

        # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
        C_tot[5,5] += fowt.yawstiff

        # add fowt's terms to system matrices (BEM arrays are not yet included here)
        M_tot += fowt.M_struc + fowt.A_hydro_morison   # mass
        C_tot += fowt.C_struc + fowt.C_hydro           # stiffness

        # check viability of matrices
        message=''
        for i in range(self.nDOF):
            if M_tot[i,i] < 1.0:
                message += f'Diagonal entry {i} of system mass matrix is less than 1 ({M_tot[i,i]}). '
            if C_tot[i,i] < 1.0:
                message += f'Diagonal entry {i} of system stiffness matrix is less than 1 ({C_tot[i,i]}). '
                
        if len(message) > 0:
            raise RuntimeError('System matrices computed by RAFT have one or more small or negative diagonals: '+message)

        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)
        eigenvals, eigenvectors = np.linalg.eig(np.matmul(np.linalg.inv(M_tot), C_tot))   # <<< need to sort this out so it gives desired modes, some are currently a bit messy

        if any(eigenvals <= 0.0):
            raise RuntimeError("Error: zero or negative system eigenvalues detected.")

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
        #self.ms.bodyList[0].setPosition(np.zeros(6))
        #self.ms.initialize()
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
