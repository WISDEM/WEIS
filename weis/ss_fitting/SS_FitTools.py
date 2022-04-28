import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt 
import math
from weis.ss_fitting.invfreqs import invfreqs
from scipy import interpolate
from scipy.signal import freqs, bode, tf2ss, ss2tf, lfilter, cont2discrete, impulse
from scipy.linalg import block_diag, eig, hankel, svd
from datetime import datetime
from control import StateSpace, matlab, balred, impulse_response
from pyhams.pyhams import read_wamit3
# from control import tf2ss, ss, ss2tf
# from control.matlab import bode

class SSFit_Radiation(object):

    def __init__(self, **kwargs):

        self.InputFile  = ''
        self.InputVars  = dict()
        self.WamType    = 1

        # Set defaults
        self.HydroFile      = ''        # Wamit Files name and Location
        self.DOFs           = [1,1,1,1,1,1]   # Degree of Freedom enabled
        self.FreqRange      = [0.0,2.5]     # Typical local frequencies range (rad/s)
        self.WeightFact     = 1  # Frequencies weighting factors
        self.FitReq         = 0.99 # Fit required for the parametric model (max. recomended 0.99)
        self.PlotFit        = 1  # Plot the parametric model fit (0 or 1)
        self.ManReduction   = 0   # Manual (1) or Automatic(0) order reduction (Only if method=4)

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(SSFit_Radiation, self).__init__()

        # Load Wamit
        self.wam = WAMIT_Out(
            HydroFile=self.HydroFile, 
            Type=1
            )

    
    # def readInputs(self):

    #     with open(self.InputFile,'r') as f:
    #         try:
    #             f.readline()
    #             self.HydroFile      = f.readline().split('%')[0].strip()
    #             self.DOFs          = np.fromstring(f.readline().split('%')[0].strip()[1:-1],dtype=int,sep = ' ')
    #             self.FreqRange     = np.fromstring(f.readline().split('%')[0].strip()[1:-1],dtype=float,sep = ' ')
    #             self.WeightFact    = float(f.readline().split('%')[0])
    #             self.IDMethod      = int(f.readline().split('%')[0])        # not used
    #             self.FitReq        = float(f.readline().split('%')[0])
    #             self.PlotFit       = int(f.readline().split('%')[0])
    #             self.ManReduction  = int(f.readline().split('%')[0])
    #         except:
    #             print('self Input File Error: Error reading the input file.')
    #             print('Please use the reference file ./test_cases/Haliade/SS_Fitting_Options.inp')


    #     # Input file error checking
    #     # FreqRange

    #     if len(self.FreqRange) != 2:
    #         raise Exception('SS_Fit Input File Error: Error with self.FreqRange in input file, must contain 2 frequencies')

    #     if self.FreqRange[0] < 0:
    #         raise Exception('SS_Fit Input File Error: Minimum frequency in self.FreqRange must be >= 0')

    #     if self.FreqRange[1] < self.FreqRange[0]:
    #         raise Exception('SS_Fit Input File Error: self.FreqRange must be increasing')

    #     # Weighting Factors
    #     if self.WeightFact < 0 or self.WeightFact > 1:
    #         raise Exception('SS_Fit Input File Error: WeightFact must be between 0 and 1')

    #     # Fit

    #     if self.FitReq < 0 or self.FitReq > 1:
    #         raise Exception('SS_Fit Input File Error: FitReq must be between 0 and 1')

        

    # def setWeights(self,omega):
        
    #     omega_Range             = self.FreqRange

    #     weight_Ind              = np.bitwise_and(omega >= omega_Range[0],omega <= omega_Range[1])

    #     weight                  = np.zeros([len(omega),1])
    #     weight[weight_Ind]      = self.WeightFact
    #     weight[~weight_Ind]     = 1 - self.WeightFact

    #     #normalize & save
    #     self.weights            = weight/sum(weight)

    def outputMats(self):
        
        # Find number of states in each sys
        states = [np.shape(S[0])[0] for S in self.sys]

        # Find number of states per dof (first index)
        statesPerDoF = np.zeros(6, dtype=int)

        for i in range(0,6):
            for k, DoF in enumerate(self.sysDOF):
                if DoF[0] == i:
                    statesPerDoF[i] += states[k]


        # states = np.zeros(len(self.sys))
        # for i,S in enumerate(self.sys):
        #     states[i] = np.shape(A.A)[1]


        # Make block diagonal matrices
        AA = np.empty(0)
        for S in self.sys:
            AA = block_diag(AA,S[0])

        # Remove first row because of initialization 
        AA = AA[1:]

        # Input BB and output CC matrix, first initialize: 6 inputs and outputs
        BB = np.zeros([1,6])
        CC = np.zeros([6,1])

        for k, (S, dof) in enumerate(zip(self.sys,self.sysDOF)):
            B_k = np.squeeze(np.zeros((states[k],6)))
            B_k[:,dof[0]]= np.squeeze(np.asarray(S[1]))
            BB = np.concatenate((BB,B_k),axis=0)

            C_k = np.squeeze(np.zeros((6,states[k])))
            C_k[dof[1],:] = -np.squeeze(np.asarray(S[2]))
            CC = np.concatenate((CC,C_k),axis=1)
            # print('here')
            
        BB = BB[1:,:] 
        CC = CC[:,1:]

        # check stability
        tfAll   = ss2tf(AA,BB,CC,np.zeros([6,6]))
        ssAll   = tf2ss(tfAll[0],tfAll[1])

        # via the roots of denominator of tfAll
        if any(np.roots(tfAll[1]) > 0):
            print('Warning: SS system unstable.  Try with a lower R^2 value or check inputs and be careful using some DoFs')

        # check if proper
        #  order of numerator       order of denominator
        if np.shape(tfAll[0])[1] > np.shape(tfAll[1])[0]:
            print('Warning: SS system is not proper')

        # check if passive: TODO: DO
        # started, but went back and forth on whether to use control or scipy.signal
        # print('here')
        # for i in range(0,6):
        #     mag = bode(sysAll[i,i],np.linspace(0,5))

        # write output files
        with open(self.HydroFile+'.ss','w') as f:


            # header info
            now = datetime.now()
            print('here')
            f.write('{}\n'.format('SS_Fitting v1.00.01: State-Spaces Matrices obtained using FDI method, on ' + now.strftime('%b-%d-%Y %H:%M:%S')))
            f.write('{}\t\t\t{}\n'.format(np.array_str(np.array(self.DOFs))[1:-1],'%Enabled DoFs'))
            f.write('{}\t\t\t\t\t{}\n'.format(sum(states), '%Radiation States'))
            f.write('{}\t\t\t{}\n'.format(np.array_str(statesPerDoF)[1:-1], '%Radiation States per DoF'))

            np.savetxt(f,AA,fmt='%.6e')
            np.savetxt(f,BB,fmt='%.6e')
            np.savetxt(f,CC,fmt='%.6e')

        print('Fit a SS model with {} states'.format(sum(states)))


class SSFit_Excitation(object):

    def __init__(self, **kwargs):

        self.InputFile  = ''
        self.InputVars  = dict()
        self.WamType    = 3

        # Input defaults
        self.HydroFile     = ''
        self.DOFs          = [1,1,1,1,1,1] 
        self.WaveTMax      = 3600 # 
        self.dT            = 0.025 # 
        self.FitReq        = 0.99 # Fit required for the parametric model (max. recomended 0.99)
        self.PlotFit       = 1 # Plot the parametric model fit (0 or 1)
        # self.ManOrdRed     = 0 # Manual (1) or Automatic(0) order reduction (Only if method=4)
        self.WtrDen        = 1025.0 # Water density (kg/m^3)
        self.Grav          = 9.80665  # Grav     Gravitational acceleration (m/s^2)
        self.WaveDir       = 0 # Incident wave propagation heading direction (degrees)
        self.tc            = 12  # offset time

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        # Load Wamit
        self.wam = WAMIT_Out(
            HydroFile=self.HydroFile, 
            Type=3,
            grav = self.Grav,
            dens = self.WtrDen,
            heading = self.WaveDir
            )

        # Time domain fit is only option currently
        self.TDF = TimeDomain_Fit(
            dT          = self.dT,
            WaveTMax    = self.WaveTMax,
            wam         = self.wam,
            tc          = self.tc
        )

        self.TDF.fit()

    
    # def readInputs(self):

    #     with open(self.InputFile,'r') as f:
    #         try:
    #             f.readline()


    #         except:
    #             print('SS_Fit Input File Error: Error reading the input file.')
    #             print('Please use the reference file ./test_cases/Haliade/SS_Fitting_Options.inp')


    #     # Input file error checking
    #     # FreqRange
    #     if sum(self.InputVars['DOFs']) < 6:
    #         print('SS_Fit Warning: num DOFs < 6, only DOFs = 6 supported for now')

    #     # Fit
    #     FitReq = self.InputVars['FitReq']

    #     if FitReq < 0 or FitReq > 1:
    #         print('SS_Fit Input File Error: FitReq must be between 0 and 1')


    def writeMats(self):

        sys_fit = self.TDF.K_fit

        # A_global is block diag of all A mats, we assume 6 dofs, so hardcode
        A_global = block_diag(sys_fit[0].A,sys_fit[1].A,sys_fit[2].A,sys_fit[3].A,sys_fit[4].A,sys_fit[5].A)
        B_global = np.asarray(np.block([[sys_fit[0].B],[sys_fit[1].B],[sys_fit[2].B],[sys_fit[3].B],[sys_fit[4].B],[sys_fit[5].B]]))
        C_global = block_diag(sys_fit[0].C,sys_fit[1].C,sys_fit[2].C,sys_fit[3].C,sys_fit[4].C,sys_fit[5].C)
        
        states_per_dof  = np.array([sys.A.shape[0] for sys in sys_fit])
        
        # write output files
        with open(self.HydroFile+'.ssexctn','w') as f:

            # header info
            now = datetime.now()
            print('here')
            f.write('{}\n'.format('SS_Fitting v1.00.01: State-Spaces Matrices obtained using Time Domain method, on ' + now.strftime('%b-%d-%Y %H:%M:%S')))
            f.write('{}\t\t\t\t\t\t\t{}\n'.format(self.WaveDir,'%Wave heading angle'))
            f.write('{}\t\t\t\t\t\t{}\n'.format(np.array_str(np.array(self.tc)),'%time offset (tc)'))
            f.write('{}\t\t\t\t\t\t\t{}\n'.format(A_global.shape[0], '%Total Excitation States'))
            f.write('{}\t\t\t{}\n'.format(np.array_str(states_per_dof)[1:-1], '%Excitation States per DoF'))

            np.savetxt(f,A_global,fmt='%.6e')
            np.savetxt(f,B_global,fmt='%.6e')
            np.savetxt(f,C_global,fmt='%.6e')

        print('Fit a SS Excitation model with {} states'.format(A_global.shape[0]))

class WAMIT_Out(object):
    def __init__(self,**kwargs):
        self.HydroFile   = ''
        self.Type       = None

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        self.readFRFs()


    def readFRFs(self,heading=[],grav=[],dens=[]):

        if self.Type == 1:

            with open(self.HydroFile+'.1','r') as f:
                all_data = [x.split() for x in f.readlines()]
                
                per     = [float(x[0]) for x in all_data]
                I       = [int(x[1]) for x in all_data]
                J       = [int(x[2]) for x in all_data]
                Abar_ij = [float(x[3]) for x in all_data]
                Bbar_ij = np.zeros([1,len(Abar_ij)]).tolist()[0]

                for iPer, p in enumerate(per):
                    if p > 0:
                        Bbar_ij[iPer] = all_data[iPer][4]

            # check that all the lengths are the same
            if len(per) != len(I) != len(J) != len(Abar_ij) != len(Bbar_ij):
                print('The lengths of per, I, J, Abar, and Bbar are not the same!')


            # put Abar_ij, Bbar_ij into 6x6x(nFreq) matrices
            Abar = np.zeros([6,6,len(np.unique(per))])
            Bbar = np.zeros([6,6,len(np.unique(per))])

            Period = np.unique(per)

            # loop through because I don't know a cleverer way
            # note that the index will be -1 compared to matlab and the normal way of writing them
            # A(i,j,per)
            for k, p in enumerate(per):
                Abar[I[k]-1,J[k]-1,np.where(Period==p)[0]] = Abar_ij[k]
                Bbar[I[k]-1,J[k]-1,np.where(Period==p)[0]] = Bbar_ij[k]
            
            
            # Break Abar into A_inf and Abar(\omega)
            Abar_inf        = np.squeeze(Abar[:,:,Period==0])
            Abar            = Abar[:,:,Period!=0]
            Bbar            = Bbar[:,:,Period!=0]
            omega           = 2 * np.pi / Period[Period!=0]
            omega[omega<0]  = 0
            
            # Sort based on frequency because
            freqIndSort     = np.argsort(omega)
            omega           = omega[freqIndSort]
            Abar            = Abar[:,:,freqIndSort]
            Bbar            = Bbar[:,:,freqIndSort]



            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(omega,Bbar[1,1,:])
                plt.show()
            

            ### Dimensionalize
            # Coefficients
            rho     = 1000. # water density as set in matlab script
            L_ref   = 1.    # reference length as set in WAMIT
            
            # scale factor depends on dimension
            # Abar_ij = A_ij / (\rho L^k), Bbar_ij = B_ij / (\rho L^k \omega)
            # k = 3 for i,j <= 3, k = 5 for i,j > 3, and k = 4 for cross terms, kind of moot since L = 1
            kk      = np.array([(3,3,3,4,4,4),\
                                (3,3,3,4,4,4),\
                                (3,3,3,4,4,4),\
                                (4,4,4,5,5,5),\
                                (4,4,4,5,5,5),\
                                (4,4,4,5,5,5)], dtype=int)

            # scaling matrices
            scaleA  = np.reshape(np.tile(rho * L_ref**kk,[1,1,len(omega)]),np.shape(Abar))
            scaleB  = [rho * L_ref**kk * om for om in omega ]  #close! but we need to rearange indices
            scaleB  = np.transpose(scaleB,[1,2,0])

            # scale, finally
            A       = Abar * scaleA
            A_inf    = Abar_inf * scaleA[:,:,0]
            B       = Bbar * scaleB

            # Form K
            K = B + 1j * np.tile(omega,[6,6,1]) * (A - np.transpose(np.tile(A_inf,[len(omega),1,1]),[1,2,0]))



            # Interpolate to get evenly spaced matrices
            exp     = 8
            om_samp = np.linspace(0,max(omega),2**exp)

            fA      = interpolate.interp1d(omega,A)
            fB      = interpolate.interp1d(omega,B)
            fK      = interpolate.interp1d(omega,K)
            
            A_samp   = fA(om_samp)
            B_samp   = fB(om_samp)
            K_samp   = fK(om_samp)




            if False:
                plt.plot(omega,K.imag[3,3,:],'o',om_samp,K_samp.imag[3,3,:],'-')
                plt.show()


            # Set insignificant entries of K to zero
            # Doing this the same way as matlab for now

            # Find maximum diagonal element of damping (B) matrix
            Bbar_max = np.diag(Bbar.max(2)).max()

            # if any diagonal elements are less than 0.0001 * Bbar_max  -> 0
            # if any non-diagonal elements are less than 0.0005 * Bbar_max -> 0
            for i in range(0,6):
                for j in range(0,6):
                    if i == j:
                        if max(abs(Bbar[i,j,:])) < 0.0001 * Bbar_max:
                            K_samp[i,j,:] = 0
                            print(idDOF(i) + ' - Unimportant DoF, not fit')
                    else:
                        if max(abs(Bbar[i,j,:])) < 0.0005 * Bbar_max:
                            K_samp[i,j,:] = 0

            
            # store values
            self.omega      = om_samp
            self.K          = K_samp
            self.A          = A_samp
            self.A_inf      = A_inf
            self.B          = B_samp
            self.Bbar_max   = Bbar_max
        
        elif self.Type == 3:
            
            # Read WAMIT outputs
            mod, phase, real, imag, omega, headings = read_wamit3(self.HydroFile+'.3')

            # Convert to data format used here
            wamOuts = {}
            wamOuts['direction_diff'] = headings
            wamOuts['X_mag']          = mod
            wamOuts['X_phase']        = phase
            wamOuts['X_re']           = real
            wamOuts['X_imag']         = imag


            # Form complex response
            Kbar = wamOuts['X_re'] + 1j*wamOuts['X_imag']

            # Interpolate based on direction
            if Kbar.shape[0] > 1:
                fDir    = interpolate.interp1d(wamOuts['direction_diff'],Kbar,axis=0)
                Kbar_dir = fDir(self.heading)
            else:
                Kbar_dir = Kbar[0]

            # Scale
            L       = 1
            A       = self.grav * self.dens
            scale   = np.array([A, A, A, A/L, A/L, A/L])
            S       = np.tile(scale,[Kbar_dir.shape[1],1])

            K_dir   = Kbar_dir.T * S

            # Remove, insignificant DoFs, Set dof to 0 if less than epsilon
            epsilon = 1e-3
            for dof in range(0,6):
                if max(abs(K_dir[:,dof])) < epsilon:
                    K_dir[:,dof] = np.zeros(np.shape(K_dir[:,dof]))

            # Append omega = 0 and K_dir[omega=0,:] = 0
            omega   = np.concatenate((np.zeros(1),omega))
            K_dir   = np.concatenate((np.zeros((1,6)),K_dir),axis=0)

            # save to object
            self.omega  = omega
            self.K_dir  = K_dir

            if False:
                iDoF = 4
                ax1 = plt.subplot(211)
                ax1.semilogx(omega,np.abs(K_dir[:,iDoF]))

                ax2 = plt.subplot(212)
                ax2.semilogx(omega,np.angle(K_dir[:,iDoF]))

                plt.show()
            
            # print('here')



        return

class FDI_Fitting(SSFit_Radiation):

    def __init__(self,**kwargs):

            #Input option structure
        self.OrdMax = 7           # - Maximum order of transfer function to be considered. Typical value 20.
        self.AinfFlag = 1          # - if set to 1, the algorithm uses Ainf (3D hydrodynamic data),  #if set to 0, the algorithm estimates Ainf (2D hydrodynamic data)
        self.Method = 2            # - There are 3 parameter estimation methods (1,2,3). See help of fit_siso_fresp. Recomended method 2 (best trade off between accuracy and speed)
        self.Iterations = 20       # - Related to parameter estimation methods 2 and 3. See help of fit_siso_fresp. Typical value 20.
        self.PlotFlag = 0          # - If set to 1 the function plots the results of each iteration of the automatic order detection in a separate figure.
        self.LogLin = 1            #  - logarithmic or linear frequency scale for plotting.
        self.wsFactor = 0.1        # - Sample faster than the Hydrodynamic code for plotting. Typical value 0.1.
        self.wminFactor = 0.1      # - The minimum frequency to be used in the plot is self.wminFactor*Wmin, where  #Wmin is the minimum frequency of the dataset used for identification.%Typical value 0.1.
        self.wmaxFactor = 5        #- the maximum frequency to be used in the plot is self.wmaxFactor*Wmax, where Wmax is the maximum frequency of the dataset used for identification. Typical value 5.
        
        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(FDI_Fitting, self).__init__(**kwargs)

    def fit(self):

        # unpack mass and damping matrices
        A       = self.wam.A
        B       = self.wam.B
        # K       = self.wam.K
        A_inf   = self.wam.A_inf

        sys     = []   #initialize state space control
        sysDOF  = []
        Khat    = []

        for i in range(0,6):
            for j in range(0,6):
                if max(abs(self.wam.K[i,j,:])) > 0:
                    
                    # one index pair at a time
                    A_ij = A[i,j,:]
                    B_ij = B[i,j,:]

                    # use only weighted frequency indices
                    useFreq = np.bitwise_and(self.wam.omega > self.FreqRange[0], self.wam.omega < self.FreqRange[1])
                    A_ij    = A_ij[useFreq]
                    B_ij    = B_ij[useFreq]
                    om      = self.wam.omega[useFreq]

                    # Compute Memory Function Frequency Response K(j\omega)
                    self.K_ij       = B_ij + 1j * om * (A_ij - A_inf[i,j])

                    # Initial order
                    self.order = 2

                    K_Num, K_Den, Khat_ij = self.ident_memory(om,Plot=False)
                    

                    # compute error
                    r2b, r2a = computeError(self.K_ij,Khat_ij,om,A_inf[i,j])

                    # Increase order of transfer function until R^2 satisfied, unless maximum order reached
                    # Currently, only A or B needs to match, as it is written in matlab, could update
                    while (self.order < self.OrdMax) and  \
                         ((r2b < self.FitReq) and (r2a < self.FitReq)):

                         self.order += 1
                         K_Num, K_Den, Khat_ij = self.ident_memory(om,Plot=False)

                         r2b, r2a = computeError(self.K_ij,Khat_ij,om,A_inf[i,j])

                    # Convert to state-space system
                    sys.append(tf2ss(K_Num,K_Den))
                    sysDOF.append([i,j])

                    _, Khat_samp  = freqs(K_Num,K_Den,worN=self.wam.omega)

                    Khat.append(Khat_samp)

                    # Check Stability
                    if any(np.real(np.roots(K_Den)) > 0):
                        print('WARNING: The system representing ' + idDOF(i)+'->'+idDOF(j) + ' is UNSTABLE')


        # Save models
        self.sys    = sys
        self.sysDOF = sysDOF
        self.Khat   = Khat


    def ident_memory(self,om,Plot=False):

        ### Start of ident_retardation_FD.m
        # Scale Data for ID
        K = self.K_ij

        scale = max(abs(K))

        K_bar = K/scale
        F_resp  = K_bar / (1j * om)      # remove zero at s=0 from the data

        # Frequency response fitting
        ord_den = self.order
        ord_num = self.order - 2


        ### Start of fit_siso_resp
        # Using iterative method with frequency dependent weighting

        Weight = np.ones(np.shape(om))

        for iter in range(0,20):
            b,a = invfreqs(F_resp, om, ord_num, ord_den, wf=Weight)
            a   = makeStable(a)    # DZ: I think this is potentially problematic, can lead to divergent solutions
            
            Weight = 1/abs(np.polyval(a,1j*om))**2

        ### End of fit_siso_resp

        _ , F_hat = freqs(b,a,worN=om)


        # rescale and incorporate zero
        b   = scale * np.concatenate([b,np.zeros(1)])

        _,K_hat     = freqs(b,a,worN=om)

        if Plot:
            

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.plot(om,abs(K),'o',om,abs(K_hat))
            # ax1.title(idDOF)
            

            ax2 = fig.add_subplot(212)
            ax2.plot(om,np.angle(K),'o',om,np.angle(K_hat))

            plt.show()

        return b, a, K_hat


    def visualizeFits(self):

        fig = []

        for q in range(len(self.sys)):

            normalIdx = np.array(self.sysDOF[q]) + 1
            sub = str(normalIdx[1]) + str(normalIdx[0])

            plt.figure()
            plt.plot(self.wam.omega,np.real(self.wam.K[self.sysDOF[q][0],self.sysDOF[q][1],:]),'o',label='K_'+sub)
            plt.plot(self.wam.omega,np.real(self.Khat[q]),label= 'Khat_'+sub)

            plt.title(idDOF(self.sysDOF[q][0])+'->'+idDOF(self.sysDOF[q][1])+ ' Transfer Function')
            
            plt.grid(b=True)
            # plt.rc('text', usetex=True)
            plt.legend()
            fig.append(plt.gcf())

        return fig

        # print('here')

class TimeDomain_Fit(object):

    def __init__(self,**kwargs):
        
        # Default Parameters
        self.MaxOrder   = 20
        self.FitTol     = 0.99

        self.dT               = 0.025
        self.WaveTMax         = 3600
        self.wam              = None
        self.tc               = 12


        # Initialization
        self.K_fit = []
        self.fig_list = []

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        # Impulse response function & time
        self.causalIRF()

    def causalIRF(self,Plot=False):
        # Upsample FRF
        dt      = self.dT
        Tmax    = self.WaveTMax
        dw      = 2 * np.pi / Tmax

        ww      = np.arange(0,np.pi/dt+dw,dw)

        K_w     = interpolate.interp1d(self.wam.omega,self.wam.K_dir,axis=0,bounds_error=False,fill_value=0)
        K       = K_w(ww)

        # Form 2-sided FRF
        K_conj  = np.conj(np.flip(K,axis=0))

        

        K_2sided = np.concatenate((K,K_conj[:-1]))

        # Take IFFT & Shift
        # Kt_2shift   = np.fft.fftshift(K_2sided,axes=0)
        Kt_2sided   = np.real(np.fft.ifft(K_2sided,axis=0)/dt)

        Kt          = np.fft.ifftshift(Kt_2sided,axes=0)


        tt          = np.arange(-Tmax/2,Tmax/2+dt,dt)

        
            

        # Find Causal Time Shift
        # where abs(Kt) < .025 * max(abs(Kt))
        frac = .25e-1
        thresh  = frac * np.amax(abs(np.real(Kt)))
        t_first = []
        for iDOF in range(0,6):
            t_first.append(tt[np.where(abs(Kt[:,iDOF]) > thresh)[0]-1])

        tc_auto = min([t0[0] for t0 in t_first if t0.any()])
        print('Suggested tc = {:.4f} sec.'.format(-tc_auto))
        

        # tc  = -tc_auto                   # time to shift by
        print('Using tc = {:.4f} sec.'.format(self.tc))

        Nc  = math.ceil(self.tc/dt)        # samples to shift by

        startInd    = math.floor(len(Kt)/2) - Nc

        totTimeOut  = 42.2 # seconds, 2 * tc?
        nSampOut    = math.floor(totTimeOut/dt)

        K_c     = Kt[startInd:startInd+nSampOut]
        tt_c    = tt[startInd:startInd+nSampOut] + self.tc


        if Plot:
            fig, axs = plt.subplots(6)

            for iDOF in range(0,6):
                axs[iDOF].plot(tt,np.real(Kt[:,iDOF]))
                axs[iDOF].plot(tt_c,np.real(K_c[:,iDOF]))
                axs[iDOF].set_xlim((-30,30))
                axs[iDOF].set_ylabel('K'+str(iDOF+1)+'(t)')

                if iDOF < 5:
                    axs[iDOF].set_xticklabels([])
            
            plt.show()

        # Save Data
        self.Kt     = K_c
        self.tt     = tt_c


    def fit(self,Plot=True,verbose=False):

        dt = np.mean(np.diff(self.tt))
        t_fit   = np.arange(0.,40.3,dt)


        for iDOF, Ki in enumerate(np.transpose(self.Kt)):

            # Filter Ki
            windowSize = 4
            b   = (1/windowSize) * np.ones(4)
            Ki_smooth = lfilter(b,1,Ki)

            A,B,C,D     = imp2ss(Ki_smooth,dt,.01)
            D           = np.zeros((1,1))
            if A.size:
                t_og, y_og  = impulse((A,B,C*dt,D),T=t_fit)
                y_og[0]     = 0  # sometime spurious data
            else:  # doesn't work with empty matrices
                t_og = t_fit
                y_og = np.zeros(t_fit.shape)


            if False:
                plt.figure()
                plt.plot(self.tt,Ki_smooth,label='Original IRF')
                plt.plot(t_og,y_og,label='High order IRF, {} states'.format(np.shape(A)[0]))
                plt.legend()
                plt.show()

            # Model Reduction
            order   = 1
            R2      = 0.

            # Form high-order state space system
            K_ho    = StateSpace(A,B,C*dt,D)

            t_ho, y_ho      = impulse_response(K_ho,T=t_fit)
            y_ho[0]         = 0

            # Is scipy impulse() the same as control.impulse_response?
            # if False:  # same, checked
            #     t_con, y_con = impulse_response(K_ho,T=t_fit)

            #     plt.figure(4)
            #     plt.plot(t,y,label='scipy impulse()')
            #     plt.plot(t_con,y_con,label='control impulse_response()')
            #     plt.legend()
            #     plt.show()

            # Increase model order until R2 adequate or order limit reached
            while order < self.MaxOrder and R2 < self.FitTol:
                if K_ho.A.size:  # is not empty
                    K_lo    = balred(K_ho,order)
                else:   # return same system
                    K_lo    = K_ho

                t_lo, y_lo  = impulse_response(K_lo,T=t_fit)

                order += 1

                R2  = computeR2(y_og,y_lo)
                if verbose:
                    print('order: {}, R2: {:4f}'.format(order,R2))

                # if False:
                #     plt.figure(5)
                #     plt.plot(t_ho,y_ho,label='high order')
                #     plt.plot(t_lo,y_lo,label='low order')
                #     plt.legend()
                #     plt.show()

            # Save Matrices for Outputting
            self.K_fit.append(K_lo)

            # Save Plots
            plt.figure(iDOF+3)
            plt.plot(self.tt,Ki_smooth,label='Original IRF')
            plt.plot(t_ho,y_ho,label='High order IRF, {} states'.format(np.shape(A)[0]))
            plt.plot(t_lo,y_lo,label='Low order IRF, {} states'.format(K_lo.nstates))
            plt.xlabel('Time (s)')
            plt.ylabel('K_{}(t)'.format(iDOF+1))
            plt.legend()

            self.fig_list.append(plt.gcf())
        

def imp2ss(K,dt,tol):
    # impulse to state space using Kung's SVD algorithm
    # SISO for now
    # porting from matlab, pls excuse 

    y = K
    nn = len(y)

    h = hankel(y[1:])
    rh, ch = np.shape(h)

    U, s, Vh = svd(h)
    # V       = np.transpose(V)       # to match matlab    

    # check svd
    h_svd = U @ np.diag(s) @ Vh
    V       = Vh.T.conj()

    tol = tol * s[0]

    if True:
        plt.figure(2)
        plt.plot(y)

    # find nx & totbnd
    tail = np.concatenate((np.convolve(np.ones(nn),s),np.zeros(1)))
    tail = tail[nn-1:]
    nx   = np.where(2*tail<= tol)
    nx   = nx[0][0] - 1
    if nx < 0:
        nx = 0
    totbnd  = 2 * tail[nx+1]

    # truncate and partition singular vector matrices U,V
    # ny = 1, and nu = 1 for SISO
    # matlab code diverges from matlab documentation...

    if True:  #matlab code
        u1  = U[0:-1,:nx]
        v1  = V[0:-1,:nx]
        u2  = U[1:,:nx]
        u3  = U[-1,:nx]

        sqs     = np.expand_dims(np.sqrt(s[:nx]),1)
        invss   = 1 / sqs

        if False:
            plt.figure(2)
            plt.plot(s)
            plt.show()

        # Kung's formula for the reduced model
        # matlab code diverges from matlab documentation...
        ubar    = u1.T.conj() @ u2
        a       = ubar * (invss @ sqs.T.conj())
        b       = sqs * np.expand_dims(v1[0,:],1)  # maybe .conj()
        c       = np.expand_dims(u1[0,:],1).T * sqs.T.conj()
        d       = 0
    else:  # following matlab documentation
        u11     = np.expand_dims(U[0,:nx],1).T
        u12     = U[1:-2,:nx]
        u13     = np.expand_dims(U[-1,:nx],1).T

        V       = Vh.T.conj()
        v11     = np.expand_dims(V[0,:nx],1).T
        v12     = V[1:-2,:nx]
        v13     = np.expand_dims(V[-1,:nx],1).T

        ubar    = np.concatenate((u11,u12)).T @ np.concatenate((u12,u13))
        sqs     = np.sqrt(s[:nx])
        invss   = np.diag(1 / sqs)

        a       = invss @ ubar @ invss
        b       = invss @ v11.T.conj()
        c       = u11 @ invss
        d       = 0

    # expand dims so we can translate into lin sys
    # b       = np.expand_dims(b,1)
    # c       = np.transpose(np.expand_dims(c,1))
    d       = np.expand_dims(np.expand_dims(d,0),1)

    A, B, C, D  = bilin(a,b,c,d,dt)
        
    return A, B, C, D

def bilin(A,B,C,D,dt):
    a = dt/2
    b = 1
    c = -a
    d = 1

    ra, ca = np.shape(A)
    iidd    = np.linalg.inv(a*np.eye(ra) - c * A)

    Ad  = np.matmul(b * A - d * np.eye(ra), iidd)
    Bd  = (a*b - c*d) * np.matmul(iidd,B)
    Cd  = np.matmul(C,iidd)
    Dd  = D + c*np.matmul(np.matmul(C,iidd),B)

    return Ad, Bd, Cd, Dd

def computeError(K,K_hat,om,A_inf):

    B_ij        = np.real(K)
    A_ij        = np.imag(K) / om + A_inf

    Bhat_ij     = np.real(K_hat)
    Ahat_ij     = np.imag(K_hat) / om + A_inf

    # using FDIRadMod.m notation to compute R^2 for both A and B
    sseb        = np.matmul(np.transpose((B_ij - Bhat_ij)),(B_ij - Bhat_ij))
    sstb        = np.matmul(np.transpose((B_ij - np.mean(B_ij))),(B_ij - np.mean(B_ij)))
    r2b         = 1- sseb/sstb

    ssea        = np.matmul(np.transpose((A_ij - Ahat_ij)),(A_ij - Ahat_ij))
    ssta        = np.matmul(np.transpose((A_ij - np.mean(A_ij))),(A_ij - np.mean(A_ij)))
    r2a         = 1- ssea/ssta

    return r2b, r2a

def computeR2(x,y):
    res     = x - y
    m       = np.mean(x)
    sst     = np.sum((x-m)*(x-m).conj())
    if sst:
        R2      = 1 - sum(res**2)/sst
    else:
        R2      = float('nan')

    return R2

def makeStable(p):
    # check that polynomial has roots in the RHP, if not, flip
    r = np.roots(p)
    r = [-root if np.real(root) > 0 else root for root in r]

    p = np.poly(r)

    return p

                
def idDOF(dof):
    DOFs    = ['Surge','Sway','Heave','Roll','Pitch','Yaw']

    return DOFs[dof]