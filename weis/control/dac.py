import numpy as np
import os, sys, subprocess
import copy
from openmdao.api import ExplicitComponent
from wisdem.ccblade.ccblade import CCAirfoil, CCBlade
from wisdem.ccblade.Polar import Polar, _find_alpha0, _find_slope, _alpha_window_in_bounds
import csv  # for exporting airfoil polar tables
import matplotlib.pyplot as plt
import time

import multiprocessing as mp
from functools import partial
from wisdem.commonse.mpi_tools import MPI

def runXfoil(xfoil_path, x, y, Re, AoA_min=-9, AoA_max=25, AoA_inc=0.5, Ma=0.0, multi_run=False, MPI_run=False):
    #This function is used to create and run xfoil simulations for a given set of airfoil coordinates

    # Set initial parameters needed in xfoil
    numNodes   = 260 # number of panels to use (260...but increases if needed)
    #dist_param = 0.15 # TE/LE panel density ratio (0.15)
    dist_param = 0.12 #This is current value that i am trying to help with convergence (!bem)
    #IterLimit = 100 # Maximum number of iterations to try and get to convergence
    IterLimit = 10 #This decreased IterLimit will speed up analysis (!bem)
    #panelBunch = 1.5 # Panel bunching parameter to bunch near larger changes in profile gradients (1.5)
    panelBunch = 1.6 #This is the value I am currently using to try and improve convergence (!bem)
    #rBunch = 0.15 # Region to LE bunching parameter (used to put additional panels near flap hinge) (0.15)
    rBunch = 0.08 #This is the current value that I am using (!bem)
    XT1 = 0.55 # Defining left boundary of bunching region on top surface (should be before flap)
    # XT1 = 1.0
    #XT2 = 0.85 # Defining right boundary of bunching region on top surface (should be after flap)
    XT2 = 0.9 #This is the value I am currently using (!bem)
    # XT2 = 1.0
    XB1 = 0.55 # Defining left boundary of bunching region on bottom surface (should be before flap)
    # XB1 = 1.0
    #XB2 = 0.85 # Defining right boundary of bunching region on bottom surface (should be after flap)
    XB2 = 0.9 #This is the current value that I am using (!bem)
    # XB2 = 1.0
    runFlag = True # Flag used in error handling
    dfdn = -0.5 # Change in angle of attack during initialization runs down to AoA_min
    runNum = 0 # Initialized run number
    dfnFlag = False # This flag is used to determine if xfoil needs to be re-run if the simulation fails due to convergence issues at low angles of attack

    # Set filenames 
    # if multi_run or MPI_run:
    pid = mp.current_process().pid
    print('Running xfoil on PID = {}'.format(pid))

    xfoil_rundir = 'xfoil_run_p{}'.format(pid)
    if not os.path.exists(xfoil_rundir):
        os.makedirs(xfoil_rundir)
    LoadFlnmAF    = os.path.join(xfoil_rundir,'airfoil_p{}.txt'.format(pid))
    saveFlnmPolar = os.path.join(xfoil_rundir,'Polar_p{}.txt'.format(pid))
    xfoilFlnm     = os.path.join(xfoil_rundir,'xfoil_input_p{}.txt'.format(pid))
    NUL_fname     = os.path.join(xfoil_rundir,'NUL_p{}'.format(pid))

    # if MPI_run:
    #     rank = MPI.COMM_WORLD.Get_rank()
    #     LoadFlnmAF = 'airfoil_r{}.txt'.format(rank) # This is a temporary file that will be deleted after it is no longer needed
    #     saveFlnmPolar = 'Polar_r{}.txt'.format(rank) # file name of outpur xfoil polar (can be useful to look at during debugging...can also delete at end if you don't want it stored)
    #     xfoilFlnm  = 'xfoil_input_r{}.txt'.format(rank) # Xfoil run script that will be deleted after it is no longer needed
    # else:
    #     LoadFlnmAF = 'airfoil.txt' # This is a temporary file that will be deleted after it is no longer needed
    #     saveFlnmPolar = 'Polar.txt' # file name of outpur xfoil polar (can be useful to look at during debugging...can also delete at end if you don't want it stored)
    #     xfoilFlnm  = 'xfoil_input.txt' # Xfoil run script that will be deleted after it is no longer needed
    #     NUL_fname = 'NUL'
    t0 = time.time()
    while runFlag:
        # Cleaning up old files to prevent replacement issues
        if os.path.exists(saveFlnmPolar):
            os.remove(saveFlnmPolar)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(LoadFlnmAF):
            os.remove(LoadFlnmAF)
        if os.path.exists(NUL_fname):
            os.remove(NUL_fname)
            
        # Writing temporary airfoil coordinate file for use in xfoil
        dat=np.array([x,y])
        np.savetxt(LoadFlnmAF, dat.T, fmt=['%f','%f'])

        # %% Writes the Xfoil run script to read in coordinates, create flap, re-pannel, and create polar
        # Create the airfoil with flap
        fid = open(xfoilFlnm,"w")
        fid.write("PLOP \n G \n\n") # turn off graphics
        fid.write("LOAD \n")
        fid.write( LoadFlnmAF + "\n" + "\n") # name of .txt file with airfoil coordinates
        # fid.write( self.AFName + "\n") # set name of airfoil (internal to xfoil)
        fid.write("GDES \n") # enter into geometry editing tools in xfoil
        fid.write("UNIT \n") # normalize profile to unit chord
        fid.write("EXEC \n \n") # move buffer airfoil to current airfoil

        # Re-panel with specified number of panes and LE/TE panel density ratio
        fid.write("PPAR\n")
        fid.write("N \n" )
        fid.write(str(numNodes) + "\n")
        fid.write("P \n") # set panel bunching parameter
        fid.write(str(panelBunch) + " \n")
        fid.write("T \n") # set TE/LE panel density ratio
        fid.write( str(dist_param) + "\n")
        fid.write("R \n") # set region panel bunching ratio
        fid.write(str(rBunch) + " \n")
        fid.write("XT \n") # set region panel bunching bounds on top surface
        fid.write(str(XT1) +" \n" + str(XT2) + " \n")
        fid.write("XB \n") # set region panel bunching bounds on bottom surface
        fid.write(str(XB1) +" \n" + str(XB2) + " \n")
        fid.write("\n\n")

        # Set Simulation parameters (Re and max number of iterations)
        fid.write("OPER\n")
        fid.write("VISC \n")
        fid.write( str(Re) + "\n") # this sets Re to value specified in yaml file as an input
        #fid.write( "5000000 \n") # bem: I was having trouble geting convergence for some of the thinner airfoils at the tip for the large Re specified in the yaml, so I am hard coding in Re (5e6 is the highest I was able to get to using these paneling parameters)
        fid.write("MACH\n")
        fid.write(str(Ma)+" \n")
        fid.write("ITER \n")
        fid.write( str(IterLimit) + "\n")

        # Run simulations for range of AoA

        if dfnFlag: # bem: This if statement is for the case when there are issues getting convergence at AoA_min.  It runs a preliminary set of AoA's down to AoA_min (does not save them)
            for ii in range(int((0.0-AoA_min)/AoA_inc+1)):
                fid.write("ALFA "+ str(0.0-ii*float(AoA_inc)) +"\n")

        fid.write("PACC\n\n\n") #Toggle saving polar on
        # fid.write("ASEQ 0 " + str(AoA_min) + " " + str(dfdn) + "\n") # The preliminary runs are just to get an initialize airfoil solution at min AoA so that the actual runs will not become unstable

        for ii in range(int((AoA_max-AoA_min)/AoA_inc+1)): # bem: run each AoA seperately (makes polar generation more convergence error tolerant)
            fid.write("ALFA "+ str(AoA_min+ii*float(AoA_inc)) +"\n")

        #fid.write("ASEQ " + str(AoA_min) + " " + "16" + " " + str(AoA_inc) + "\n") #run simulations for desired range of AoA using a coarse step size in AoA up to 16 deg
        #fid.write("ASEQ " + "16.5" + " " + str(AoA_max) + " " + "0.1" + "\n") #run simulations for desired range of AoA using a fine AoA increment up to final AoA to help with convergence issues at high Re
        fid.write("PWRT\n") #Toggle saving polar off
        fid.write(saveFlnmPolar + " \n \n")
        fid.write("QUIT \n")
        fid.close()

        # Run the XFoil calling command
        try:
            subprocess.run([xfoil_path], stdin=open(xfoilFlnm,'r'), stdout=open(NUL_fname, 'w'), timeout=300)
            dac_polar = np.loadtxt(saveFlnmPolar,skiprows=12)
        except subprocess.TimeoutExpired:
            print('XFOIL timeout on p{}'.format(pid)) 
            try: 
                dac_polar = np.loadtxt(saveFlnmPolar,skiprows=12) # Sometimes xfoil will hang up but still generate a good set of polars
            except:
                dac_polar = []  # in case no convergence was achieved
        except:
            dac_polar = []  # in case no convergence was achieved

        # Check for linear region
        try:
            window = _alpha_window_in_bounds(dac_polar[:,0],[-30, 30])
            alpha0 = _find_alpha0(np.array(dac_polar[:,0]), np.array(dac_polar[:,1]), window)
            window2 = [alpha0, alpha0+4]
            window2 = _alpha_window_in_bounds(dac_polar[:,0], [alpha0, alpha0 + 4])
            # Max and Safety checks
            s1, _ = _find_slope(dac_polar[:,0], dac_polar[:,1], xi=alpha0, window=window2, method="max")
            if len(dac_polar[:,1]) > 10:
                s2, _ = _find_slope(dac_polar[:,0], dac_polar[:,1], xi=alpha0, window=window2, method="finitediff_1c")
            lin_region_len = len(np.where(dac_polar[:,0] < alpha0)[0])
            lin_region_len_idx = np.where(dac_polar[:,0] < alpha0)[0][-1]
            if lin_region_len_idx < 1:
                lin_region_len = 0 
                raise IndexError('Invalid index for linear region.')
        except (IndexError, TypeError):
            lin_region_len = 0
            
        if lin_region_len < 1:
            print('Error: No linear region detected for XFOIL run on p{}'.format(pid))

        # Error handling (re-run simulations with more panels if there is not enough data in polars)
        if np.size(dac_polar) < 3 or lin_region_len < 1: # This case is if there are convergence issues or bad angles of attack
            plen = 0
            a0 = 0
            a1 = 0
            dfdn = -0.25 # decrease AoA step size during initialization to try and get convergence in the next run
            dfnFlag = True # Set flag to run initialization AoA down to AoA_min
            print('XFOIL convergence issues on p{}'.format(pid))
        else:
            plen = len(dac_polar[:,0]) # Number of AoA's in polar
            a0 = dac_polar[-1,0] # Maximum AoA in Polar (deg)
            a1 = dac_polar[0,0] # Minimum AoA in Polar (deg)
            dfnFlag = False # Set flag so that you don't need to run initialization sequence

        if a0 > 19. and plen >= 40 and a1 < -12.5: # The a0 > 19 is to check to make sure polar entered into stall regiem plen >= 40 makes sure there are enough AoA's in polar for interpolation and a1 < -15 makes sure polar contains negative stall.
            runFlag = False # No need ro re-run polar
            if numNodes > 310:
                print('Xfoil completed after {} attempts on run on p{}.'.format(runNum+1, pid))
        else:
            numNodes += 50 # Re-run with additional panels
            # AoA_inc *= 0.5
            runNum += 1 # Update run number
            # AoA_min = -9
            # AoA_max = 25
            # if numNodes > 480:
            if runNum > 10:
                # Warning('NO convergence in XFoil achieved!')
                print('No convergence in XFOIL achieved on p{}!'.format(pid))
                if not os.path.exists('xfoil_errorfiles'):
                    os.makedirs('xfoil_errorfiles')
                try:
                    os.rename(xfoilFlnm, os.path.join('xfoil_errorfiles', xfoilFlnm))
                except:
                    pass
                try:
                    os.rename(saveFlnmPolar, os.path.join('xfoil_errorfiles', saveFlnmPolar))
                except:
                    pass
                try:
                    os.rename(LoadFlnmAF, os.path.join('xfoil_errorfiles', LoadFlnmAF))
                except:
                    pass
                try:
                    os.rename(NUL_fname, os.path.join('xfoil_errorfiles', NUL_fname))
                except:
                    pass
                
                break
            print('Refining paneling to ' + str(numNodes) + ' nodes')

    # Load back in polar data to be saved in instance variables
    # dac_polar = np.loadtxt(LoadFlnmAF,skiprows=12) # (note, we are assuming raw Xfoil polars when skipping the first 12 lines)
    # self.af_dac_polar = dac_polar
    # self.dac_polar_flnm = saveFlnmPolar # Not really needed unless you keep the files and want to load them later

    # Delete Xfoil run script file
    if os.path.exists(xfoilFlnm):
        os.remove(xfoilFlnm)
    if os.path.exists(saveFlnmPolar): # bem: For now leave the files, but eventually we can get rid of them (remove # in front of commands) so that we don't have to store them
        os.remove(saveFlnmPolar)
    if os.path.exists(LoadFlnmAF):
        os.remove(LoadFlnmAF)
    if os.path.exists(NUL_fname):
        os.remove(NUL_fname)
    if os.path.exists(xfoil_rundir):
        os.rmdir(xfoil_rundir)

    print('Xfoil calls on p{} completed in {} seconds'.format(pid, time.time()-t0))

    return dac_polar

class RunXFOIL(ExplicitComponent):
    # Openmdao component to run XFOIL and re-compute polars
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        rotorse_options = self.options['modeling_options']['WISDEM']['RotorSE']
        self.n_span        = n_span     = rotorse_options['n_span']
        self.n_dac         = n_dac      = rotorse_options['n_dac'] 
        self.n_tab         = rotorse_options['n_tab']
        self.n_aoa         = n_aoa      = rotorse_options['n_aoa'] # Number of angle of attacks
        self.n_Re          = n_Re       = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab      = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_xy          = n_xy       = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        
        if n_dac > 0: #May need to add error handeling
            self.dac_model = dac_model = self.options['modeling_options']['ROSCO']['DAC_Model']
        
        # Use openfast cores for parallelization of xfoil 
        xfoilpref = self.options['modeling_options']['Level3']['xfoil']
        self.xfoil_path = xfoilpref['path']

        try:
            if xfoilpref['run_parallel']:
                self.cores = mp.cpu_count()
            else:
                self.cores = 1
        except KeyError:
            self.cores = 1
        
        if MPI and self.options['modeling_options']['Level3']['flag'] and not self.options['opt_options']['driver']['optimization']['flag']:
            self.mpi_comm_map_down = self.options['modeling_options']['General']['openfast_configuration']['mpi_comm_map_down']

        # Inputs blade outer shape
        self.add_input('s',          val=np.zeros(n_span),                      desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('r',             val=np.zeros(n_span), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)),     desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.')
        self.add_input('chord',         val=np.zeros(n_span), units='m',   desc='chord length at each section')

        # Inputs DAC Devices
        self.add_input('span_end',   val=np.zeros(n_dac),                  desc='1D array of the positions along blade span where the DAC device(s) end. Only values between 0 and 1 are meaningful.')
        self.add_input('span_ext',   val=np.zeros(n_dac),                  desc='1D array of the extensions along blade span of the DAC device(s). Only values between 0 and 1 are meaningful.')
        self.add_input('chord_start',val=np.zeros(n_dac),                  desc='1D array of the positions along chord where the DAC device(s) start. Only values between 0 and 1 are meaningful.')
        self.add_input('delta_max_pos', val=np.zeros(n_dac),               desc='1D array of the max device value (i.e. deflection angle, spoiler height, etc.) of the distributed aerodynamic device. (all angles are in rad, but this value could also have other units)') #bem: I got rid of unit of radians
        self.add_input('delta_max_neg', val=np.zeros(n_dac),               desc='1D array of the min device value (i.e. deflection angle, spoiler height, etc.) of the distributed aerodynamic device. (all angles are in rad, but this value could also have other units)')

        # Inputs control
        self.add_input('max_TS',         val=0.0, units='m/s',     desc='Maximum allowed blade tip speed.')
        self.add_input('rated_TSR',      val=0.0,                  desc='Constant tip speed ratio in region II.')

        # Inputs environment
        self.add_input('rho_air',      val=1.225,        units='kg/m**3',    desc='Density of air')
        self.add_input('mu_air',       val=1.81e-5,      units='kg/(m*s)',   desc='Dynamic viscosity of air')
        self.add_input('speed_sound_air',  val=340.,     units='m/s',        desc='Speed of sound in air.')

        # Inputs polars
        self.add_input('aoa',       val=np.zeros(n_aoa),        units='rad',    desc='1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        self.add_input('cl_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cd_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cm_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')

        # Outputs DAC geometry
        self.add_output('span_start', val=np.zeros(n_dac),                  desc='1D array of the positions along blade span where the DAC device(s) start. Only values between 0 and 1 are meaningful.')
        
        # Output polars
        self.add_output('cl_interp_dac',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cd_interp_dac',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cm_interp_dac',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        # self.add_output('dac_param',      val=np.zeros((n_span, n_Re, n_tab)), units = 'deg',   desc='3D array with the flap angles of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')
        self.add_output('dac_param',      val=np.zeros((n_span, n_Re, n_tab)),    desc='3D array with the DAC parameter values of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')

        self.add_output('Re_loc',           val=np.zeros((n_span, n_Re, n_tab)),   desc='3D array with the Re. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')
        self.add_output('Ma_loc',           val=np.zeros((n_span, n_Re, n_tab)),   desc='3D array with the Mach number. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')

        # initialize saved data polar data. 
        # - This is filled if we're not changing the DAC values, so we don't need to re-run xfoil every time
        self.saved_polar_data = {}

    def compute(self, inputs, outputs):

        # If trailing edge flaps are present, compute the perturbed profiles with XFOIL TODO bem: need to account for other DAC devices.
        self.dac_profiles = [{} for i in range(self.n_span)]
        if self.n_dac > 0:
            try:
                from scipy.ndimage import gaussian_filter
            except:
                print('Cannot import the library gaussian_filter from scipy. Please check the conda environment and potential conflicts between numpy and scipy')
            
            # Make sure flaps are viable
            if inputs['span_end'] > 1.0:
                print('WARNING: DAC device end is off the blade! Moving it to the end of the blade.')
            if self.options['opt_options']['design_variables']['control']['dac']['dac_end']['flag']: 
                np.clip(inputs['span_end'], 
                        self.options['opt_options']['design_variables']['control']['dac']['dac_end']['min'], 
                        self.options['opt_options']['design_variables']['control']['dac']['dac_end']['max']
                        )

            outputs['span_start'] = inputs['span_end'] - inputs['span_ext']

            xfoil_kw = {}
            if MPI:
                xfoil_kw['MPI_run'] = True
            elif self.cores > 1:
                xfoil_kw['multi_run'] = True
            for i in range(self.n_span):
                # Loop through the DAC devices specified in yaml file
                for k in range(self.n_dac):
                    # Only create dac geometries where the yaml file specifies there is a dac device (Currently going to nearest blade station location)
                    if inputs['s'][i] >= outputs['span_start'][k] and inputs['s'][i] <= inputs['span_end'][k]: 
                        self.dac_profiles[i]['dac_param']= []
                        # Initialize the profile coordinates to zeros
                        self.dac_profiles[i]['coords']     = np.zeros([self.n_xy,2,self.n_tab]) 
                            # Ben:I am not going to force it to include delta=0.  If this is needed, a more complicated way of getting flap deflections to calculate is needed.
                        dac_param = np.linspace(inputs['delta_max_neg'][k],inputs['delta_max_pos'][k],self.n_tab) #* 180. / np.pi # TODO bem: should change to use radians instead of degrees
                        # Loop through the flap angles
                        for ind, fa in enumerate(dac_param):
                            # NOTE: negative flap angles are deflected to the suction side, i.e. positively along the positive z- (radial) axis
                            af_dac = CCAirfoil(np.array([1,2,3]), np.array([100]), np.zeros(3), np.zeros(3), np.zeros(3), inputs['coord_xy_interp'][i,:,0], inputs['coord_xy_interp'][i,:,1], "Profile"+str(i)) # bem:I am creating an airfoil name based on index...this structure/naming convention is being assumed in CCAirfoil.runXfoil() via the naming convention used in CCAirfoil.af_dac_coords(). Note that all of the inputs besides profile coordinates and name are just dummy varaiables at this point.
                            if self.dac_model == 1:
                                af_dac.af_dac_coords(self.xfoil_path, fa* 180. / np.pi,  inputs['chord_start'][k],0.5,200, **xfoil_kw) #bem: the last number is the number of points in the profile.  It is currently being hard coded at 200 but should be changed to make sure it is the same number of points as the other profiles
                            else:
                                af_dac.af_dac_coords_lems(inputs['coord_xy_interp'][i,:,0], inputs['coord_xy_interp'][i,:,1]) # Just passing zeros...doesn't matter for generic dac model
                            # self.flap_profiles[i]['coords'][:,0,ind] = af_flap.af_flap_xcoords # x-coords from xfoil file with flaps
                            # self.flap_profiles[i]['coords'][:,1,ind] = af_flap.af_flap_ycoords # y-coords from xfoil file with flaps
                            # self.flap_profiles[i]['coords'][:,0,ind] = af_flap.af_flap_xcoords  # x-coords from xfoil file with flaps and NO gaussian filter for smoothing
                            # self.flap_profiles[i]['coords'][:,1,ind] = af_flap.af_flap_ycoords  # y-coords from xfoil file with flaps and NO gaussian filter for smoothing
                            try:
                                self.dac_profiles[i]['coords'][:,0,ind] = gaussian_filter(af_dac.af_dac_xcoords, sigma=1) # x-coords from xfoil file with flaps and gaussian filter for smoothing
                                self.dac_profiles[i]['coords'][:,1,ind] = gaussian_filter(af_dac.af_dac_ycoords, sigma=1) # y-coords from xfoil file with flaps and gaussian filter for smoothing
                            except:
                                self.dac_profiles[i]['coords'][:,0,ind] = af_dac.af_dac_xcoords
                                self.dac_profiles[i]['coords'][:,1,ind] = af_dac.af_dac_ycoords
                            self.dac_profiles[i]['dac_param'].append([])
                            self.dac_profiles[i]['dac_param'][ind] = fa # Putting in DAC parameter values to blade for each profile (can be used for debugging later)
                            #print('fa = ' + str(fa))

                        if False:
                            import pickle
                            f = open('dac_profiles.pkl', 'wb')
                            pickle.dump(self.dac_profiles, f)
                            f.close()
                            
                        # # ** The code below will plot the first three flap deflection profiles (in the case where there are only 3 this will correspond to max negative, zero, and max positive deflection cases)
                        # font = {'family': 'Times New Roman',
                        #         'weight': 'normal',
                        #         'size': 18}
                        # plt.rc('font', **font)
                        # plt.figure
                        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                        # # plt.plot(self.flap_profiles[i]['coords'][:,0,0], self.flap_profiles[i]['coords'][:,1,0], 'r',self.flap_profiles[i]['coords'][:,0,1], self.flap_profiles[i]['coords'][:,1,1], 'k',self.flap_profiles[i]['coords'][:,0,2], self.flap_profiles[i]['coords'][:,1,2], 'b')
                        # plt.plot(self.flap_profiles[i]['coords'][:, 0, 0],
                        #         self.flap_profiles[i]['coords'][:, 1, 0], '.r',
                        #         self.flap_profiles[i]['coords'][:, 0, 2],
                        #         self.flap_profiles[i]['coords'][:, 1, 2], '.b',
                        #         self.flap_profiles[i]['coords'][:, 0, 1],
                        #         self.flap_profiles[i]['coords'][:, 1, 1], '.k')
                        
                        # # plt.xlabel('x')
                        # # plt.ylabel('y')
                        # plt.axis('equal')
                        # plt.axis('off')
                        # plt.tight_layout()
                        # plt.show()
                        # # # plt.savefig('temp/airfoil_polars/NACA63-self.618_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.211_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.241_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.301_flap_profiles.png', dpi=300)


        # # ----------------------------------------------------- #
        # # Determine airfoil polar tables blade sections #

        # #  ToDo: shape of blade['profile'] differs from self.flap_profiles <<< change to same shape
        # # only execute when flag_airfoil_polars = True
        # flag_airfoil_polars = False  # <<< ToDo get through Yaml in the future ?!?

        # if flag_airfoil_polars == True:
        #     # OUTDATED!!! - NJA

        #     af_orig_grid = blade['outer_shape_bem']['airfoil_position']['grid']
        #     af_orig_labels = blade['outer_shape_bem']['airfoil_position']['labels']
        #     af_orig_chord_grid = blade['outer_shape_bem']['chord']['grid']  # note: different grid than airfoil labels
        #     af_orig_chord_value = blade['outer_shape_bem']['chord']['values']

        #     for i_af_orig in range(len(af_orig_grid)):
        #         if af_orig_labels[i_af_orig] != 'circular':
        #             print('Determine airfoil polars:')

        #             # check index of chord grid for given airfoil radial station
        #             for i_chord_grid in range(len(af_orig_chord_grid)):
        #                 if af_orig_chord_grid[i_chord_grid] == af_orig_grid[i_af_orig]:
        #                     c = af_orig_chord_value[i_chord_grid]  # get chord length at current radial station of original airfoil
        #                     c_index = i_chord_grid


        #             flag_coord = 3  # Define which blade airfoil outer shapes coordinates to use (watch out for consistency throughout the model/analysis !!!)
        #             #  Get orig coordinates (too many for XFoil)
        #             if flag_coord == 1:
        #                 x_af = self.wt_ref['airfoils'][1]['coordinates']['x']
        #                 y_af = self.wt_ref['airfoils'][1]['coordinates']['y']


        #             #  Get interpolated coords
        #             if flag_coord == 2:
        #                 x_af = blade['profile'][:,0,c_index]
        #                 y_af = blade['profile'][:,1,c_index]


        #             # create coords using ccblade and calling XFoil in order to be consistent with the flap method
        #             if flag_coord == 3:
        #                 flap_angle = 0  # no te-flaps !
        #                 af_temp = CCAirfoil(np.array([1,2,3]), np.array([100]), np.zeros(3), np.zeros(3), np.zeros(3), blade['profile'][:,0,c_index],blade['profile'][:,1,c_index], "Profile"+str(c_index)) # bem:I am creating an airfoil name based on index...this structure/naming convention is being assumed in CCAirfoil.runXfoil() via the naming convention used in CCAirfoil.af_flap_coords(). Note that all of the inputs besides profile coordinates and name are just dummy varaiables at this point.
        #                 af_temp.af_flap_coords(self.xfoil_path, flap_angle,  0.8, 0.5, 200) #bem: the last number is the number of points in the profile.  It is currently being hard coded at 200 but should be changed to make sure it is the same number of points as the other profiles
        #                 # x_af = af_temp.af_flap_xcoords
        #                 # y_af = af_temp.af_flap_ycoords

        #                 x_af = gaussian_filter(af_temp.af_flap_xcoords, sigma=1)  # gaussian filter for smoothing (in order to be consistent with flap capabilities)
        #                 y_af = gaussian_filter(af_temp.af_flap_ycoords, sigma=1)  # gaussian filter for smoothing (in order to be consistent with flap capabilities)


        #             rR = af_orig_grid[i_af_orig]  # non-dimensional blade radial station at cross section
        #             R = blade['pf']['r'][-1]  # blade (global) radial length
        #             tsr = blade['config']['tsr']  # tip-speed ratio
        #             maxTS = blade['assembly']['control']['maxTS']  # max blade-tip speed (m/s) from yaml file
        #             KinVisc = blade['environment']['air_data']['KinVisc']  # Kinematic viscosity (m^2/s) from yaml file
        #             SpdSound = blade['environment']['air_data']['SpdSound']  # speed of sound (m/s) from yaml file
        #             Re_af_orig_loc = c * maxTS * rR / KinVisc
        #             Ma_af_orig_loc = maxTS * rR / SpdSound

        #             print('Run xfoil for airfoil ' + af_orig_labels[i_af_orig] + ' at span section r/R = ' + str(rR) + ' with Re equal to ' + str(Re_af_orig_loc) + ' and Ma equal to ' + str(Ma_af_orig_loc))
        #             # if af_orig_labels[i_af_orig] == 'NACA63-618':  # reduce AoAmin for (thinner) airfoil at the blade tip due to convergence reasons in XFoil
        #             #     data = self.runXfoil(x_af, y_af_orig, Re_af_orig_loc, -13.5, 25., 0.5, Ma_af_orig_loc)
        #             # else:
        #             data = self.runXfoil(x_af, y_af, Re_af_orig_loc, -20., 25., 0.5, Ma_af_orig_loc)

        #             oldpolar = Polar(Re_af_orig_loc, data[:, 0], data[:, 1], data[:, 2], data[:, 4])  # p[:,0] is alpha, p[:,1] is Cl, p[:,2] is Cd, p[:,4] is Cm

        #             polar3d = oldpolar.correction3D(rR, c/R, tsr)  # Apply 3D corrections (made sure to change the r/R, c/R, and tsr values appropriately when calling AFcorrections())
        #             cdmax = 1.5
        #             polar = polar3d.extrapolate(cdmax)  # Extrapolate polars for alpha between -180 deg and 180 deg

        #             cl_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cl)
        #             cd_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cd)
        #             cm_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cm)

        #             # --- PROFILE ---#
        #             # write profile (that was input to XFoil; although previously provided in the yaml file)
        #             with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_profile.csv', 'w') as profile_csvfile:
        #                 profile_csvfile_writer = csv.writer(profile_csvfile, delimiter=',')
        #                 profile_csvfile_writer.writerow(['x', 'y'])
        #                 for i in range(len(x_af)):
        #                     profile_csvfile_writer.writerow([x_af[i], y_af[i]])

        #             # plot profile
        #             plt.figure(i_af_orig)
        #             plt.plot(x_af, y_af, 'k')
        #             plt.axis('equal')
        #             # plt.show()
        #             plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_profile.png')
        #             plt.close(i_af_orig)

        #             # --- CL --- #
        #             # write cl
        #             with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cl.csv', 'w') as cl_csvfile:
        #                 cl_csvfile_writer = csv.writer(cl_csvfile, delimiter=',')
        #                 cl_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cl'])
        #                 for i in range(len(cl_interp)):
        #                     cl_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cl_interp[i]])

        #             # plot cl
        #             plt.figure(i_af_orig)
        #             fig, ax = plt.subplots(1,1, figsize= (8,5))
        #             plt.plot(np.degrees(alpha), cl_interp, 'b')
        #             plt.xlim(xmin=-25, xmax=25)
        #             plt.grid(True)
        #             autoscale_y(ax)
        #             plt.xlabel('Angles of attack, deg')
        #             plt.ylabel('Lift coefficient')
        #             # plt.show()
        #             plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cl.png')
        #             plt.close(i_af_orig)

        #             # write cd
        #             with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cd.csv', 'w') as cd_csvfile:
        #                 cd_csvfile_writer = csv.writer(cd_csvfile, delimiter=',')
        #                 cd_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cd'])
        #                 for i in range(len(cd_interp)):
        #                     cd_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cd_interp[i]])

        #             # plot cd
        #             plt.figure(i_af_orig)
        #             fig, ax = plt.subplots(1,1, figsize= (8,5))
        #             plt.plot(np.degrees(alpha), cd_interp, 'r')
        #             plt.xlim(xmin=-25, xmax=25)
        #             plt.grid(True)
        #             autoscale_y(ax)
        #             plt.xlabel('Angles of attack, deg')
        #             plt.ylabel('Drag coefficient')
        #             # plt.show()
        #             plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cd.png')
        #             plt.close(i_af_orig)

        #             # write cm
        #             with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cm.csv', 'w') as cm_csvfile:
        #                 cm_csvfile_writer = csv.writer(cm_csvfile, delimiter=',')
        #                 cm_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cm'])
        #                 for i in range(len(cm_interp)):
        #                     cm_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cm_interp[i]])

        #             # plot cm
        #             plt.figure(i_af_orig)
        #             fig, ax = plt.subplots(1,1, figsize= (8,5))
        #             plt.plot(np.degrees(alpha), cm_interp, 'g')
        #             plt.xlim(xmin=-25, xmax=25)
        #             plt.grid(True)
        #             autoscale_y(ax)
        #             plt.xlabel('Angles of attack, deg')
        #             plt.ylabel('Torque coefficient')
        #             # plt.show()
        #             plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cm.png')
        #             plt.close(i_af_orig)

        #             # write additional information (Re, Ma, r/R)
        #             with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_add_info.csv', 'w') as csvfile:
        #                 csvfile_writer = csv.writer(csvfile, delimiter=',')
        #                 csvfile_writer.writerow(['Re', 'Ma', 'r/R'])
        #                 csvfile_writer.writerow([Re_af_orig_loc, Ma_af_orig_loc, rR])

        #             plt.close('all')
        # # ------------------------------------------------------------ #
        # Determine airfoil polar tables for blade sections with flaps #

        self.R        = inputs['r'][-1]  # Rotor radius in meters
        self.tsr      = inputs['rated_TSR']  # tip-speed ratio
        self.maxTS    = inputs['max_TS']  # max blade-tip speed (m/s) from yaml file
        self.KinVisc  = inputs['mu_air'] / inputs['rho_air']  # Kinematic viscosity (m^2/s) from yaml file
        self.SpdSound = inputs['speed_sound_air'] # speed of sound (m/s) from yaml file
        
        # Initialize
        cl_interp_dac = inputs['cl_interp']
        cd_interp_dac = inputs['cd_interp']
        cm_interp_dac = inputs['cm_interp']
        dac_control = np.zeros((self.n_span, self.n_Re, self.n_tab))
        Re_loc = np.zeros((self.n_span, self.n_Re, self.n_tab))
        Ma_loc = np.zeros((self.n_span, self.n_Re, self.n_tab))

        # Get polars for DAC parameter values
        if self.n_dac > 0:
            if 'cl_interp_dac' not in self.saved_polar_data.keys():
                
                run_xfoil_params = {}
                # Self
                run_xfoil_params['xfoil_path'] = self.xfoil_path
                run_xfoil_params['cores'] = self.cores
                run_xfoil_params['n_span'] = self.n_span
                run_xfoil_params['n_Re'] = self.n_Re
                run_xfoil_params['n_tab'] = self.n_tab
                run_xfoil_params['dac_profiles'] = copy.copy(self.dac_profiles)
                run_xfoil_params['R'] = self.R
                run_xfoil_params['tsr'] = self.tsr
                run_xfoil_params['maxTS'] = self.maxTS
                run_xfoil_params['KinVisc'] = self.KinVisc
                run_xfoil_params['SpdSound'] = self.SpdSound
                run_xfoil_params['dac_model'] = self.dac_model
                # inputs
                run_xfoil_params['cl_interp'] = inputs['cl_interp']
                run_xfoil_params['cd_interp'] = inputs['cd_interp']
                run_xfoil_params['cm_interp'] = inputs['cm_interp']
                run_xfoil_params['chord'] = inputs['chord']
                run_xfoil_params['s'] = inputs['s']
                run_xfoil_params['r'] = inputs['r']
                run_xfoil_params['aoa'] = inputs['aoa']


                # Run XFoil as multiple processors with MPI
                if MPI and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
                    run_xfoil_params['run_MPI'] = True
                    # mpi comm management
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    sub_ranks = self.mpi_comm_map_down[rank]
                    size = len(sub_ranks)
                    
                    print('Parallelizing Xfoil on {} subranks.'.format(len(sub_ranks)))
                    N_cases = self.n_span # total number of airfoil sections
                    N_loops = int(np.ceil(float(N_cases)/float(size)))  # number of times function calls need to "loop"

                    # iterate loops, populate polar tables
                    for i in range(N_loops):
                        idx_s = i*size
                        idx_e = min((i+1)*size, N_cases)

                        for idx, afi in enumerate(np.arange(idx_s,idx_e)):
                            data = [partial(get_dac_polars, run_xfoil_params), afi]
                            rank_j = sub_ranks[idx]
                            comm.send(data, dest=rank_j, tag=0)

                        # for rank_j in sub_ranks:
                        for idx, afi in enumerate(np.arange(idx_s, idx_e)):
                            rank_j = sub_ranks[idx]
                            polars_separate_af = comm.recv(source=rank_j, tag=1)
                            cl_interp_dac[afi,:,:,:] = polars_separate_af[0]
                            cd_interp_dac[afi,:,:,:] = polars_separate_af[1]
                            cm_interp_dac[afi,:,:,:] = polars_separate_af[2]
                            dac_control[afi,:,:] = polars_separate_af[3]
                            Re_loc[afi,:,:] = polars_separate_af[4]
                            Ma_loc[afi,:,:] = polars_separate_af[5]
                    
                    # for afi in range(self.n_span):
                    #     # re-structure outputs
                        
                # Multiple processors, but not MPI
                elif self.cores > 1 and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
                    run_xfoil_params['run_multi'] = True

                    # separate airfoil sections w/ and w/o DAC devices
                    af_with_dac = []
                    af_without_dac = []
                    for afi in range(len(run_xfoil_params['dac_profiles'])):
                        if 'coords' in run_xfoil_params['dac_profiles'][afi]:
                            af_with_dac.append(afi)
                        else:
                            af_without_dac.append(afi)

                    print('Parallelizing Xfoil on {} cores'.format(self.cores))
                    pool = mp.Pool(self.cores)
                    polars_separate_dac = pool.map(
                        partial(get_dac_polars, run_xfoil_params), af_with_dac)
                    # parallelize dac-specific calls for better efficiency
                    polars_separate_nodac = pool.map(
                        partial(get_dac_polars, run_xfoil_params), af_without_dac)
                    pool.close()
                    pool.join()

                    for i, afi in enumerate(af_with_dac):
                        cl_interp_dac[afi,:,:,:] = polars_separate_dac[i][0]
                        cd_interp_dac[afi,:,:,:] = polars_separate_dac[i][1]
                        cm_interp_dac[afi,:,:,:] = polars_separate_dac[i][2]
                        dac_control[afi,:,:] = polars_separate_dac[i][3]
                        Re_loc[afi,:,:] = polars_separate_dac[i][4]
                        Ma_loc[afi,:,:] = polars_separate_dac[i][5]

                    for i, afi in enumerate(af_without_dac):
                        cl_interp_dac[afi,:,:,:] = polars_separate_nodac[i][0]
                        cd_interp_dac[afi,:,:,:] = polars_separate_nodac[i][1]
                        cm_interp_dac[afi,:,:,:] = polars_separate_nodac[i][2]
                        dac_control[afi,:,:] = polars_separate_nodac[i][3]
                        Re_loc[afi,:,:] = polars_separate_nodac[i][4]
                        Ma_loc[afi,:,:] = polars_separate_nodac[i][5]
                                            
                else:
                    for afi in range(self.n_span): # iterate number of radial stations for various airfoil tables
                        cl_interp_dac_af, cd_interp_dac_af, cm_interp_dac_af, dac_control_af, Re_loc_af, Ma_loc_af = get_dac_polars(run_xfoil_params, afi)

                        cl_interp_dac[afi,:,:,:] = cl_interp_dac_af
                        cd_interp_dac[afi,:,:,:] = cd_interp_dac_af
                        cm_interp_dac[afi,:,:,:] = cm_interp_dac_af
                        dac_control[afi,:,:] = dac_control_af
                        Re_loc[afi,:,:] = Re_loc_af
                        Ma_loc[afi,:,:] = Ma_loc_af

                if not any([self.options['opt_options']['design_variables']['control']['dac']['dac_ext']['flag'],
                            self.options['opt_options']['design_variables']['control']['dac']['dac_end']['flag']]):
                    self.saved_polar_data['cl_interp_dac'] = copy.copy(cl_interp_dac)
                    self.saved_polar_data['cd_interp_dac'] = copy.copy(cd_interp_dac)
                    self.saved_polar_data['cm_interp_dac'] = copy.copy(cm_interp_dac)
                    self.saved_polar_data['dac_control'] = copy.copy(dac_control)
                    self.saved_polar_data['Re_loc'] = copy.copy(Re_loc)
                    self.saved_polar_data['Ma_loc'] = copy.copy(Ma_loc)
                    
            else:
                # load xfoil data from previous runs
                print('Skipping XFOIL and loading blade polar data from previous iteration.')
                cl_interp_dac = self.saved_polar_data['cl_interp_dac']
                cd_interp_dac = self.saved_polar_data['cd_interp_dac']
                cm_interp_dac = self.saved_polar_data['cm_interp_dac']
                dac_control = self.saved_polar_data['dac_control']  
                Re_loc = self.saved_polar_data['Re_loc']       
                Ma_loc = self.saved_polar_data['Ma_loc']       



                    # else:  # no flap at specific radial location (but in general 'aerodynamic_control' is defined in blade from yaml)
                    #     # for j in range(n_Re): # ToDo incorporade variable Re capability
                    #     for ind in range(self.n_tab):  # fill all self.n_tab slots even though no flaps exist at current radial position
                    #         c = inputs['chord'][afi]  # blade chord length at cross section
                    #         rR = inputs['r'][afi] / inputs['r'][-1]  # non-dimensional blade radial station at cross section
                    #         Re_loc[afi, :, ind] = c * maxTS * rR / KinVisc
                    #         Ma_loc[afi, :, ind] = maxTS * rR / SpdSound
                    #         for j in range(self.n_Re):
                    #             cl_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]
                    #             cd_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]
                    #             cm_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]

        else:
            for afi in range(self.n_span):
                # for j in range(n_Re):  # ToDo incorporade variable Re capability
                for ind in range(self.n_tab):  # fill all self.n_tab slots even though no flaps exist at current radial position
                    c = inputs['chord'][afi]  # blade chord length at cross section
                    rR = inputs['r'][afi] / inputs['r'][-1]  # non-dimensional blade radial station at cross section
                    Re_loc[afi, :, ind] = c * self.maxTS * rR / self.KinVisc
                    Ma_loc[afi, :, ind] = self.maxTS * rR / self.SpdSound
                    
        outputs['cl_interp_dac']  = cl_interp_dac
        outputs['cd_interp_dac']  = cd_interp_dac
        outputs['cm_interp_dac']  = cm_interp_dac
        outputs['dac_param']      = dac_control # use vector of dac parameter controls
        outputs['Re_loc'] = Re_loc
        outputs['Ma_loc'] = Ma_loc

def get_dac_polars(run_xfoil_params, afi):
    '''
    Sort of a wrapper script for runXfoil - makes parallelization possible

    Parameters:
    -----------
    run_xfoil_params: dict
        contains all necessary information to succesfully run xFoil
    afi: int
        airfoil section index

    Returns:
    --------
    cl_interp_dac_af: 3D array
        lift coefficient tables
    cd_interp_dac_af: 3D array
        drag coefficient  tables
    cm_interp_dac_af: 3D array
        moment coefficient tables
    dac_control_af: 2D array
       dac angle tables
    Re_loc_af: 2D array
        Reynolds number table
    Ma_loc_af: 2D array
        Mach number table
    '''
    cl_interp_dac_af  = copy.deepcopy(run_xfoil_params['cl_interp'][afi])
    cd_interp_dac_af  = copy.deepcopy(run_xfoil_params['cd_interp'][afi])
    cm_interp_dac_af  = copy.deepcopy(run_xfoil_params['cm_interp'][afi])
    dac_control_af       = copy.deepcopy(np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab'])))
    Re_loc_af           = copy.deepcopy(np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab'])))
    Ma_loc_af           = copy.deepcopy(np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab'])))
    n_tab               = copy.deepcopy(run_xfoil_params['n_tab'])
    dac_profiles       = copy.deepcopy(run_xfoil_params['dac_profiles'])
    chord               = copy.deepcopy(run_xfoil_params['chord'])
    span                = copy.deepcopy(run_xfoil_params['s'])
    rad_loc             = copy.deepcopy(run_xfoil_params['r'])
    R                   = copy.deepcopy(run_xfoil_params['R'])
    KinVisc             = copy.deepcopy(run_xfoil_params['KinVisc'])
    maxTS               = copy.deepcopy(run_xfoil_params['maxTS'])
    SpdSound            = copy.deepcopy(run_xfoil_params['SpdSound'])
    xfoil_path          = copy.deepcopy(run_xfoil_params['xfoil_path'])
    aoa                 = copy.deepcopy(run_xfoil_params['aoa'])
    dac_model           = copy.deepcopy(run_xfoil_params['dac_model'])

    if dac_model > 0 and 'dac_param' in dac_profiles[afi]: # check if airfoil polars need to be created using either XFOIL or general_dac_model
    #if 'coords' in dac_profiles[afi]: # check if 'coords' is an element of 'dac_profiles', i.e. if we have various DAC parameter values
        # for j in range(n_Re): # ToDo incorporade variable Re capability
        for ind in range(n_tab):
            #fa = flap_profiles[afi]['dac_param'][ind] # value of respective dac parameter value
            dac_control_af[:,ind] =dac_profiles[afi]['dac_param'][ind] # dac parameter value vector of distributed aerodynamics control
            # eta = (blade['pf']['r'][afi]/blade['pf']['r'][-1])
            # eta = blade['outer_shape_bem']['chord']['grid'][afi]
            c   = chord[afi]  # blade chord length at cross section
            s   = span[afi]
            cr  = chord[afi] / rad_loc[afi]
            rR  = rad_loc[afi] / rad_loc[-1]  # non-dimensional blade radial station at cross section in the rotor coordinate system
            Re_loc_af[:,ind] = c* maxTS * rR / KinVisc
            Ma_loc_af[:,ind] = maxTS * rR / SpdSound
            
            if dac_model == 1:
                print('Run xfoil for nondimensional blade span section s = ' + str(s) + ' with ' + str(dac_control_af[0,ind]*180./np.pi) + ' deg flap deflection angle; Re equal to ' + str(Re_loc_af[0,ind]) + '; Ma equal to ' + str(Ma_loc_af[0,ind]))

                xfoil_kw = {'AoA_min': -20,
                            'AoA_max': 25,
                            'AoA_inc': 0.25,
                            'Ma':  Ma_loc_af[0, ind],
                            }

                data = runXfoil(xfoil_path, dac_profiles[afi]['coords'][:, 0, ind],dac_profiles[afi]['coords'][:, 1, ind],Re_loc_af[0, ind], **xfoil_kw)
            elif dac_model == 2:
                print('Run General DAC Model for nondimensional blade span section s = ' + str(round(s,6)) + ' with ' + str(dac_control_af[0,ind]) + ' dac control parameter; Re equal to ' + str(round(Re_loc_af[0,ind],1)) + '; Ma equal to ' + str(round(Ma_loc_af[0,ind],5)))

                clmax_ratio,stall_shift,LD_ratio,alpha0_shift,S_ratio,CD0_shift = LE_Spoiler(dac_control_af[0,ind]) #bem: TODO I need to add in the ability to use the low fidelity TE flap as an option or build in the ability to use different low fidelity models
                data = general_dac_mod(aoa,cl_interp_dac_af[:,0,ind],cd_interp_dac_af[:,0,ind],cm_interp_dac_af[:,0,ind],clmax_ratio,stall_shift,LD_ratio,alpha0_shift,S_ratio,CD0_shift)
                # print("Lift data: ", data[:,1])
            else:
                print('No DAC_Model chosen') #TODO bem: Need to add error handling here
            
            oldpolar= Polar(Re_loc_af[0,ind], data[:,0],data[:,1],data[:,2],data[:,4]) # data[:,0] is alpha, data[:,1] is Cl, data[:,2] is Cd, data[:,4] is Cm
            try:
                polar3d = oldpolar.correction3D(rR,cr,run_xfoil_params['tsr']) # Apply 3D corrections (made sure to change the r/R, c/r, and tsr values appropriately when calling AFcorrections())
            except IndexError:
                for key in run_xfoil_params:
                    print('{} = {}'.format(key, run_xfoil_params[key]))
                print('XFOIL DATA: {}'.format(data))
                raise

            cdmax   = np.max(data[:,2]) # Keep the same max Cd as before
            polar   = polar3d.extrapolate(cdmax) # Extrapolate polars for alpha between -180 deg and 180 deg

            for j in range(run_xfoil_params['n_Re']):
                cl_interp_dac_af[:,j,ind] = np.interp(np.degrees(aoa), polar.alpha, polar.cl)
                cd_interp_dac_af[:,j,ind] = np.interp(np.degrees(aoa), polar.alpha, polar.cd)
                cm_interp_dac_af[:,j,ind] = np.interp(np.degrees(aoa), polar.alpha, polar.cm)

        # # ** The code below will plot the three cl polars
            # import matplotlib.pyplot as plt
            # font = {'family': 'Times New Roman',
            #         'weight': 'normal',
            #         'size': 18}
            # plt.rc('font', **font)
            # plt.figure
            # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,0],'r', label='$\\delta_{flap}$ = -10 deg')  # -10
            # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,1],'k', label='$\\delta_{flap}$ = 0 deg')  # 0
            # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,2],'b', label='$\\delta_{flap}$ = +10 deg')  # +10
            # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,0],'r')  # -10
            # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,1],'k')  # 0
            # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,2],'b')  # +10
            # plt.xlim(xmin=-15, xmax=15)
            # plt.ylim(ymin=-1.7, ymax=2.2)
            # plt.grid(True)
            # # autoscale_y(ax)
            # plt.xlabel('Angles of attack, deg')
            # plt.ylabel('Lift coefficient')
            # plt.legend(loc='lower right')
            # plt.tight_layout()
            # plt.show()
            # # # # plt.savefig('airfoil_polars_check/r_R_1_0_cl_flaps.png', dpi=300)
            # # # # plt.savefig('airfoil_polars_check/NACA63-618_cl_flaps.png', dpi=300)
            # # # # plt.savefig('airfoil_polars_check/FFA-W3-211_cl_flaps.png', dpi=300)
            # # # # plt.savefig('airfoil_polars_check/FFA-W3-241_cl_flaps.png', dpi=300)
            # # # # plt.savefig('airfoil_polars_check/FFA-W3-301_cl_flaps.png', dpi=300)



    else:  # no dac device at specific radial location (but in general 'aerodynamic_control' is defined in blade from yaml)
        for ind in range(n_tab):  # fill all run_xfoil_params['n_tab'] slots even though no device exist at current radial position
            c = chord[afi]  # blade chord length at cross section
            rR = rad_loc[afi] / rad_loc[-1]  # non-dimensional blade radial station at cross section
            Re_loc_af[:, ind] = c * maxTS * rR / KinVisc
            Ma_loc_af[:, ind] = maxTS * rR / SpdSound            

            for j in range(run_xfoil_params['n_Re']):
                cl_interp_dac_af[:, j, ind] = copy.deepcopy(cl_interp_dac_af[:, j, 0])
                cd_interp_dac_af[:, j, ind] = copy.deepcopy(cd_interp_dac_af[:, j, 0])
                cm_interp_dac_af[:, j, ind] = copy.deepcopy(cm_interp_dac_af[:, j, 0])
    
    return cl_interp_dac_af, cd_interp_dac_af, cm_interp_dac_af, dac_control_af, Re_loc_af, Ma_loc_af

def general_dac_mod(alpha,cl,cd,cm,clmax_ratio,stall_shift,LD_ratio,alpha0_shift,S_ratio,CD0_shift):
    '''
    Models changes to lift, drag, and moment polars to mimic the effects of an active aerodynamic flow device.

    Parameters:
    -----------
    alpha: 1D array
        Values of angle of attack (AoA) for unmodified polars
    cl: 1D array
        Values of lift coefficient for unmodified polars
    cd: 1D array
        Values of drag coefficient for unmodified polars
    cm: 1D array
        Values of moment coefficient for unmodified polars
    clmax_ratio: float
        Ratio of maximum lift coefficients (near stall) for modified over unmodified value
    stall_shift: float
        A shift in the value of the AoA location where maximum cl occurs in degrees (a positive value shifts the modified stall point to the left, stall occurs at a lower AoA)
    LD_ratio: float
        Ratio of maximum lift-to-drag ratios between modified and unmodified polars (value greater than 1 means the lift-to-drag ratio of the modified polar will be greater than the unmodified polar)
    alpha0_shift: float
        A shift in the value of zero lift AoA in degrees (a positive value will shift the zero lift angle of attack to the left)
    S_ratio: float
        ratio of lift curve slopes between modified and unmodified [1/deg] (a value less than ones means modified lift curve slope is less than unmodified)
    CD0_shift: float
        A shift in the drag coefficient at zero-lift AoA (a positive value will increase the drag coefficient at zero lift AoA)

    Returns:
    --------
    mod_polar: 2D array
        Column 1: AoA in deg, Column 2: Lift Coefficient, Column 3: Drag Coefficient, Column 4: Moment Coefficient
    '''
   
    # Initializing variables from unmodified polars
    n = 100                                                     # Number of angle of attacks in the modified polars
    a_min_ind, a_max_ind = find_prestall_range(alpha,cl)        # finds starting and ending indicies for unstalled arifoil conditions
    A0 = find_alpha0(alpha,cl,a_min_ind,a_max_ind)              # finds zero lift angle of attack
    S1 = find_liftcurve_slope(alpha,cl,A0)                      # finds lift curve slope

    ACLmax_ind = a_max_ind                                      # Index where max lift occurs
    ACLmax = alpha[ACLmax_ind]                                  # AoA where maximum lift coefficient occurs
    Clmax = cl[ACLmax_ind]                                      # Maximum lift coefficient
    Cdmax = cd[ACLmax_ind]                                      # Maximum drag coefficient
    Cmmax = cm[ACLmax_ind]                                      # Moment coefficient at max lift                                    
    CD0 = find_C0(alpha,cd,A0)                                  # Drag coefficient at zero-lift angle of attack
    CM0 = find_C0(alpha,cm,A0)                                  # Moment coefficient at zero-lift angle of attack
    amax = (Cmmax-CM0)/Clmax                                    # shifted moment arm approximation for moment coefficient at maximum lift point

    # Modifying variables according to input parameters
    A0 -= alpha0_shift*np.pi/180.0
    S1 *= S_ratio
    CD0 += CD0_shift
    Clmax *= clmax_ratio
    ACLmax -= (stall_shift + alpha0_shift)*np.pi/180.0
    LD = Clmax/Cdmax                                            # calculate max L/D
    LD *= LD_ratio                          
    Cdmax_2 = Clmax/LD

    CD2max = 1.5                                                # assumed value for max Cd post stall (blunt body drag coefficient)
    M = 2.0                                                     # power of of drag coefficient in linear range (assumed quadratic relationship)
    ext = 3.0                                                   # Amount to extent beyond possitive/negative stall
    da = ((ACLmax + ext*np.pi/180.0)-(-(ACLmax-A0) - ext*np.pi/180.0))/(float(n)-1.0)   # Calculate AoA step-size

    # Initializing output matricies
    mod_polar = np.zeros((n,5))

    # Calculating modified lift, drag, and moment coefficients
    RCL = S1*(ACLmax - A0) - Clmax
    # N1 = 1000.
    # while RCL <= 0.0 and N1 > 10:
    # print("s1 = ", S1)
    while RCL <= 0.0:
        S1 *= 1.05
        RCL = S1*(ACLmax - A0) - Clmax
        # print("s1 = ", S1)
        # N1 = 1. + Clmax/RCL
    N1 = 1. + Clmax/RCL

    for j in range(n):
        # Calculate Angle of Attack (in radians)
        mod_polar[j,0] = -(ACLmax-A0) - ext*np.pi/180.0 + j*da      # Note the ext variable extends the range of angle of attacks being used by that much beyond the stall angle

        # Calculate Lift Coefficient
        if mod_polar[j,0] < A0:
            mod_polar[j,1] = S1*(mod_polar[j,0]-A0) + RCL*((A0-mod_polar[j,0])/(ACLmax-A0))**N1
        else:
            mod_polar[j,1] = S1*(mod_polar[j,0]-A0) - RCL*((mod_polar[j,0]-A0)/(ACLmax-A0))**N1     # Assumed to be symmetric about A0
        # print("cl: ", mod_polar[j,1], "N1: ", N1)

        # Calculate Drag Coefficient    
        if mod_polar[j,0] < (2*A0-ACLmax):
            mod_polar[j,2] = Cdmax_2 + (CD2max - Cdmax_2)*np.sin(((2*A0-mod_polar[j,0])-ACLmax)/(np.pi/2.0-ACLmax)*np.pi/2.0)
        elif mod_polar[j,0] >= (2*A0-ACLmax)  and mod_polar[j,0] <= ACLmax:
            mod_polar[j,2] = CD0 + (Cdmax_2-CD0)*((mod_polar[j,0]-A0)/(ACLmax-A0))**M       # In linear range, drag is quadratic
        else:
            mod_polar[j,2] = Cdmax_2 + (CD2max - Cdmax_2)*np.sin((mod_polar[j,0]-ACLmax)/(np.pi/2.0-ACLmax)*np.pi/2.0) 
        
        # Calculate Moment Coefficient
        a_arm = amax*(mod_polar[j,0]-A0)/(ACLmax-A0)
        mod_polar[j,4] = a_arm*mod_polar[j,1] + CM0 - alpha0_shift*np.pi/180.0*S1/8.  # Note that moment coefficient is intentionally placed in index 4 column (skipped a column) to maintain consistency with xfoil output format

    mod_polar[:,0] = np.degrees(mod_polar[:,0])             # Convert output AoA to degrees for consistency with xfoil output

    return mod_polar

def find_prestall_range(alpha,cl):
    '''
    Searches lift polar to find indicies where postive and negative stall occur (ignores deep stall and extrapolated data beyond natural stall points of most airfoils)

    Parameters:
    -----------
    alpha: 1D array
        Values of angle of attack (AoA) for polar
    cl: 1D array
        Values of lift coefficient for polars

    Returns:
    --------
    a_min_ind: int
        Index where minimum lift occurs (near negative stall point)
    a_max_ind: int
        Index where maximum lift occurs (near positive stall point)
    '''
    for ind in range(len(alpha)):
        if alpha[ind]>=-25.0* np.pi/180.0 and alpha[ind]<=25.*np.pi/180.0:          # Reasonable range of AoA for pre-stall range for most airfoils
            if ind != 0 and ind != len(alpha)-1:                                    # Avoiding end points (need to look at slopes to find maxima/minima)
                if (cl[ind-1]-cl[ind])*(cl[ind]-cl[ind+1])<0. and cl[ind]<=0.:      # Finding minimum lift for negative stall
                    a_min_ind = ind
                    #print('1')
                elif (cl[ind-1]-cl[ind])*(cl[ind]-cl[ind+1])<0. and cl[ind]>0.:     # Finding maximum lift for positive stall
                    a_max_ind = ind
                    #print('2')
                    
            elif ind == 0:                                                          # Dealing with possibility of negative stall not captured
                if cl[ind] <= cl[ind+1] and a_min_ind == []:
                    a_min_ind = ind
                    #print('3')
            else:                                                                   # Dealing with possibility where positive stall is not captured
                if cl[ind-1] <= cl[ind] and a_max_ind == []:
                    a_max_ind = ind
                    #print('4')
    
    return a_min_ind, a_max_ind

def find_alpha0(alpha,cl,a_min_ind,a_max_ind):
    '''
    Searches lift polar over specified range of angle of attack (pre-stall range) to find value of zero-lift angle of attack through interpolation of lift polar

    Parameters:
    -----------
    alpha: 1D array
        Values of angle of attack (AoA) for polar
    cl: 1D array
        Values of lift coefficient for polars
    a_min_ind: int
        Index for lower bound of search
    a_max_ind: int
        Index for upper bound of search

    Returns:
    --------
    alpha0: float
        Value of zero-lift angle of attack
    '''
    for ind in range(len(alpha)-1):
        if ind >= a_min_ind and ind <=a_max_ind:
            if cl[ind]<=0. and cl[ind+1]>0.:                                                    # Finding where lift transitions form negative to positive
                alpha0 = (-alpha[ind+1]+alpha[ind])/(cl[ind+1]-cl[ind])*(cl[ind])+alpha[ind]    # Linear interpolation between two points
                break

    return alpha0

def find_liftcurve_slope(alpha,cl,alpha0 = 0.):
    '''
    Calculates lift curve slope in linear range of a given lift polar

    Parameters:
    -----------
    alpha: 1D array
        Values of angle of attack (AoA) for polar
    cl: 1D array
        Values of lift coefficient for polars
    alpha0: float
        Value of zero lift angle of attack (defaults to zero if none given)

    Returns:
    --------
    slope: float
        lift curve slope
    '''
    for ind in range(len(alpha)):
        if alpha[ind] <= alpha0-2.0*np.pi/180.0 and alpha[ind+1] >= alpha0-2.0*np.pi/180.0:       # Lower bound for slop approximately 3 deg below zero-lift AoA
            a_min = alpha[ind]
            cl_min = cl[ind]
        elif alpha[ind] <= alpha0+5.0*np.pi/180.0 and alpha[ind+1] >= alpha0+5.0*np.pi/180.0:     # Upper bound for slop approximately 3 deg above zero-lift AoA
            a_max = alpha[ind]
            cl_max = cl[ind]
            break

    slope = (cl_max-cl_min)/(a_max-a_min)

    return slope

def find_C0(alpha,c,alpha0 = 0.):
    '''
    Calculates interpolated value of a coeficient (lift, drag, moment, etc.) about a certain angle of attack

    Parameters:
    -----------
    alpha: 1D array
        Values of angle of attack (AoA) for polar
    c: 1D array
        Values of coefficient for polars
    alpha0: float
        Value of angle of attack being interpolated about (defaults to zero if none given)

    Returns:
    --------
    C0: float
        Interpolated value of coefficient
    '''
    for ind in range(len(alpha)):
        if alpha[ind] <= alpha0 and alpha[ind+1] >= alpha0:                                 # Finds location of segment to be interpolated between
            C0 = (c[ind+1]-c[ind])/(alpha[ind+1]-alpha[ind])*(alpha0-alpha[ind])+c[ind]     # Linear interpolation
            break

    return C0

def LE_Spoiler(h):
    '''
    A low fidelity model relating LE spoiler height to the polar modification parameters used in general_dac_mod function

    Parameters:
    -----------
    h: float
        Leading edge spoiler height    

    Returns:
    --------
    clmax_ratio: float
        Ratio of maximum lift coefficients (near stall) for modified over unmodified value
    stall_shift: float
        A shift in the value of the AoA location where maximum cl occurs in degrees (a positive value shifts the modified stall point to the left, stall occurs at a lower AoA)
    LD_ratio: float
        Ratio of maximum lift-to-drag ratios between modified and unmodified polars (value greater than 1 means the lift-to-drag ratio of the modified polar will be greater than the unmodified polar)
    alpha0_shift: float
        A shift in the value of zero lift AoA in degrees (a positive value will shift the zero lift angle of attack to the left)
    S_ratio: float
        ratio of lift curve slopes between modified and unmodified [1/deg] (a value less than ones means modified lift curve slope is less than unmodified)
    CD0_shift: float
        A shift in the drag coefficient at zero-lift AoA (a positive value will increase the drag coefficient at zero lift AoA)
    '''
    
    clmax_ratio = 0.01 * h**2 - 0.1601*h +1.0
    stall_shift = -0.117*h**2 + 2.0329*h
    LD_ratio = 0.0127*h**2 - 0.1993*h + 1.0
    if h <= 1.0:
        alpha0_shift = 0.0
    else:
        alpha0_shift = -1.0
                
    if h <= 4.0:
        S_ratio = 1.0
    else:
        S_ratio = 0.9
                
    CD0_shift = 0.002*h

    return clmax_ratio, stall_shift, LD_ratio, alpha0_shift, S_ratio, CD0_shift

def TE_flap_gen_mod(delt): #This simple model is only for flaps that are 20% of chord. To be more useful, this model should really be a surface map for flap deflection angle and size.
    '''
    A low fidelity model relating TE flap deflection angle to the polar modification parameters used in general_dac_mod function

    Parameters:
    -----------
    delt: float
        Trailing edge flap angle (deg)  

    Returns:
    --------
    clmax_ratio: float
        Ratio of maximum lift coefficients (near stall) for modified over unmodified value
    stall_shift: float
        A shift in the value of the AoA location where maximum cl occurs in degrees (a positive value shifts the modified stall point to the left, stall occurs at a lower AoA)
    LD_ratio: float
        Ratio of maximum lift-to-drag ratios between modified and unmodified polars (value greater than 1 means the lift-to-drag ratio of the modified polar will be greater than the unmodified polar)
    alpha0_shift: float
        A shift in the value of zero lift AoA in degrees (a positive value will shift the zero lift angle of attack to the left)
    S_ratio: float
        ratio of lift curve slopes between modified and unmodified [1/deg] (a value less than ones means modified lift curve slope is less than unmodified)
    CD0_shift: float
        A shift in the drag coefficient at zero-lift AoA (a positive value will increase the drag coefficient at zero lift AoA)
    '''
    
    clmax_ratio = 0.0135*delt + 0.9811
    stall_shift = -0.3*delt
    LD_ratio = 0.0017*delt**2 - 0.0025*delt + 1.0
    alpha0_shift = 0.5*delt
                
    S_ratio = 1.0
                
    CD0_shift = 0.0

    return clmax_ratio, stall_shift, LD_ratio, alpha0_shift, S_ratio, CD0_shift