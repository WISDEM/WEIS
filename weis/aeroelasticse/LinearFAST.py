'''
Class and function for generating linear models from OpenFAST

1. Run steady state simulations
2. Process sims to find operating point (TODO: determine how important this is and enable recieving this info from elsewhere)
3. Run OpenFAST in linear mode

examples/control_opt/run_lin_turbine.py will run outputs from gen_linear_model()

'''
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml

# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
from ROSCO_toolbox import utilities as ROSCO_utilites
fast_io = ROSCO_utilites.FAST_IO()
fast_pl = ROSCO_utilites.FAST_Plots()

# WISDEM modules
from weis.aeroelasticse.Util import FileTools

# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis


import numpy as np
import sys, os, platform, glob

class LinearFAST(runFAST_pywrapper_batch):
    ''' 
        Class for 
        1. Running steady state simulations for operating points
            - this functionality is in the process of being added to OpenFAST, will keep for now
            - I think it's important to include because if DOFs are not enabled in linearization sims, the displacement is held at these values
        2. Processing steady state simulation information
        3. Running openfast linearization cases to generate linear models across wind speeds
    '''

    def __init__(self, **kwargs):

        self.FAST_ver           = 'OpenFAST'
        # self.FAST_exe           = None
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.debug_level        = 0
        self.dev_branch         = True

        self.read_yaml          = False
        self.FAST_yamlfile_in   = ''
        self.fst_vt             = {}
        self.write_yaml         = False
        self.FAST_yamlfile_out  = ''

        self.case_list          = []
        self.case_name_list     = []
        self.channels           = {}

        self.post               = None

        # Linear specific default attributes
        # linearization setup
        self.v_rated            = 11         # needed as input from RotorSE or something, to determine TrimCase for linearization
        self.GBRatio            = 1
        self.WindSpeeds         = [15]
        self.DOFs               = ['GenDOF','TwFADOF1','PtfmPDOF']
        self.TMax               = 2000.
        self.NLinTimes          = 12

        #if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
        self.HydroStates        = True  

        # simulation setup
        self.parallel           = False
        self.cores              = 4

        # overwrite steady & linearizations
        self.overwrite          = False

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(LinearFAST, self).__init__()

    def runFAST_steady(self):
        """ 
        Run batch of steady state cases for initial conditions, in serial or in parallel
        TODO: determine whether we can skip this step
        """

        self.FAST_runDirectory = self.FAST_steadyDirectory

        case_inputs = {}
        case_inputs[("Fst","TMax")] = {'vals':[self.TMax], 'group':0}
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        case_inputs[("Fst","OutFileFmt")] = {'vals':[2], 'group':0}

        # Wind Speeds
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':self.WindSpeeds, 'group':1}

        if platform.system() == 'Windows':
            path2dll = os.path.join(self.weis_dir, 'local/lib/libdiscon.dll')
        elif platform.system() == 'Darwin':
            path2dll = os.path.join(self.weis_dir, 'local/lib/libdiscon.dylib')
        else:
            path2dll = os.path.join(self.weis_dir, 'local/lib/libdiscon.so')

        case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}

        channels = {}
        for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", "TipRDxr", "TipRDyr", "TipRDzr"]:
            channels[var] = True

        self.channels = channels

        # # Initial Conditions: less important, trying to find them here
        # case_inputs[("ElastoDyn","RotSpeed")] = {'vals':[7.55], 'group':0}
        # case_inputs[("ElastoDyn","BlPitch1")] = {'vals':[3.823], 'group':0}
        # case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
        # case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
        
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=self.FAST_steadyDirectory, namebase='steady')

        self.case_list = case_list
        self.case_name_list = case_name_list

        outfiles = glob.glob(os.path.join(self.FAST_steadyDirectory,'steady*.outb'))

        if self.overwrite or (len(outfiles) != len(self.WindSpeeds)): # if the steady output files are all there
            if self.parallel:
                self.run_multi(self.cores)
            else:
                self.run_serial()

        
    def postFAST_steady(self):
        """
        Post process results to get steady state information for all initial conditions at each wind speed
        Save as ss_ops.yaml for 
        """

        # Plot steady states vs wind speed
        PLOT = 0

        # Define input files paths
        output_dir      = self.FAST_steadyDirectory

        # Find all outfiles
        outfiles = []
        for file in os.listdir(output_dir):
            if file.endswith('.outb'):
                outfiles.append(os.path.join(output_dir,file))
            # elif file.endswith('.out') and not file.endswith('.MD.out'):  
            #     outfiles.append(os.path.join(output_dir,file))


        # Initialize processing classes
        fp = Processing.FAST_Processing()

        # Set some processing parameters
        fp.OpenFAST_outfile_list        = outfiles
        fp.t0                           = self.TMax - 400            # make sure this is less than simulation time
        fp.parallel_analysis            = self.parallel
        fp.parallel_cores               = self.cores
        fp.results_dir                  = os.path.join(output_dir, 'stats')
        fp.verbose                      = True
        fp.save_LoadRanking             = True
        fp.save_SummaryStats            = True

        # Load and save statistics and load rankings
        if self.overwrite or not os.path.exists(os.path.join(output_dir,'ss_ops.yaml')):
            stats, _ =fp.batch_processing()

            if hasattr(stats,'__len__'):
                stats = stats[0]

            windSortInd = np.argsort(stats['Wind1VelX']['mean'])

            #            FAST output name,  FAST IC name
            ssChannels = [['Wind1VelX',     'Wind1VelX'],  
                        ['OoPDefl1',        'OoPDefl'],
                        ['IPDefl1',         'IPDefl'],
                        ['BldPitch1',       'BlPitch1'],
                        ['RotSpeed',        'RotSpeed'],
                        ['TTDspFA',         'TTDspFA'],
                        ['TTDspSS',         'TTDspSS'],
                        ['PtfmSurge',       'PtfmSurge'],
                        ['PtfmSway',        'PtfmSway'],
                        ['PtfmHeave',       'PtfmHeave'],
                        ['PtfmRoll',        'PtfmRoll'],
                        ['PtfmYaw',         'PtfmYaw'],
                        ['PtfmPitch',       'PtfmPitch'],
                        ]

            ssChanData = {}
            for iChan in ssChannels:
                try:
                    ssChanData[iChan[1]] = np.array(stats[iChan[0]]['mean'])[windSortInd].tolist()
                except:
                    print('Warning: ' + iChan[0] + ' is is not in OutList')


            if PLOT:
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(211)
                ax2 = fig1.add_subplot(212)

                ax1.plot(ssChanData['Wind1VelX'],ssChanData['BlPitch1'])
                ax2.plot(ssChanData['Wind1VelX'],ssChanData['RotSpeed'])


                fig2 = plt.figure()
                ax1 = fig2.add_subplot(411)
                ax2 = fig2.add_subplot(412)
                ax3 = fig2.add_subplot(413)
                ax4 = fig2.add_subplot(414)

                ax1.plot(ssChanData['Wind1VelX'],ssChanData['OoPDefl'])
                ax2.plot(ssChanData['Wind1VelX'],ssChanData['IPDefl'])
                ax3.plot(ssChanData['Wind1VelX'],ssChanData['TTDspFA'])
                ax4.plot(ssChanData['Wind1VelX'],ssChanData['TTDspSS'])

                fig3 = plt.figure()
                ax1 = fig3.add_subplot(611)
                ax2 = fig3.add_subplot(612)
                ax3 = fig3.add_subplot(613)
                ax4 = fig3.add_subplot(614)
                ax5 = fig3.add_subplot(615)
                ax6 = fig3.add_subplot(616)

                ax1.plot(ssChanData['Wind1VelX'],ssChanData['PtfmSurge'])
                ax2.plot(ssChanData['Wind1VelX'],ssChanData['PtfmSway'])
                ax3.plot(ssChanData['Wind1VelX'],ssChanData['PtfmHeave'])
                ax4.plot(ssChanData['Wind1VelX'],ssChanData['PtfmRoll'])
                ax5.plot(ssChanData['Wind1VelX'],ssChanData['PtfmPitch'])
                ax6.plot(ssChanData['Wind1VelX'],ssChanData['PtfmYaw'])

                plt.show()


            # output steady states to yaml
            save_yaml(output_dir,'ss_ops.yaml',ssChanData)




    def runFAST_linear(self):
        """ 
        Example of running a batch of cases, in serial or in parallel
        """

        ss_opFile = os.path.join(self.FAST_steadyDirectory,'ss_ops.yaml')
        self.FAST_runDirectory = self.FAST_linearDirectory

        ## Generate case list using General Case Generator
        ## Specify several variables that change independently or collectly
        case_inputs = {}
        case_inputs[("Fst","TMax")] = {'vals':[self.TMax], 'group':0}
        case_inputs[("Fst","Linearize")] = {'vals':['True'], 'group':0}
        case_inputs[("Fst","CalcSteady")] = {'vals':['True'], 'group':0}
        case_inputs[("Fst","TrimGain")] = {'vals':[4e-5], 'group':0}

        case_inputs[("Fst","OutFileFmt")] = {'vals':[2], 'group':0}
        case_inputs[("Fst","CompMooring")] = {'vals':[0], 'group':0}

        if not self.HydroStates:
            case_inputs[("Fst","CompHydro")] = {'vals':[0], 'group':0}
        
        # InflowWind
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        if not isinstance(self.WindSpeeds,list):
            self.WindSpeeds = [self.WindSpeeds]
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':self.WindSpeeds, 'group':1}

        # AeroDyn Inputs
        case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':[1], 'group':0}

        # Servodyn Inputs
        case_inputs[("ServoDyn","PCMode")] = {'vals':[0], 'group':0}
        case_inputs[("ServoDyn","VSContrl")] = {'vals':[1], 'group':0}

        # Torque Control: these are turbine specific, update later
        case_inputs[("ServoDyn","VS_RtGnSp")] = {'vals':[7.56], 'group':0}
        case_inputs[("ServoDyn","VS_RtTq")] = {'vals':[19.62e6], 'group':0}
        case_inputs[("ServoDyn","VS_Rgn2K")] = {'vals':[3.7e5], 'group':0}
        case_inputs[("ServoDyn","VS_SlPc")] = {'vals':[10.], 'group':0}


        # Hydrodyn Inputs, these need to be state-space (2), but they should work if 0
        case_inputs[("HydroDyn","ExctnMod")] = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","RdtnMod")] = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","DiffQTF")] = {'vals':[0], 'group':0}
        case_inputs[("HydroDyn","WvDiffQTF")] = {'vals':['False'], 'group':0}
        

        # Degrees-of-freedom: set all to False & enable those defined in self
        case_inputs[("ElastoDyn","FlapDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","FlapDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","EdgeDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TeetDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","GenDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","YawDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSgDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSwDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmHvDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmRDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmPDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmYDOF")] = {'vals':['False'], 'group':0}

        for dof in self.DOFs:
            case_inputs[("ElastoDyn",dof)] = {'vals':['True'], 'group':0}
        
        # Initial Conditions
        ss_ops = load_yaml(ss_opFile)
        uu = ss_ops['Wind1VelX']

        for ic in ss_ops:
            if ic != 'Wind1VelX':
                case_inputs[("ElastoDyn",ic)] = {'vals': np.interp(case_inputs[("InflowWind","HWindSpeed")]['vals'],uu,ss_ops[ic]).tolist(), 'group': 1}

        case_inputs[('ElastoDyn','BlPitch2')] = case_inputs[('ElastoDyn','BlPitch1')]
        case_inputs[('ElastoDyn','BlPitch3')] = case_inputs[('ElastoDyn','BlPitch1')]

        # Gen Speed to track
        # set for now and update with GB ratio next
        RefGenSpeed = 0.95 * np.array(case_inputs[('ElastoDyn','RotSpeed')]['vals']) * self.GBRatio
        case_inputs[('ServoDyn','VS_RtGnSp')] = {'vals': RefGenSpeed.tolist(), 'group': 1}

        channels = {}
        for var in ["BldPitch1","BldPitch2","BldPitch3","IPDefl1","IPDefl2","IPDefl3","OoPDefl1","OoPDefl2","OoPDefl3", \
            "NcIMURAxs","TipDxc1", "TipDyc1", "Spn2MLxb1", "Spn2MLxb2","Spn2MLxb3","Spn2MLyb1", "Spn2MLyb2","Spn2MLyb3" \
                "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", \
                  "TipDzc3", "TipDxb3", "TipDyb3", "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", \
                      "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", \
                          "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", \
                              "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza", \
                                  "LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", "TipRDxr", "TipRDyr", "TipRDzr" \
                                      "TwstDefl1","TwstDefl2","TwstDefl3"]:
            channels[var] = True

        self.channels = channels

        # Lin Times
        # rotPer = 60. / np.array(case_inputs['ElastoDyn','RotSpeed']['vals'])
        # linTimes = np.linspace(self.TMax-100,self.TMax-100 + rotPer,num = self.NLinTimes, endpoint=False)
        # linTimeStrings = []

        # if linTimes.ndim == 1:
        #     linTimeStrings = np.array_str(linTimes,max_line_width=9000,precision=3)[1:-1]
        # else:
        #     for iCase in range(0,linTimes.shape[1]):
        #         linTimeStrings.append(np.array_str(linTimes[:,iCase],max_line_width=9000,precision=3)[1:-1])
        
        case_inputs[("Fst","NLinTimes")] = {'vals':[self.NLinTimes], 'group':0}

        # Trim case depends on rated wind speed (torque below-rated, pitch above)
        TrimCase = 3 * np.ones(len(self.WindSpeeds),dtype=int)
        TrimCase[np.array(self.WindSpeeds) < self.v_rated] = 2

        case_inputs[("Fst","TrimCase")] = {'vals':TrimCase.tolist(), 'group':1}


        case_inputs[("Fst","TrimTol")] = {'vals':[1e-5], 'group':0}
        

        # Generate Cases
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=self.FAST_linearDirectory, namebase='lin')

        self.case_list = case_list
        self.case_name_list = case_name_list

        outfiles = glob.glob(os.path.join(self.FAST_linearDirectory,'lin*.outb'))

        if self.overwrite or (len(outfiles) != len(self.WindSpeeds)): # if the steady output files are all there
            if self.parallel:
                self.run_multi(self.cores)
            else:
                self.run_serial()
        


def gen_linear_model(wind_speeds, Tmax=600.):
    """ 
    Generate OpenFAST linearizations across wind speeds

    Only needs to be performed once for each model

    """


    linear = LinearFAST(FAST_ver='OpenFAST', dev_branch=True);

    # fast info
    linear.weis_dir                 = os.path.dirname( os.path.dirname ( os.path.dirname( __file__ ) ) ) + os.sep
    
    linear.FAST_InputFile           = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    linear.FAST_directory           = os.path.join(linear.weis_dir, 'examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
    linear.FAST_steadyDirectory     = os.path.join(linear.weis_dir,'outputs','iea_semi_steady')
    linear.FAST_linearDirectory     = os.path.join(linear.weis_dir,'outputs','iea_semi_lin')
    linear.debug_level              = 2
    linear.dev_branch               = True
    linear.write_yaml               = True

    # do a read to get gearbox ratio
    fastRead = InputReader_OpenFAST(FAST_ver='OpenFAST', dev_branch=True)
    fastRead.FAST_InputFile = linear.FAST_InputFile   # FAST input file (ext=.fst)
    fastRead.FAST_directory = linear.FAST_directory   # Path to fst directory files

    fastRead.execute()

    # linearization setup
    linear.v_rated          = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
    linear.GBRatio          = fastRead.fst_vt['ElastoDyn']['GBRatio']
    linear.WindSpeeds       = wind_speeds  #[8.,10.,12.,14.,24.]
    linear.DOFs             = ['GenDOF'] #,'TwFADOF1','PtfmPDOF']  # enable with 
    linear.TMax             = Tmax   # should be 1000-2000 sec or more with hydrodynamic states
    linear.NLinTimes        = 12

    #if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
    linear.HydroStates      = False   # taking out to speed up for test

    # simulation setup
    linear.parallel         = True
    linear.cores            = 8

    # overwrite steady & linearizations
    linear.overwrite        = False


    # run steady state sims
    linear.runFAST_steady()

    # process results 
    linear.postFAST_steady()

    # run linearizations
    linear.runFAST_linear()



if __name__ == '__main__':
    gen_linear_model(np.arange(5,25,1,dtype=float).tolist())