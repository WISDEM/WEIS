"""
A basic python script that demonstrates how to use the FST8 reader, writer, and wrapper in a purely
python setting. These functions are constructed to provide a simple interface for controlling FAST
programmatically with minimal additional dependencies.
"""

import os
import sys
import platform
import multiprocessing as mp

from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST, InputReader_FAST7
from weis.aeroelasticse.FAST_writer import InputWriter_OpenFAST, InputWriter_FAST7
from weis.aeroelasticse.FAST_wrapper import FastWrapper
from weis.aeroelasticse.FAST_post   import FAST_IO_timeseries
from pCrunch.io import OpenFASTOutput
from pCrunch import LoadsAnalysis

import numpy as np

from ctypes import create_string_buffer, c_double
import weis

weis_dir = os.path.dirname( os.path.dirname(os.path.realpath(weis.__file__) ) )  # get path to this file
lib_dir  = os.path.abspath( os.path.join(weis_dir, 'local/lib/') )
openfast_pydir = os.path.join(weis_dir,'OpenFAST','glue-codes','python')
sys.path.append(openfast_pydir)
from openfast_library import FastLibAPI

mactype = platform.system().lower()
if mactype == "linux" or mactype == "linux2":
    libext = ".so"
elif mactype == "darwin":
    libext = '.dylib'
elif mactype == "win32":
    libext = '.dll'
elif mactype == "cygwin":
    libext = ".dll"
else:
    raise ValueError('Unknown platform type: '+mactype)


magnitude_channels = {
    'LSShftF': ["RotThrust", "LSShftFys", "LSShftFzs"], 
    'LSShftM': ["RotTorq", "LSSTipMys", "LSSTipMzs"],
    'RootMc1': ["RootMxc1", "RootMyc1", "RootMzc1"],
    'RootMc2': ["RootMxc2", "RootMyc2", "RootMzc2"],
    'RootMc3': ["RootMxc3", "RootMyc3", "RootMzc3"],
    'TipDc1': ['TipDxc1', 'TipDyc1', 'TipDzc1'],
    'TipDc2': ['TipDxc2', 'TipDyc2', 'TipDzc2'],
    'TipDc3': ['TipDxc3', 'TipDyc3', 'TipDzc3'],
}

fatigue_channels = {
    'RootMc1': 10,
    'RootMc2': 10,
    'RootMc3': 10,
    'RootMyb1': 10,
    'RootMyb2': 10,
    'RootMyb3': 10,
    'TwrBsMyt': 10
}

channel_extremes = [
    'RotSpeed',
    'BldPitch1','BldPitch2','BldPitch3',
    "RotThrust","LSShftFys","LSShftFzs","RotTorq","LSSTipMys","LSSTipMzs","LSShftF","LSShftM",
    'Azimuth',
    'TipDxc1',
    'TipDxc2',
    'TipDxc3',
    "RootMxc1","RootMyc1","RootMzc1",
    "RootMxc2","RootMyc2","RootMzc2",
    "RootMxc3","RootMyc3","RootMzc3",
    'B1N1Fx','B1N2Fx','B1N3Fx','B1N4Fx','B1N5Fx','B1N6Fx','B1N7Fx','B1N8Fx','B1N9Fx',
    'B1N1Fy','B1N2Fy','B1N3Fy','B1N4Fy','B1N5Fy','B1N6Fy','B1N7Fy','B1N8Fy','B1N9Fy',
    'B2N1Fx','B2N2Fx','B2N3Fx','B2N4Fx','B2N5Fx','B2N6Fx','B2N7Fx','B2N8Fx','B2N9Fx',
    'B2N1Fy','B2N2Fy','B2N3Fy','B2N4Fy','B2N5Fy','B2N6Fy','B2N7Fy','B2N8Fy','B2N9Fy',
    "B3N1Fx","B3N2Fx","B3N3Fx","B3N4Fx","B3N5Fx","B3N6Fx","B3N7Fx","B3N8Fx","B3N9Fx",
    "B3N1Fy","B3N2Fy","B3N3Fy","B3N4Fy","B3N5Fy","B3N6Fy","B3N7Fy","B3N8Fy","B3N9Fy",
]

la = LoadsAnalysis(
    outputs=[],
    magnitude_channels=magnitude_channels,
    fatigue_channels=fatigue_channels,
    extreme_channels=channel_extremes,
)


class runFAST_pywrapper(object):

    def __init__(self, **kwargs):
        self.FAST_ver = 'OPENFAST' #(FAST7, FAST8, OPENFAST)

        self.FAST_exe           = None
        self.FAST_lib           = None
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.FAST_namingOut     = None
        self.read_yaml          = False
        self.write_yaml         = False
        self.fst_vt             = {}
        self.case               = {}     # dictionary of variable values to change
        self.channels           = {}     # dictionary of output channels to change
        self.debug_level        = 0
        self.keep_time          = False

        self.overwrite_outfiles = True   # True: existing output files will be overwritten, False: if output file with the same name already exists, OpenFAST WILL NOT RUN; This is primarily included for code debugging with OpenFAST in the loop or for specific Optimization Workflows where OpenFAST is to be run periodically instead of for every objective function anaylsis

        # Optional population class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(runFAST_pywrapper, self).__init__()

    def execute(self):

        # FAST version specific initialization
        if self.FAST_ver.lower() == 'fast7':
            reader = InputReader_FAST7(FAST_ver=self.FAST_ver)
            writer = InputWriter_FAST7(FAST_ver=self.FAST_ver)
        elif self.FAST_ver.lower() in ['fast8','openfast']:
            reader = InputReader_OpenFAST(FAST_ver=self.FAST_ver)
            writer = InputWriter_OpenFAST(FAST_ver=self.FAST_ver)

        # Read input model, FAST files or Yaml
        if self.fst_vt == {}:
            if self.read_yaml:
                reader.FAST_yamlfile = self.FAST_yamlfile_in
                reader.read_yaml()
            else:
                reader.FAST_InputFile = self.FAST_InputFile
                reader.FAST_directory = self.FAST_directory
                reader.execute()
        
            # Initialize writer variables with input model
            writer.fst_vt = self.fst_vt = reader.fst_vt
        else:
            writer.fst_vt = self.fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut = self.FAST_namingOut
        # Make any case specific variable changes
        if self.case:
            writer.update(fst_update=self.case)
        # Modify any specified output channels
        if self.channels:
            writer.update_outlist(self.channels)
        # Write out FAST model
        writer.execute()
        if self.write_yaml:
            writer.FAST_yamlfile = self.FAST_yamlfile_out
            writer.write_yaml()

        FAST_directory = os.path.split(writer.FAST_InputFileOut)[0]
        input_file_name = create_string_buffer(os.path.abspath(writer.FAST_InputFileOut).encode('utf-8'))
        t_max = c_double(self.fst_vt['Fst']['TMax'])

        orig_dir = os.getcwd()
        os.chdir(FAST_directory)
        
        openfastlib = FastLibAPI(self.FAST_lib, input_file_name, t_max)
        openfastlib.fast_run()

        output_dict = {}
        for i, channel in enumerate(openfastlib.output_channel_names):
            output_dict[channel] = openfastlib.output_values[:,i]

        output = OpenFASTOutput.from_dict(output_dict, self.FAST_namingOut, magnitude_channels=magnitude_channels)
        case_name, sum_stats, extremes, dels = la._process_output(output)

        # if save_file: write_fast
        os.chdir(orig_dir)

        if not self.keep_time: output_dict = None
        return case_name, sum_stats, extremes, dels, output_dict


class runFAST_pywrapper_batch(object):

    def __init__(self, **kwargs):

        self.FAST_ver           = 'OpenFAST'
        self.FAST_exe           = os.path.join(weis_dir, 'local/bin/openfast')   # Path to executable
        self.FAST_lib           = os.path.join(lib_dir, 'libopenfastlib'+libext) 
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.debug_level        = 0

        self.read_yaml          = False
        self.FAST_yamlfile_in   = ''
        self.fst_vt             = {}
        self.write_yaml         = False
        self.FAST_yamlfile_out  = ''

        self.case_list          = []
        self.case_name_list     = []
        self.channels           = {}

        self.overwrite_outfiles = True
        self.keep_time          = False
        
        self.post               = None

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(runFAST_pywrapper_batch, self).__init__()

    def create_case_data(self):

        case_data_all = []
        for i in range(len(self.case_list)):
            case_data = {}
            case_data['case'] = self.case_list[i]
            case_data['case_name'] = self.case_name_list[i]
            case_data['FAST_ver'] = self.FAST_ver
            case_data['FAST_exe'] = self.FAST_exe
            case_data['FAST_lib'] = self.FAST_lib
            case_data['FAST_runDirectory'] = self.FAST_runDirectory
            case_data['FAST_InputFile'] = self.FAST_InputFile
            case_data['FAST_directory'] = self.FAST_directory
            case_data['read_yaml'] = self.read_yaml
            case_data['FAST_yamlfile_in'] = self.FAST_yamlfile_in
            case_data['fst_vt'] = self.fst_vt
            case_data['write_yaml'] = self.write_yaml
            case_data['FAST_yamlfile_out'] = self.FAST_yamlfile_out
            case_data['channels'] = self.channels
            case_data['debug_level'] = self.debug_level
            case_data['overwrite_outfiles'] = self.overwrite_outfiles
            case_data['keep_time'] = self.keep_time
            case_data['post'] = self.post

            case_data_all.append(case_data)

        return case_data_all
    
    def run_serial(self):
        # Run batch serially
        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        case_data_all = self.create_case_data()
            
        ss = {}
        et = {}
        dl = {}
        ct = []
        for c in case_data_all:
            _name, _ss, _et, _dl, _ct = evaluate(c)
            ss[_name] = _ss
            et[_name] = _et
            dl[_name] = _dl
            ct.append(_ct)
        
        summary_stats, extreme_table, DELs = la.post_process(ss, et, dl)

        return summary_stats, extreme_table, DELs, ct

    def run_multi(self, cores=None):
        # Run cases in parallel, threaded with multiprocessing module

        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        if not cores:
            cores = mp.cpu_count()
        pool = mp.Pool(cores)

        case_data_all = self.create_case_data()

        output = pool.map(evaluate_multi, case_data_all)
        pool.close()
        pool.join()

        ss = {}
        et = {}
        dl = {}
        ct = []
        for _name, _ss, _et, _dl, _ct in output:
            ss[_name] = _ss
            et[_name] = _et
            dl[_name] = _dl
            ct.append(_ct)

        summary_stats, extreme_table, DELs = la.post_process(ss, et, dl)

        return summary_stats, extreme_table, DELs, ct

    def run_mpi(self, mpi_comm_map_down):

        # Run in parallel with mpi
        from mpi4py import MPI

        # mpi comm management
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sub_ranks = mpi_comm_map_down[rank]
        size = len(sub_ranks)

        N_cases = len(self.case_list)
        N_loops = int(np.ceil(float(N_cases)/float(size)))
        
        # file management
        if not os.path.exists(self.FAST_runDirectory) and rank == 0:
            os.makedirs(self.FAST_runDirectory)

        case_data_all = self.create_case_data()

        output = []
        for i in range(N_loops):
            idx_s    = i*size
            idx_e    = min((i+1)*size, N_cases)

            for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
                data   = [evaluate_multi, case_data]
                rank_j = sub_ranks[j]
                comm.send(data, dest=rank_j, tag=0)

            # for rank_j in sub_ranks:
            for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
                rank_j = sub_ranks[j]
                data_out = comm.recv(source=rank_j, tag=1)
                output.append(data_out)

        ss = {}
        et = {}
        dl = {}
        ct = []
        for _name, _ss, _et, _dl, _ct in output:
            ss[_name] = _ss
            et[_name] = _et
            dl[_name] = _dl
            ct.append(_ct)

        summary_stats, extreme_table, DELs = la.post_process(ss, et, dl)

        return summary_stats, extreme_table, DELs, ct



def evaluate(indict):
    # Batch FAST pyWrapper call, as a function outside the runFAST_pywrapper_batch class for pickle-ablility

    known_keys = ['case', 'case_name', 'FAST_ver', 'FAST_exe', 'FAST_lib', 'FAST_runDirectory',
                  'FAST_InputFile', 'FAST_directory', 'read_yaml', 'FAST_yamlfile_in', 'fst_vt',
                  'write_yaml', 'FAST_yamlfile_out', 'channels', 'debug_level', 'overwrite_outfiles', 'keep_time', 'post']
    for k in indict:
        if k in known_keys: continue
        print(f'WARNING: Unknown OpenFAST executation parameter, {k}')
    
    fast = runFAST_pywrapper(FAST_ver=indict['FAST_ver'])     # FAST_ver = "OpenFAST"
    fast.FAST_exe           = indict['FAST_exe']              # Path to FAST
    fast.FAST_lib           = indict['FAST_lib']              # Path to FAST
    fast.FAST_InputFile     = indict['FAST_InputFile']        # Name of the fst - does not include full path
    fast.FAST_directory     = indict['FAST_directory']        # Path to directory containing the case files
    fast.FAST_runDirectory  = indict['FAST_runDirectory']     # Where 

    fast.read_yaml          = indict['read_yaml']
    fast.FAST_yamlfile_in   = indict['FAST_yamlfile_in']
    fast.fst_vt             = indict['fst_vt']
    fast.write_yaml         = indict['write_yaml']
    fast.FAST_yamlfile_out  = indict['FAST_yamlfile_out']

    fast.FAST_namingOut     = indict['case_name']
    fast.case               = indict['case']
    fast.channels           = indict['channels']
    fast.debug_level        = indict['debug_level']

    fast.overwrite_outfiles = indict['overwrite_outfiles']
    fast.keep_time = indict['keep_time']

    FAST_Output = fast.execute()
    return FAST_Output

def evaluate_multi(indict):
    # helper function for running with multiprocessing.Pool.map
    # converts list of arguement values to arguments
    return evaluate(indict)
