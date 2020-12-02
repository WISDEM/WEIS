"""

Example script to run the stability analysis in OpenFAST

"""


from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
import numpy as np
import os

fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)

run_dir                     = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fastBatch.FAST_directory    = os.path.join(run_dir, 'OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-Monopile')   # Path to fst directory files
fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
fastBatch.FAST_runDirectory = 'stability/iea15mw'
fastBatch.debug_level       = 2

n_cores     = 1     # Number of available cores
rot_speeds  = [3.894029027708457047e+00, 4.704400394633140969e+00, 5.372491443338420325e+00, 5.889040370100040001e+00, 6.246886212676528771e+00, 6.441068123345250029e+00, 6.468894141561451150e+00, 6.607809770283845197e+00, 6.911541246624591572e+00, 7.375877914978890004e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00, 7.866444932348202634e+00]
vs_rttq     = [14562.19651233, 21253.82369411, 27719.15475684, 33305.62448534, 37476.21388142, 39842.29099748, 40187.27944808, 41931.80733996, 45875.23758968, 52246.35025829, 59427.23605761, 59427.28294601, 59427.29213803, 59427.73667015, 59427.34127935, 59427.29316426, 59427.29449649, 59427.29140488, 59427.82754031, 59427.30813839]
pitch       = [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 5.028566666950788466e+00, 7.295432135567896381e+00, 9.350468211315952516e+00, 1.132212278050246645e+01, 1.324996361480808460e+01, 1.514266119341111327e+01, 1.699625976655589454e+01, 1.880119840739241255e+01, 2.054541584970921164e+01]
hws         = [4.000000000000000000e+00, 4.832424577380788122e+00, 5.518696861384214003e+00, 6.049303000255854990e+00, 6.416887155412576149e+00, 6.616353476066063166e+00, 6.644936743441012261e+00, 6.787632781640955848e+00, 7.099629917953507174e+00, 7.576602909218082438e+00, 8.080520074579327527e+00, 8.996831778689713843e+00, 9.920398901811577019e+00, 1.096983731662181683e+01, 1.213059856802962244e+01, 1.338659092287661245e+01, 1.472040245101897860e+01, 1.611354240841405527e+01, 1.754669757589685020e+01, 1.900000000000000000e+01]
NLinTimes   = 36
TMax        = 10.

trim_case = np.zeros(len(rot_speeds), dtype=int)
trim_gain = np.zeros(len(rot_speeds))

for i in range(len(rot_speeds)):
    if pitch[i] == 0.:
        trim_case[i] = 2
        trim_gain[i] = 300
    else:
        trim_case[i] = 3
        trim_gain[i] = 0.001


LinTimes  = np.zeros([len(rot_speeds), NLinTimes])
for i in range(len(rot_speeds)):
    for j in range(NLinTimes):
        LinTimes[i,j] = TMax + j * 60./rot_speeds[i]/ NLinTimes
        
case_inputs = {}
case_inputs[("ElastoDyn","FlapDOF1")]   = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","FlapDOF2")]   = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","EdgeDOF")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TeetDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","DrTrDOF")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","GenDOF")]     = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","YawDOF")]     = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwFADOF1")]   = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwFADOF2")]   = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF1")]   = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF2")]   = {'vals':["True"], 'group':0}
case_inputs[("Fst","TMax")]             = {'vals':[TMax], 'group':0}
case_inputs[("Fst","DT")]               = {'vals':[0.0001], 'group':0}
case_inputs[("Fst","CompInflow")]       = {'vals':[1], 'group':0}
case_inputs[("Fst","CompHydro")]        = {'vals':[0], 'group':0}
case_inputs[("Fst","CompSub")]          = {'vals':[0], 'group':0}
case_inputs[("Fst","Linearize")]        = {'vals':["True"], 'group':0}
case_inputs[("Fst","CalcSteady")]       = {'vals':["True"], 'group':0}
case_inputs[("Fst","TrimCase")]         = {'vals':trim_case, 'group':1}
case_inputs[("Fst","TrimTol")]          = {'vals':[0.001], 'group':0}
case_inputs[("Fst","TrimGain")]         = {'vals':trim_gain, 'group':1}
case_inputs[("Fst","Twr_Kdmp")]         = {'vals':[0], 'group':0}
case_inputs[("Fst","Bld_Kdmp")]         = {'vals':[0], 'group':0}
case_inputs[("Fst","NLinTimes")]        = {'vals':[NLinTimes], 'group':0}
case_inputs[("Fst","LinTimes")]         = {'vals':LinTimes, 'group':1}
case_inputs[("Fst","LinInputs")]        = {'vals':[1], 'group':0}
case_inputs[("Fst","LinOutputs")]       = {'vals':[1], 'group':0}
case_inputs[("Fst","LinOutJac")]        = {'vals':["False"], 'group':0}
case_inputs[("Fst","LinOutMod")]        = {'vals':["False"], 'group':0}
case_inputs[("Fst","WrVTK")]            = {'vals':[2], 'group':0}
case_inputs[("Fst","VTK_fps")]          = {'vals':[50], 'group':0}


case_inputs[("ServoDyn","PCMode")]      = {'vals':[0], 'group':0}
case_inputs[("ServoDyn","VSContrl")]    = {'vals':[1], 'group':0}
case_inputs[("ServoDyn","GenModel")]    = {'vals':[2], 'group':0}
case_inputs[("ServoDyn","VS_RtGnSp")]   = {'vals':[9.9999E-6], 'group':0}
case_inputs[("ServoDyn","VS_Rgn2K")]    = {'vals':[9.9999E-6], 'group':0}
case_inputs[("ServoDyn","VS_SlPc")]     = {'vals':[9.9999E-6], 'group':0}
case_inputs[("ServoDyn","VS_RtTq")]     = {'vals':vs_rttq, 'group': 1}
case_inputs[("AeroDyn15","AFAeroMod")]  = {'vals':[1], 'group':0}
case_inputs[("InflowWind","WindType")]  = {'vals':[1], 'group':0}
case_inputs[("InflowWind","HWindSpeed")]= {'vals': hws, 'group': 1}
case_inputs[("Fst","OutFileFmt")]       = {'vals':[0], 'group':0}
case_inputs[("ElastoDyn","RotSpeed")]   = {'vals': rot_speeds, 'group': 1}
case_inputs[("ElastoDyn","BlPitch1")]   = {'vals': pitch, 'group': 1}
case_inputs[("ElastoDyn","BlPitch2")]   = case_inputs[("ElastoDyn","BlPitch1")]
case_inputs[("ElastoDyn","BlPitch3")]   = case_inputs[("ElastoDyn","BlPitch1")]

case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase='iea15mw')

fastBatch.case_list = case_list
fastBatch.case_name_list = case_name_list

if n_cores == 1:
    fastBatch.run_serial()
else:
    fastBatch.run_multi(n_cores)