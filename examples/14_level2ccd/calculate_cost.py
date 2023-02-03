import os
import numpy as np
from scipy.io import savemat
from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM
import openmdao.api as om
import weis.inputs as sch
from copy import deepcopy
from weis.glue_code.glue_code               import WindPark
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao

def Calc_TC(wt_init, modeling_options, analysis_options,DV):
    
    # create a copy of modeling,analysis, geometry options
    MO = deepcopy(modeling_options)
    AO = deepcopy(analysis_options)
    TM = deepcopy(wt_init)
    
    CS = DV[0]; CD = DV[1]

    # switch off important flags
    MO['Level1']['flag'] = False
    MO['Level2']['flag'] = False
    MO['Level3']['flag'] = False
    MO['ROSCO']['flag'] = False
    AO['recorder']['flag'] = False
    AO['driver']['optimization']['flag'] = False
    AO['driver']['design_of_experiments']['flag'] = False
    TM['costs']['turbine_number'] = 1
    
    opex_per_kW = TM['costs']['opex_per_kW']
    fcr = TM['costs']['fixed_charge_rate']
    wlf = TM['costs']['wake_loss_factor']
    
    
    # update values

    # column spacing
    TM["components"]["floating_platform"]["joints"][2]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][3]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][4]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][5]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][6]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][7]["location"][0] = CS

    # ballast volume
    TM["components"]["floating_platform"]["members"][1]["outer_shape"]["outer_diameter"]['values'] = [CD,CD]
    TM["components"]["floating_platform"]["members"][2]["outer_shape"]["outer_diameter"]['values'] = [CD,CD]
    TM["components"]["floating_platform"]["members"][3]["outer_shape"]["outer_diameter"]['values'] = [CD,CD]


    # pose optimization
    myopt = PoseOptimizationWISDEM(TM, MO, AO)
    
    # initialize the open mdao problem
    wt_opt = om.Problem(model=WindPark(modeling_options=MO, opt_options=AO))
    wt_opt.setup()
    
    # assign the different values for the various subsystems
    wt_opt = yaml2openmdao(wt_opt, MO, TM, AO)
    wt_opt = myopt.set_initial(wt_opt, TM)
    
    # run model
    wt_opt.run_model()
    
   # extract values
    MR = wt_opt.get_val('financese.machine_rating',units = 'kW')
    tcc_per_kW = wt_opt.get_val('financese.tcc_per_kW', units='USD/kW')[0]
    bos_per_kW = wt_opt.get_val('financese.bos_per_kW', units='USD/kW')[0]


    return tcc_per_kW,bos_per_kW,MR,opex_per_kW,fcr,wlf




## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
wisdem_examples        = os.path.join(os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ), "WISDEM", "examples")
fname_wt_input         = os.path.join(wisdem_examples,'09_floating/IEA-15-240-RWT_VolturnUS-S.yaml')

fname_modeling_options = run_dir + os.sep + "modeling_options.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options.yaml"

wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
wt_init,modeling_options,analysis_options = wt_ontology.get_input_data()

cs_bounds = [36,78]
cd_bounds = [6,24]

# number of samples
ns = 2

# create samples
CS_b = np.linspace(cs_bounds[0],cs_bounds[1],ns)
CD_b = np.linspace(cd_bounds[0],cd_bounds[1],ns)

# meshgrid
CS,CD = np.meshgrid(CS_b,CD_b)

CS = CS.reshape([-1,1],order = 'F'); CS = CS.squeeze()
CD = CD.reshape([-1,1],order = 'F'); CD = CD.squeeze()

DV_ = np.zeros((ns*ns,2))
DV_[:,0] = CS; DV_[:,1] = CD
#DV_ = np.array([[51.75,12.5],
#                [36,6]])


# initialize
TC_kW = np.zeros((ns*ns))
OC_kW = np.zeros((ns*ns))
BC_kW = np.zeros((ns*ns))

# calculate turbine, balance of system costs

for i in range(ns*ns):
    
    tcckW,boskW,MR,opexkW,fcr,wlf = Calc_TC(wt_init, modeling_options, analysis_options,DV_[i,:])
    
    # assign 
    TC_kW[i] = tcckW
    BC_kW[i] = boskW
    OC_kW[i] = opexkW 
        
    
# compile
matname = 'turbinecost_RWT.mat'

#turbine_cost = {'DV':DV_,'TC_kW':TC_kW,'OC_kW':OC_kW,'BC_kW':BC_kW,'wlf':wlf,'fcr':fcr,'MR':MR}

        