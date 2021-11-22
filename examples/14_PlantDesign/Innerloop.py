#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import weis.inputs as sch
import os
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.control.LinearModel           import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse.CaseGen_General import case_naming
import pickle
from weis.control.dtqp_wrapper          import dtqp_wrapper
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
import matplotlib.pyplot as plt
import numpy as np
from mat4py import loadmat  
from copy                                   import deepcopy

from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from weis.glue_code.gc_PoseOptimization     import PoseOptimizationWEIS
from weis.glue_code.glue_code               import WindPark
from weis.glue_code.gc_ROSCOInputs          import assign_ROSCO_values
from weis.control.LinearModel               import LinearTurbineModel
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM

import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance


def Calc_rho(mydir):
    
    sql_path = mydir + os.sep + "outputs" + os.sep + "log_opt_FF.sql"
    
    cr = om.CaseReader(sql_path)
    
    driver_cases = cr.get_cases("driver")
    
    DVs = []
    
    for idx, case in enumerate(driver_cases):
        
        dvs = case.get_design_vars(scaled=False)
        for key in dvs.keys():
            DVs.append(dvs[key][0])
            
    return DVs
        
    


def Calc_LCOE(Turbine_Cost,AEP,MR,opex_per_kW,bos_per_kW,fcr,wlf):
    
    prob = om.Problem()
    prob.model = PlantFinance()
    prob.setup() 
    
    ncase = len(Turbine_Cost)
    lcoe = np.zeros((ncase,))
    
    for i in range(ncase):
        # Set variable inputs with intended units
        prob.set_val("machine_rating", MR, units="MW")
        prob.set_val("tcc_per_kW", Turbine_Cost[i]/MR, units="USD/MW")
        prob.set_val("turbine_number", 1)
        prob.set_val("opex_per_kW", opex_per_kW, units="USD/kW/yr")
        prob.set_val("fixed_charge_rate", fcr)
        prob.set_val("bos_per_kW", bos_per_kW, units="USD/kW")
        prob.set_val("wake_loss_factor", wlf)
        prob.set_val("turbine_aep", AEP[i], units="MW*h")
        
        prob.run_model()
        lcoe[i] = prob.get_val('lcoe')
        
    prob.model.list_inputs(units=True)
    prob.model.list_outputs(units=True)

    return lcoe



def Calc_TC(fname_wt_input, fname_modeling_options, fname_analysis_options,rho):
    
    wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
    TM, MO, AO = wt_ontology.get_input_data()
    
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
    bos_per_kW = TM['costs']['bos_per_kW']
    fcr = TM['costs']['fixed_charge_rate']
    wlf = TM['costs']['wake_loss_factor']
    
    ncase  = len(rho)
    
    Turbine_Cost = np.zeros((ncase,))
    
    for i in range(ncase): 
        

        TM["materials"][1]["rho"] = rho[i]
        myopt = PoseOptimizationWISDEM(TM, MO, AO)
        #breakpoint()
        # initialize the open mdao problem
        wt_opt = om.Problem(model=WindPark(modeling_options=MO, opt_options=AO))
        wt_opt.setup()
        
        # assign the different values for the various subsystems
        wt_opt = yaml2openmdao(wt_opt, MO, TM, AO)
        wt_opt = myopt.set_initial(wt_opt, TM)
        wt_opt.run_model()
        


        MR = wt_opt.get_val('financese.machine_rating',units = 'MW')
        Cost_turbine_MW = wt_opt.get_val('financese.tcc_per_kW', units='USD/MW')[0]
        
        Turbine_Cost[i] = Cost_turbine_MW*MR
        
        
    return Turbine_Cost,MR,opex_per_kW,bos_per_kW,fcr,wlf

def Calc_AEP(summary_stats,dlc_generator,Turbine_class):
    
    
    idx_pwrcrv = []
    U = []
    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].label == '1.1':
            idx_pwrcrv = np.append(idx_pwrcrv, i_case)
            U = np.append(U, dlc_generator.cases[i_case].URef)

    stats_pwrcrv = summary_stats.iloc[idx_pwrcrv].copy()
    
    if len(U) > 1:
        pp = PowerProduction(Turbine_class)
        pwr_curve_vars   = ["GenPwr", "RtAeroCp", "RotSpeed", "BldPitch1"]
        AEP, perf_data = pp.AEP(stats_pwrcrv, U, pwr_curve_vars)
    else:
        AEP = 0
    
    return AEP
    
class dict2class(object):
    
    def __init__(self,my_dict):
        
        for key in my_dict:
            setattr(self,key,my_dict[key])
            
        self.A_ops = self.A
        self.B_ops = self.B
        self.C_ops = self.C
        self.D_ops = self.D

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == "__main__":
    # read WEIS options:
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    modeling_options = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input   = mydir + os.sep + "IEA-15-floating.yaml"
    wt_init          = sch.load_geometry_yaml(fname_wt_input)
    
    Turbine_class = wt_init["assembly"]["turbine_class"]
    
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)

    # Wind turbine inputs 
    ws_cut_in               = wt_init['control']['supervisory']['Vin']
    ws_cut_out              = wt_init['control']['supervisory']['Vout']
    ws_rated                = 11.2
    wind_speed_class        = wt_init['assembly']['turbine_class']
    wind_turbulence_class   = wt_init['assembly']['turbulence_class']

    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']
    
    # Initialize the generator
    fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(ws_cut_in, ws_cut_out, ws_rated, wind_speed_class, wind_turbulence_class, fix_wind_seeds, fix_wave_seeds, metocean)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)


    # generate wind files
    FAST_namingOut = 'oloc'
    wind_directory = 'outputs/oloc/wind'
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init['assembly']['rotor_diameter']
    hub_height = wt_init['assembly']['hub_height']

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases

    level2_disturbance = []

    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)

        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(WindFile_name[i_case])
        ts_file.compute_rot_avg(rotorD/2)
        u_h         = ts_file['rot_avg'][0,:]
        
        
        off = max(u_h) - 25
        ind = u_h > 25;
        
        # remove any windspeeds > 25 m/s
        if ind.any():
            u_h[ind] = u_h[ind] - off
        
        print(np.mean(u_h))
        tt = ts_file['t']
        level2_disturbance.append({'Time':tt, 'Wind': u_h})
        
       
    # Linear Model
    pkl_file = mydir + os.sep + "outputs" + os.sep + "tower_doe_run" + os.sep + "ABCD_matrices.pkl" 
    #pkl_file = mydir + os.sep + 'LinModel_removed_fix.pkl'
    
    with open(pkl_file,"rb") as handle:
        ABCD_list = pickle.load(handle)

    
    fst_vt = {}
    fst_vt['DISCON_in'] = {}
    fst_vt['DISCON_in']['PC_RefSpd'] = 0.7853192931562493

    la = LoadsAnalysis(
            outputs=[],
        )

    magnitude_channels = {
        "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
        "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
        "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
        }

    run_directory = modeling_options['General']['openfast_configuration']['OF_run_dir']
    
    
    n_cases = len(ABCD_list)
    AEP = np.zeros((n_cases,)) 
    
    # Evaluate the cost
    #rho = Calc_rho(mydir)
    #rho = [7800,7800,7800,7800,7800]
    
    #TC,MR,opex_per_kW,bos_per_kW,fcr,wlf = Calc_TC(fname_wt_input, fname_modeling_options, fname_analysis_options, rho)
    
    
    for n in range(n_cases):
        ABCD = ABCD_list[n]
        
        LinearTurbine = dict2class(ABCD)
        
        summary_stats, extreme_table, DELs, Damage = dtqp_wrapper(
        LinearTurbine, 
        level2_disturbance, 
        analysis_options, 
        fst_vt, 
        la, 
        magnitude_channels, 
        run_directory
        )
          
        
        AEP[n] = Calc_AEP(summary_stats,dlc_generator,Turbine_class)
        
        
    # LCOE = Calc_LCOE(TC,AEP,MR,opex_per_kW,bos_per_kW,fcr,wlf)
    
    # fig,ax = plt.subplots(1,1)
    # ax.plot(rho,LCOE,'*-',linewidth = 2,markersize = 10)
    # ax.set_xlabel('rho [kg/m**3]')
    # ax.set_ylabel('LCOE [USD/kW/h]')
    # ax.set_title('LCOE vs Tower Density')
    
    # fig,ax = plt.subplots(1,1)
    # ax.plot(rho,AEP,'*-',linewidth = 2,markersize = 10)
    # ax.set_xlabel('rho [kg/m**3]')
    # ax.set_ylabel('AEP [kWh]')
    # ax.set_title(' AEP vs Tower Desity')
