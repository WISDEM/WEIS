#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:39:01 2021

@author: athulsun
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om
import weis 
import yaml 


from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from weis.glue_code.gc_PoseOptimization     import PoseOptimizationWEIS
from weis.glue_code.glue_code               import WindPark
from weis.glue_code.gc_ROSCOInputs          import assign_ROSCO_values
from weis.control.LinearModel               import LinearTurbineModel
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM

def Calc_TC(turbine_model, modeling_options, analysis_options,rho):
    
    ncase  = len(rho)
    for i in range(ncase): 
        
        turbine_model["materials"][1]["rho"] = rho[i]
        myopt = PoseOptimizationWISDEM(turbine_model, modeling_options, analysis_options)
        
        # initialize the open mdao problem
        wt_opt = om.Problem(model=WindPark(modeling_options=modeling_options, opt_options=analysis_options))
        wt_opt.setup()
        
        # assign the different values for the various subsystems
        wt_opt = yaml2openmdao(wt_opt, modeling_options, turbine_model, analysis_options)
        wt_opt = myopt.set_initial(wt_opt, turbine_model)
        wt_opt.run_model()
        
        modeling_options = modeling_options
        analysis_options = analysis_options
        result = wt_opt
        MR = wt_opt.get_val('financese.machine_rating',units = 'MW')
        Cost_turbine_KW = wt_opt.get_val('financese.tcc_per_kW', units='USD/MW')[0]
        Cost_turbine = Cost_turbine_KW*MR
        LCOE = wt_opt.get_val('financese.lcoe',units = 'USD/KW/h')
        #print(LCOE)
        print('LCOE: {:} USD/KW/h'.format(LCOE))
               
    return Cost_turbine


if __name__ == "__main__":
    
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # location of 
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    fname_wt_input   = mydir + os.sep + "IEA-15-floating.yaml"
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    
    wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
    turbine_model, modeling_options, analysis_options = wt_ontology.get_input_data()
    
    
    modeling_options['Level1']['flag'] = False
    modeling_options['Level2']['flag'] = False
    modeling_options['Level3']['flag'] = False
    modeling_options['ROSCO']['flag'] = False
    #analysis_options['recorder']['flag'] = False
    
    
    rho_s = [7800]
    
    
    CT = Calc_TC(turbine_model, modeling_options, analysis_options, rho_s)
