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
    
    Cost_turbine = np.zeros((ncase,))
    LCOE = np.zeros((ncase,))
    
    for i in range(ncase): 
        
        #turbine_model["materials"][1]["rho"] = rho[i]
        myopt = PoseOptimizationWISDEM(turbine_model, modeling_options, analysis_options)
        wt_opt = om.Problem(model=WindPark(modeling_options=modeling_options, opt_options=analysis_options))
        
        wt_opt = myopt.set_driver(wt_opt)
        wt_opt = myopt.set_design_variables(wt_opt, turbine_model)
        wt_opt = myopt.set_objective(wt_opt)
        wt_opt = myopt.set_recorders(wt_opt)

        #ndv = myopt.get_number_design_variables() # doesnt work when rho is a DV
        
        # initialize the open mdao problem
        
        wt_opt.setup()
        
        # assign the different values for the various subsystems
        wt_opt = yaml2openmdao(wt_opt, modeling_options, turbine_model, analysis_options)
        wt_opt = myopt.set_initial(wt_opt, turbine_model)
        
        wt_opt.run_driver()
        
        modeling_options = modeling_options
        analysis_options = analysis_options
        result = wt_opt
        MR = wt_opt.get_val('financese.machine_rating',units = 'MW')
        breakpoint()
        Cost_turbine_KW = wt_opt.get_val('financese.tcc_per_kW', units='USD/MW')[0]
        Cost_turbine[i] = Cost_turbine_KW*MR
        LCOE[i] = wt_opt.get_val('financese.lcoe',units = 'USD/kW/h')
        rho = wt_opt.get_val('floatingse.tower.rho')
        #print(LCOE)
        #print('LCOE: {:} USD/KW/h'.format(LCOE))
        
    return LCOE

#def pose_wt_model(turbine_model, modeling_options, analysis_options):
    
    

def OL_grad(turbine_model,modeling_options,analysis_options,plant_vars):
    
    n_vars = len(plant_vars)
    
    e =np.eye((n_vars))
    
    #jac = np.zeros((n_vars,))
    
    h = 1e-6 
    
    #for i in range(n_vars):
    
    xh = plant_vars + h
    
    f = Calc_TC(turbine_model, modeling_options, analysis_options, plant_vars)
    
    fh = Calc_TC(turbine_model, modeling_options, analysis_options, xh)
   
    jac = (fh-f)/h
    
    return jac
      

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
    analysis_options['recorder']['flag'] = True
    
    
    des_vars = [7800] #np.linspace(5000,10000,5)
    
    #Jac = OL_grad(turbine_model, modeling_options, analysis_options, des_vars)
    
    #plt.plot(des_vars,Jac)
    CT = Calc_TC(turbine_model, modeling_options, analysis_options, des_vars)
