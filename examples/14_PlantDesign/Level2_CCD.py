#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:24:19 2021

@author: athulsun
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om
import weis 
import yaml 
import logging


from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from weis.glue_code.gc_PoseOptimization     import PoseOptimizationWEIS
from weis.glue_code.glue_code               import WindPark
from weis.glue_code.gc_ROSCOInputs          import assign_ROSCO_values
from weis.control.LinearModel               import LinearTurbineModel
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM


def run_level2CCD(turbine_model,modeling_options,analysis_options):
    
    # Load all yaml inputs and validate (also fills in defaults)
    wt_ontology = WindTurbineOntologyPythonWEIS(turbine_model, modeling_options, analysis_options)
    turbine_model, modeling_options, analysis_options = wt_ontology.get_input_data()
    
    # Pose optimization problem and initialize openmdao problem
    myopt = PoseOptimizationWISDEM(turbine_model, modeling_options, analysis_options)
    
    # create logger
    folder_output = analysis_options["general"]["folder_output"]
    
    # create logger
    logger = logging.getLogger("wisdem/weis")
    logger.setLevel(logging.INFO)

    # create handlers
    ht = logging.StreamHandler()
    ht.setLevel(logging.WARNING)

    flog = os.path.join(folder_output, analysis_options["general"]["fname_output"] + ".log")
    hf = logging.FileHandler(flog, mode="w")
    hf.setLevel(logging.INFO)

    # create formatters
    formatter_t = logging.Formatter("%(module)s:%(funcName)s:%(lineno)d %(levelname)s:%(message)s")
    formatter_f = logging.Formatter(
        "P%(process)d %(asctime)s %(module)s:%(funcName)s:%(lineno)d %(levelname)s:%(message)s"
    )

    # add formatter to handlers
    ht.setFormatter(formatter_t)
    hf.setFormatter(formatter_f)

    # add handlers to logger
    logger.addHandler(ht)
    logger.addHandler(hf)
    logger.info("Started")
    
    
    
    