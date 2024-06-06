import numpy as np
import os
import openmdao.api as om
from smt.surrogate_models import KRG

class WindTurbineSM():

    def __init__(self):
        pass

    def read_doe(self, sql_files):

        # construct data structure

        data_structure = {}
        sql_file = sql_files[0]
        cr = om.CaseReader(sql_file)
        case = cr.list_cases('driver')[0]
        outputs = cr.get_case(case).outputs
        keys = outputs.keys()

        


        # read DOE data

        input_values = []
        output_values = []

        for sql_file in sql_files:
            cr = om.CaseReader(sql_file)
            cases = cr.list_cases('driver')

            for case in cases:
                outputs = cr.get_case(case).outputs



class WindTurbineModel(om.ExplicitComponent):

    def setup(self):

        # inputs (DVs, fixed parameters)
        
        pass 
