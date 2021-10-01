from openmdao.api import ExplicitComponent, Group, IndepVarComp
import numpy as np

def assign_TMD_values(wt_opt,wt_init):
    '''
    Assign initial TMD values
    only one TMD supported for now, but it's an array in wt_init for future use
    '''
    TMD = wt_init['TMDs'][0]
    wt_opt['TMDs.mass']         = TMD['mass']
    wt_opt['TMDs.stiffness']    = TMD['stiffness']
    wt_opt['TMDs.damping']      = TMD['damping']

    wt_opt['TMDs.num_tower_StCs']   = 1


    return wt_opt


class TMDs(IndepVarComp):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.opt_options = self.options['opt_options']

        self.add_output('mass',         val=0.0, units='kg',     desc='TMD Mass')
        self.add_output('stiffness',    val=0.0, units='N/m',     desc='TMD Stiffnes')
        self.add_output('damping',      val=0.0, units='N/(m/s)',     desc='TMD Damping')
        self.add_discrete_output('num_tower_StCs',      val=0,           desc='Number of tower TMDs')


    # def compute(self, inputs, outputs):
    #     '''
    #     Someday, we'll do things like mapping the relative locations to global coords
    #     '''
    #     pass
