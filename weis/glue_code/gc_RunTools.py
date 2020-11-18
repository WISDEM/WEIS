import os
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np
from wisdem.commonse.mpi_tools import MPI

class Outputs_2_Screen(om.ExplicitComponent):
    # Class to print outputs on screen
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        n_te_flaps = modeling_options['blade']['n_te_flaps']

        self.add_input('aep',           val=0.0, units = 'GW * h')
        self.add_input('blade_mass',    val=0.0, units = 'kg')
        self.add_input('lcoe',          val=0.0, units = 'USD/MW/h')
        self.add_input('DEL_RootMyb',   val=0.0, units = 'N*m')
        self.add_input('DEL_TwrBsMyt',  val=0.0, units = 'N*m')
        self.add_input('PC_omega',      val=0.0, units = 'rad/s')
        self.add_input('PC_zeta',       val=0.0)
        self.add_input('VS_omega',      val=0.0, units='rad/s')
        self.add_input('VS_zeta',       val=0.0)
        self.add_input('Flp_omega',     val=0.0, units='rad/s')
        self.add_input('Flp_zeta',      val=0.0)
        self.add_input('IPC_Ki1p',      val=0.0, units='rad/(N*m)')
        self.add_input('tip_deflection',val=0.0, units='m')
        self.add_input('te_flap_end'   ,val=np.zeros(n_te_flaps))
        self.add_input('rotor_overspeed',val=0.0)

    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('Turbine AEP: {:<8.10f} GWh'.format(inputs['aep'][0]))
        print('Blade Mass:  {:<8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:<8.10f} USD/MWh'.format(inputs['lcoe'][0]))
        print('Tip Defl.:   {:<8.10f} m'.format(inputs['tip_deflection'][0]))
        
        # OpenFAST simulation summary
        if self.options['modeling_options']['Analysis_Flags']['OpenFAST']: 
            # Print optimization variables
            if self.options['opt_options']['optimization_variables']['control']['servo']['pitch_control']['flag']:
                print('Pitch PI gain inputs: pc_omega = {:2.3f}, pc_zeta = {:2.3f}'.format(inputs['PC_omega'][0], inputs['PC_zeta'][0]))
            if self.options['opt_options']['optimization_variables']['control']['servo']['torque_control']['flag']:
                print('Torque PI gain inputs: vs_omega = {:2.3f}, vs_zeta = {:2.3f}'.format(inputs['VS_omega'][0], inputs['VS_zeta'][0]))
            if self.options['opt_options']['optimization_variables']['control']['servo']['flap_control']['flag']:
                print('Flap PI gain inputs: flp_omega = {:2.3f}, flp_zeta = {:2.3f}'.format(inputs['Flp_omega'][0], inputs['Flp_zeta'][0]))
            if self.options['opt_options']['optimization_variables']['control']['servo']['ipc_control']['flag']:
                print('IPC Ki1p = {:2.3e}'.format(inputs['IPC_Ki1p'][0]))
            if self.options['opt_options']['optimization_variables']['blade']['dac']['te_flap_end']['flag']:
                print('Trailing-edge flap end = {:2.3f}%'.format(inputs['te_flap_end'][0]*100.))
            # Print merit figure
            if self.options['opt_options']['merit_figure'] == 'DEL_TwrBsMyt':
                print('DEL(TwrBsMyt): {:<8.10f} Nm'.format(inputs['DEL_TwrBsMyt'][0]))
            if self.options['opt_options']['merit_figure'] == 'DEL_RootMyb':
                print('Max DEL(RootMyb): {:<8.10f} Nm'.format(inputs['DEL_RootMyb'][0]))
            if self.options['opt_options']['merit_figure'] == 'rotor_overspeed':
                print('rotor_overspeed: {:<8.10f} Nm'.format(inputs['rotor_overspeed'][0]))
            # Print constraints
            if self.options['opt_options']['constraints']['control']['rotor_overspeed']['flag']:
                print('rotor_overspeed: {:<8.10f} %'.format(inputs['rotor_overspeed'][0]))
        
        print('########################################')
