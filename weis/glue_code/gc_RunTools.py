import os
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np
from openmdao.utils.mpi import MPI

class Outputs_2_Screen(om.ExplicitComponent):
    # Class to print outputs on screen
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        n_te_flaps = modeling_options['WISDEM']['RotorSE']['n_te_flaps']

        self.add_input('aep',           val=0.0, units = 'GW * h')
        self.add_input('blade_mass',    val=0.0, units = 'kg')
        self.add_input('lcoe',          val=0.0, units = 'USD/MW/h')
        self.add_input('DEL_RootMyb',   val=0.0, units = 'N*m')
        self.add_input('DEL_TwrBsMyt',  val=0.0, units = 'N*m')
        self.add_input('Std_PtfmPitch', val=0.0, units = 'deg')
        if modeling_options['ROSCO']['linmodel_tuning']['type'] == 'robust':
            n_PC = 1
        else:
            n_PC = len(modeling_options['ROSCO']['U_pc'])
        self.add_input('omega_pc',      val=np.zeros(n_PC), units = 'rad/s')
        self.add_input('zeta_pc',       val=np.zeros(n_PC))
        self.add_input('Kp_float',      val=0.0, units = 's')
        self.add_input('ptfm_freq',     val=0.0, units = 'rad/s')
        self.add_input('omega_vs',      val=0.0, units='rad/s')
        self.add_input('zeta_vs',       val=0.0)
        self.add_input('flp_kp_norm',   val=0.0)
        self.add_input('flp_tau',       val=0.0, units='s')
        self.add_input('IPC_Kp1p',      val=0.0, units='s')
        self.add_input('IPC_Ki1p',      val=0.0,)
        self.add_input('tip_deflection',val=0.0, units='m')
        self.add_input('te_flap_end'   ,val=np.zeros(n_te_flaps))
        self.add_input('rotor_overspeed',val=0.0)
        self.add_input('Max_PtfmPitch',val=0.0)
        if modeling_options['OL2CL']['flag']:
            self.add_input('OL2CL_pitch',   val=0.0, units = 'deg')

    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('Turbine AEP: {:<8.10f} GWh'.format(inputs['aep'][0]))
        print('Blade Mass:  {:<8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:<8.10f} USD/MWh'.format(inputs['lcoe'][0]))
        print('Tip Defl.:   {:<8.10f} m'.format(inputs['tip_deflection'][0]))
        
        # OpenFAST simulation summary
        if self.options['modeling_options']['OpenFAST']['flag']: 
            # Print optimization variables  TODO: automate?
            

            # Print merit figure
            if self.options['opt_options']['merit_figure'] == 'DEL_TwrBsMyt':
                print('DEL(TwrBsMyt): {:<8.10f} Nm'.format(inputs['DEL_TwrBsMyt'][0]))
            if self.options['opt_options']['merit_figure'] == 'DEL_RootMyb':
                print('Max DEL(RootMyb): {:<8.10f} Nm'.format(inputs['DEL_RootMyb'][0]))
            if self.options['opt_options']['merit_figure'] == 'rotor_overspeed':
                print('rotor_overspeed: {:<8.10f} %'.format(inputs['rotor_overspeed'][0]*100))
            if self.options['opt_options']['merit_figure'] == 'Std_PtfmPitch':
                print('Std_PtfmPitch: {:<8.10f} deg.'.format(inputs['Std_PtfmPitch'][0]))
            if self.options['opt_options']['merit_figure'] == 'OL2CL_pitch':
                print('RMS Pitch Error (OL2CL): {:<8.10f} deg.'.format(inputs['OL2CL_pitch'][0]))
            # Print constraints
            if self.options['opt_options']['constraints']['control']['rotor_overspeed']['flag']:
                print('rotor_overspeed: {:<8.10f} %'.format(inputs['rotor_overspeed'][0]*100))
            if self.options['opt_options']['constraints']['control']['Max_PtfmPitch']['flag']:
                print('Max_PtfmPitch: {:<8.10f} deg.'.format(inputs['Max_PtfmPitch'][0]))
            if self.options['opt_options']['constraints']['control']['Std_PtfmPitch']['flag']:
                print('Std_PtfmPitch: {:<8.10f} deg.'.format(inputs['Std_PtfmPitch'][0]))
        
        print('########################################')