from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

class PoseOptimizationWEIS(PoseOptimization):
        
    def get_number_design_variables(self):
        # Determine the number of design variables
        n_DV = super(PoseOptimizationWEIS, self).get_number_design_variables()

        n_add = 0
        if self.opt['design_variables']['control']['servo']['pitch_control']['flag']:
            n_add += 2
        if self.opt['design_variables']['control']['servo']['torque_control']['flag']:
            n_add += 2
        if self.opt['design_variables']['control']['servo']['flap_control']['flag']:
            n_add += 2
        if self.opt['design_variables']['control']['flaps']['te_flap_end']['flag']:
            n_add += self.modeling['WISDEM']['RotorSE']['n_te_flaps']
        if self.opt['design_variables']['control']['flaps']['te_flap_ext']['flag']:
            n_add += self.modeling['WISDEM']['RotorSE']['n_te_flaps']
        
        if self.opt['driver']['form'] == 'central':
            n_add *= 2

        return n_DV+n_add


    
    def set_objective(self, wt_opt):
        if self.opt['merit_figure'] == 'blade_tip_deflection':
            wt_opt.model.add_objective('tcons_post.tip_deflection_ratio')
            
        elif self.opt['merit_figure'] == 'My_std':   # for DAC optimization on root-flap-bending moments
            wt_opt.model.add_objective('aeroelastic.My_std', ref = 1.e6)
            
        elif self.opt['merit_figure'] == 'DEL_RootMyb':   # for DAC optimization on root-flap-bending moments
            wt_opt.model.add_objective('aeroelastic.DEL_RootMyb', ref = 1.e3)
            
        elif self.opt['merit_figure'] == 'DEL_TwrBsMyt':   # for pitch controller optimization
            wt_opt.model.add_objective('aeroelastic.DEL_TwrBsMyt', ref=1.e4)
            
        elif self.opt['merit_figure'] == 'flp1_std':   # for DAC optimization on flap angles - TORQUE 2020 paper (need to define time constant in ROSCO)
            wt_opt.model.add_objective('aeroelastic.flp1_std')  #1.e-8)
            
        elif self.opt['merit_figure'] == 'rotor_overspeed':
            wt_opt.model.add_objective('aeroelastic.rotor_overspeed')
            
        else:
            super(PoseOptimizationWEIS, self).set_objective(wt_opt)
                
        return wt_opt

    
    def set_design_variables(self, wt_opt, wt_init):
        super(PoseOptimizationWEIS, self).set_design_variables(wt_opt, wt_init)

        # -- Control --
        control_opt = self.opt['design_variables']['control']
        if control_opt['servo']['pitch_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.PC_omega', lower=control_opt['servo']['pitch_control']['omega_min'], 
                                                            upper=control_opt['servo']['pitch_control']['omega_max'])
            wt_opt.model.add_design_var('tune_rosco_ivc.PC_zeta', lower=control_opt['servo']['pitch_control']['zeta_min'], 
                                                           upper=control_opt['servo']['pitch_control']['zeta_max'])
        if control_opt['servo']['torque_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.VS_omega', lower=control_opt['servo']['torque_control']['omega_min'], 
                                                            upper=control_opt['servo']['torque_control']['omega_max'])
            wt_opt.model.add_design_var('tune_rosco_ivc.VS_zeta', lower=control_opt['servo']['torque_control']['zeta_min'], 
                                                           upper=control_opt['servo']['torque_control']['zeta_max'])
        if control_opt['servo']['ipc_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.IPC_Ki1p', lower=control_opt['servo']['ipc_control']['Ki_min'],
                                                            upper=control_opt['servo']['ipc_control']['Ki_max'],
                                                            ref=1.e-7)
        if 'flap_control' in control_opt['servo']:
            if control_opt['servo']['flap_control']['flag']:
                wt_opt.model.add_design_var('tune_rosco_ivc.Flp_omega', 
                                    lower=control_opt['servo']['flap_control']['omega_min'], 
                                    upper=control_opt['servo']['flap_control']['omega_max'])
                wt_opt.model.add_design_var('tune_rosco_ivc.Flp_zeta', 
                                    lower=control_opt['servo']['flap_control']['zeta_min'], 
                                    upper=control_opt['servo']['flap_control']['zeta_max'])
        
        return wt_opt

    
    def set_constraints(self, wt_opt):
        super(PoseOptimizationWEIS, self).set_constraints(wt_opt)

        blade_opt = self.opt["design_variables"]["blade"]
        blade_constr = self.opt["constraints"]["blade"]
        if blade_constr['tip_deflection']['flag']:
            # Remove generic WISDEM one
            name = 'tcons.tip_deflection_ratio'
            if name in wt_opt.model._responses:
                wt_opt.model._responses.pop( name )
            if name in wt_opt.model._static_responses:
                wt_opt.model._static_responses.pop( name )
                
            if blade_opt['structure']['spar_cap_ss']['flag'] or blade_opt['structure']['spar_cap_ps']['flag']:
                wt_opt.model.add_constraint('tcons_post.tip_deflection_ratio', upper=1.0)
            else:
                print('WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        control_constraints = self.opt['constraints']['control']
        if control_constraints['flap_control']['flag']:
            if self.modeling['Level3']['flag'] != True:
                exit('Please turn on the call to OpenFAST if you are trying to optimize trailing edge flaps.')
            wt_opt.model.add_constraint('sse_tune.tune_rosco.Flp_Kp',
                lower = control_constraints['flap_control']['min'],
                upper = control_constraints['flap_control']['max'])
            wt_opt.model.add_constraint('sse_tune.tune_rosco.Flp_Ki', 
                lower = control_constraints['flap_control']['min'],
                upper = control_constraints['flap_control']['max'])    
        if control_constraints['rotor_overspeed']['flag']:
            if self.modeling['Level3']['flag'] != True:
                exit('Please turn on the call to OpenFAST if you are trying to optimize rotor overspeed constraints.')
            wt_opt.model.add_constraint('aeroelastic.rotor_overspeed',
                lower = control_constraints['rotor_overspeed']['min'],
                upper = control_constraints['rotor_overspeed']['max'])
        if control_constraints['rotor_overspeed']['flag'] or self.opt['merit_figure'] == 'rotor_overspeed':
            wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Kp',
                upper = 0.0)
            wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Ki', 
                upper = 0.0)    

        return wt_opt
