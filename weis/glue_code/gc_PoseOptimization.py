from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
import numpy as np

class PoseOptimizationWEIS(PoseOptimization):

    def __init__(self, wt_init, modeling_options, analysis_options):
        
        # Set solve component for some optimization constraints, and merit figures (RAFT or openfast)
        if modeling_options['Level3']['flag']:
            self.solve_component = 'aeroelastic'
        else:
            self.solve_component = 'raft'

        self.level_flags = np.array([modeling_options[level]['flag'] for level in ['Level1','Level2','Level3']])
        # if sum(self.level_flags) > 1:
            # raise Exception('Only one level in WEIS can be enabled at the same time')

        super(PoseOptimizationWEIS, self).__init__(wt_init, modeling_options, analysis_options)
        
    def get_number_design_variables(self):
        # Determine the number of design variables
        n_DV = super(PoseOptimizationWEIS, self).get_number_design_variables()

        n_add = 0
        if self.opt['design_variables']['control']['servo']['pitch_control']['omega']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['pitch_control']['zeta']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['pitch_control']['Kp_float']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['pitch_control']['ptfm_freq']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['torque_control']['omega']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['torque_control']['zeta']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['flap_control']['flp_kp_norm']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['servo']['flap_control']['flp_tau']['flag']:
            n_add += 1
        if self.opt['design_variables']['control']['flaps']['te_flap_end']['flag']:
            n_add += self.modeling['WISDEM']['RotorSE']['n_te_flaps']
        if self.opt['design_variables']['control']['flaps']['te_flap_ext']['flag']:
            n_add += self.modeling['WISDEM']['RotorSE']['n_te_flaps']
        if self.opt['design_variables']['control']['ps_percent']['flag']:
            n_add += 1
        
        if self.opt['driver']['optimization']['form'] == 'central':
            n_add *= 2

        # TMD DVs
        if self.opt['design_variables']['TMDs']['flag']:
            TMD_opt = self.opt['design_variables']['TMDs']

            # We only support one TMD for now
            for tmd_group in TMD_opt['groups']:
                if 'mass' in tmd_group:
                    n_add += 1
                if 'stiffness' in tmd_group:
                    n_add += 1
                if 'damping' in tmd_group:
                    n_add += 1



        

        return n_DV+n_add


    
    def set_objective(self, wt_opt):
        if self.opt['merit_figure'] == 'blade_tip_deflection':
            wt_opt.model.add_objective('tcons_post.tip_deflection_ratio')
            
        elif self.opt['merit_figure'] == 'DEL_RootMyb':   # for DAC optimization on root-flap-bending moments
            wt_opt.model.add_objective('aeroelastic.DEL_RootMyb', ref = 1.e3)
            
        elif self.opt['merit_figure'] == 'DEL_TwrBsMyt':   # for pitch controller optimization
            wt_opt.model.add_objective('aeroelastic.DEL_TwrBsMyt', ref=1.e4)
            
        elif self.opt['merit_figure'] == 'rotor_overspeed':
            wt_opt.model.add_objective(f'{self.solve_component}.rotor_overspeed')
        
        elif self.opt['merit_figure'] == 'Std_PtfmPitch':
            wt_opt.model.add_objective('aeroelastic.Std_PtfmPitch')
        
        elif self.opt['merit_figure'] == 'Max_PtfmPitch':
            wt_opt.model.add_objective('aeroelastic.Max_PtfmPitch')

        elif self.opt['merit_figure'] == 'Cp':
            wt_opt.model.add_objective('aeroelastic.Cp_out', ref=-1.)
        
        elif self.opt['merit_figure'] == 'weis_lcoe':
            wt_opt.model.add_objective('financese_post.lcoe')
        
        elif self.opt['merit_figure'] == 'OL2CL_pitch':
            wt_opt.model.add_objective('aeroelastic.OL2CL_pitch')
        
        else:
            super(PoseOptimizationWEIS, self).set_objective(wt_opt)
                
        return wt_opt

    
    def set_design_variables(self, wt_opt, wt_init):
        super(PoseOptimizationWEIS, self).set_design_variables(wt_opt, wt_init)

        # -- Control --
        control_opt = self.opt['design_variables']['control']
        if control_opt['servo']['pitch_control']['omega']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.omega_pc', lower=control_opt['servo']['pitch_control']['omega']['min'], 
                                                            upper=control_opt['servo']['pitch_control']['omega']['max'])
        if control_opt['servo']['pitch_control']['zeta']['flag']:                            
            wt_opt.model.add_design_var('tune_rosco_ivc.zeta_pc', lower=control_opt['servo']['pitch_control']['zeta']['min'], 
                                                           upper=control_opt['servo']['pitch_control']['zeta']['max'])
        if control_opt['servo']['torque_control']['omega']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.omega_vs', lower=control_opt['servo']['torque_control']['omega']['min'], 
                                                            upper=control_opt['servo']['torque_control']['omega']['max'])
        if control_opt['servo']['torque_control']['zeta']['flag']:                                                    
            wt_opt.model.add_design_var('tune_rosco_ivc.zeta_vs', lower=control_opt['servo']['torque_control']['zeta']['min'], 
                                                           upper=control_opt['servo']['torque_control']['zeta_max'])
        if control_opt['servo']['ipc_control']['Kp']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.IPC_Kp1p', lower=control_opt['servo']['ipc_control']['Kp']['min'],
                                                            upper=control_opt['servo']['ipc_control']['Kp']['max'],
                                                            ref=control_opt['servo']['ipc_control']['Kp']['ref'])
        if control_opt['servo']['ipc_control']['Ki']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.IPC_Ki1p', lower=control_opt['servo']['ipc_control']['Ki']['min'],
                                                            upper=control_opt['servo']['ipc_control']['Ki']['max'],
                                                            ref=control_opt['servo']['ipc_control']['Kp']['ref'])
        if control_opt['servo']['pitch_control']['stability_margin']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.stability_margin', lower=control_opt['servo']['pitch_control']['stability_margin']['min'],
                                                            upper=control_opt['servo']['pitch_control']['stability_margin']['max'])
        if control_opt['flaps']['te_flap_end']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_end', lower=control_opt['flaps']['te_flap_end']['min'],
                                                            upper=control_opt['flaps']['te_flap_end']['max'])
        if control_opt['flaps']['te_flap_ext']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_ext', lower=control_opt['flaps']['te_flap_ext']['min'],
                                                            upper=control_opt['flaps']['te_flap_ext']['max'])
        if 'flap_control' in control_opt['servo']:
            if control_opt['servo']['flap_control']['flp_kp_norm']['flag']:
                wt_opt.model.add_design_var('tune_rosco_ivc.flp_kp_norm', 
                                    lower=control_opt['servo']['flap_control']['flp_kp_norm']['min'], 
                                    upper=control_opt['servo']['flap_control']['flp_kp_norm']['max'])
            if control_opt['servo']['flap_control']['flp_tau']['flag']:
                wt_opt.model.add_design_var('tune_rosco_ivc.flp_tau', 
                                    lower=control_opt['servo']['flap_control']['flp_tau']['min'], 
                                    upper=control_opt['servo']['flap_control']['flp_tau']['max'])

        if control_opt['ps_percent']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.ps_percent', lower=control_opt['ps_percent']['lower_bound'],
                                                            upper=control_opt['ps_percent']['upper_bound'])

        if control_opt['servo']['pitch_control']['Kp_float']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.Kp_float', lower=control_opt['servo']['pitch_control']['Kp_float']['min'], 
                                                           upper=control_opt['servo']['pitch_control']['Kp_float']['max'])

        if control_opt['servo']['pitch_control']['ptfm_freq']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.ptfm_freq', lower=control_opt['servo']['pitch_control']['ptfm_freq']['min'], 
                                                           upper=control_opt['servo']['pitch_control']['ptfm_freq']['max'])

        if self.opt['design_variables']['TMDs']['flag']:
            TMD_opt = self.opt['design_variables']['TMDs']

            # We only support one TMD for now
            for i_group, tmd_group in enumerate(TMD_opt['groups']):
                if 'mass' in tmd_group:
                    wt_opt.model.add_design_var(
                        f'TMDs.TMD_IVCs.group_{i_group}_mass', 
                        lower=tmd_group['mass']['lower_bound'],
                        upper=tmd_group['mass']['upper_bound'],
                        )
                if 'stiffness' in tmd_group:
                    wt_opt.model.add_design_var(
                        f'TMDs.TMD_IVCs.group_{i_group}_stiffness', 
                        lower=tmd_group['stiffness']['lower_bound'],
                        upper=tmd_group['stiffness']['upper_bound']
                        )
                    if 'natural_frequency' in tmd_group:
                        raise Exception("natural_frequency and stiffness can not be design variables in the same group")
                if 'damping' in tmd_group:
                    wt_opt.model.add_design_var(
                        f'TMDs.TMD_IVCs.group_{i_group}_damping', 
                        lower=tmd_group['damping']['lower_bound'],
                        upper=tmd_group['damping']['upper_bound']
                        )
                    if 'damping_ratio' in tmd_group:
                        raise Exception("damping_ratio and damping can not be design variables in the same group")
                if 'natural_frequency' in tmd_group:
                    wt_opt.model.add_design_var(
                        f'TMDs.TMD_IVCs.group_{i_group}_natural_frequency', 
                        lower=tmd_group['natural_frequency']['lower_bound'],
                        upper=tmd_group['natural_frequency']['upper_bound']
                        )
                if 'damping_ratio' in tmd_group:
                    wt_opt.model.add_design_var(
                        f'TMDs.TMD_IVCs.group_{i_group}_damping_ratio', 
                        lower=tmd_group['damping_ratio']['lower_bound'],
                        upper=tmd_group['damping_ratio']['upper_bound']
                        )
        
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

        if blade_constr["strains_spar_cap_ss"]["flag"]:
            # Remove generic WISDEM one
            name = 'rotorse.rs.constr.constr_max_strainU_spar'
            if name in wt_opt.model._responses:
                wt_opt.model._responses.pop( name )
            if name in wt_opt.model._static_responses:
                wt_opt.model._static_responses.pop( name )
            if blade_opt["structure"]["spar_cap_ss"]["flag"]:
                indices_strains_spar_cap_ss = range(blade_constr["strains_spar_cap_ss"]["index_start"], blade_constr["strains_spar_cap_ss"]["index_end"])
                wt_opt.model.add_constraint("rlds_post.constr.constr_max_strainU_spar", indices = indices_strains_spar_cap_ss, upper=1.0)

        if blade_constr["strains_spar_cap_ps"]["flag"]:
            if (
                blade_opt["structure"]["spar_cap_ps"]["flag"]
                or blade_opt["structure"]["spar_cap_ps"]["equal_to_suction"]
            ):
                # Remove generic WISDEM one
                name = 'rotorse.rs.constr.constr_max_strainL_spar'
                if name in wt_opt.model._responses:
                    wt_opt.model._responses.pop( name )
                if name in wt_opt.model._static_responses:
                    wt_opt.model._static_responses.pop( name )
                indices_strains_spar_cap_ps = range(blade_constr["strains_spar_cap_ps"]["index_start"], blade_constr["strains_spar_cap_ps"]["index_end"])
                wt_opt.model.add_constraint("rlds_post.constr.constr_max_strainL_spar", indices = indices_strains_spar_cap_ps, upper=1.0)

        ### CONTROL CONSTRAINTS
        control_constraints = self.opt['constraints']['control']
        
        # Flap control
        if control_constraints['flap_control']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize trailing edge flaps.')
            wt_opt.model.add_constraint('sse_tune.tune_rosco.flptune_coeff1',
                lower = control_constraints['flap_control']['min'],
                upper = control_constraints['flap_control']['max'])
            wt_opt.model.add_constraint('sse_tune.tune_rosco.flptune_coeff2', 
                lower = control_constraints['flap_control']['min'],
                upper = control_constraints['flap_control']['max'])    
        
        # Rotor overspeed
        if control_constraints['rotor_overspeed']['flag']:
            if not any(self.level_flags):
                raise Exception('Please turn on the call to OpenFAST or RAFT if you are trying to optimize rotor overspeed constraints.')
            wt_opt.model.add_constraint(f'{self.solve_component}.rotor_overspeed',
                lower = control_constraints['rotor_overspeed']['min'],
                upper = control_constraints['rotor_overspeed']['max'])
        
        # Add PI gains if overspeed is merit_figure or constraint
        if control_constraints['rotor_overspeed']['flag'] or self.opt['merit_figure'] == 'rotor_overspeed':
            wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Kp',
                upper = 0.0)
            wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Ki', 
                upper = 0.0)  
        
        # Nacelle Accelleration magnitude
        if control_constraints['nacelle_acceleration']['flag']:
            wt_opt.model.add_constraint('aeroelastic.max_nac_accel',
                    upper = control_constraints['nacelle_acceleration']['max'])
        
        # Max platform pitch
        if control_constraints['Max_PtfmPitch']['flag']:
            if not any(self.level_flags):
                raise Exception('Please turn on the call to OpenFAST or RAFT if you are trying to optimize Max_PtfmPitch constraints.')
            wt_opt.model.add_constraint(f'{self.solve_component}.Max_PtfmPitch',
                upper = control_constraints['Max_PtfmPitch']['max'])
        
        # Platform pitch motion
        if control_constraints['Std_PtfmPitch']['flag']:
            if not any(self.level_flags):
                raise Exception('Please turn on the call to OpenFAST or RAFT if you are trying to optimize Std_PtfmPitch constraints.')
            wt_opt.model.add_constraint(f'{self.solve_component}.Std_PtfmPitch',
                upper = control_constraints['Std_PtfmPitch']['max'])
        if control_constraints['Max_TwrBsMyt']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize Max_TwrBsMyt constraints.')
            wt_opt.model.add_constraint('aeroelastic.max_TwrBsMyt_ratio', 
                upper = 1.0)
        if control_constraints['DEL_TwrBsMyt']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize Max_TwrBsMyt constraints.')
            wt_opt.model.add_constraint('aeroelastic.DEL_TwrBsMyt_ratio', 
                upper = 1.0)

        # Blade pitch travel
        if control_constraints['avg_pitch_travel']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize avg_pitch_travel constraints.')
            wt_opt.model.add_constraint('aeroelastic.avg_pitch_travel',
                upper = control_constraints['avg_pitch_travel']['max'])

        # Blade pitch duty cycle (number of direction changes)
        if control_constraints['pitch_duty_cycle']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize pitch_duty_cycle constraints.')
            wt_opt.model.add_constraint('aeroelastic.pitch_duty_cycle',
                upper = control_constraints['pitch_duty_cycle']['max'])

        # OpenFAST failure
        if self.opt['constraints']['openfast_failed']['flag']:
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize with openfast_failed constraint.')
            wt_opt.model.add_constraint('aeroelastic.openfast_failed',upper = 1.)

        # Max offset
        if self.opt['constraints']['Max_Offset']['flag']:
            if not any(self.level_flags):
                raise Exception('Please turn on the call to OpenFAST or RAFT if you are trying to optimize with openfast_failed constraint.')
            wt_opt.model.add_constraint(
                f'{self.solve_component}.Max_Offset',
                upper = self.opt['constraints']['Max_Offset']['max']
                )
                
        # Tower constraints
        tower_opt = self.opt["design_variables"]["tower"]
        tower_constr = self.opt["constraints"]["tower"]
        if tower_constr["global_buckling"]["flag"] and self.modeling['Level3']['flag']:
            # Remove generic WISDEM one
            name = 'towerse.post.constr_global_buckling'
            if name in wt_opt.model._responses:
                wt_opt.model._responses.pop( name )
            if name in wt_opt.model._static_responses:
                wt_opt.model._static_responses.pop( name )
                
            wt_opt.model.add_constraint("towerse_post.constr_global_buckling", upper=1.0)
        
        if tower_constr["shell_buckling"]["flag"] and self.modeling['Level3']['flag']:
            # Remove generic WISDEM one
            name = 'towerse.post.constr_shell_buckling'
            if name in wt_opt.model._responses:
                wt_opt.model._responses.pop( name )
            if name in wt_opt.model._static_responses:
                wt_opt.model._static_responses.pop( name )
                
            wt_opt.model.add_constraint("towerse_post.constr_shell_buckling", upper=1.0)
        
        if tower_constr["stress"]["flag"] and self.modeling['Level3']['flag']:
            # Remove generic WISDEM one
            name = 'towerse.post.constr_stress'
            if name in wt_opt.model._responses:
                wt_opt.model._responses.pop( name )
            if name in wt_opt.model._static_responses:
                wt_opt.model._static_responses.pop( name )
                
            wt_opt.model.add_constraint("towerse_post.constr_stress", upper=1.0)

        # Damage constraints
        damage_constraints = self.opt['constraints']['damage']
        if damage_constraints['tower_base']['flag'] and (self.modeling['Level2']['flag'] or self.modeling['Level3']['flag']):
            if self.modeling['Level3']['flag'] != True:
                raise Exception('Please turn on the call to OpenFAST if you are trying to optimize with tower_base damage constraint.')

            tower_base_damage_max = damage_constraints['tower_base']['max']
            if damage_constraints['tower_base']['log']:
                tower_base_damage_max = np.log(tower_base_damage_max)

            wt_opt.model.add_constraint('aeroelastic.damage_tower_base',upper = tower_base_damage_max)

        return wt_opt


    def set_initial_weis(self, wt_opt):

        if self.modeling["flags"]["blade"]:
            blade_constr = self.opt["constraints"]["blade"]
            wt_opt["rlds_post.constr.max_strainU_spar"] = blade_constr["strains_spar_cap_ss"]["max"]
            wt_opt["rlds_post.constr.max_strainL_spar"] = blade_constr["strains_spar_cap_ps"]["max"]
            wt_opt["stall_check_of.stall_margin"] = blade_constr["stall"]["margin"] * 180.0 / np.pi
            wt_opt["tcons_post.max_allowable_td_ratio"] = blade_constr["tip_deflection"]["margin"]

        return wt_opt
