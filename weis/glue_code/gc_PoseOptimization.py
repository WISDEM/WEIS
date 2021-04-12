from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
import numpy as np

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
        
        if self.opt['driver']['optimization']['form'] == 'central':
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
        
        elif self.opt['merit_figure'] == 'Cp':
            wt_opt.model.add_objective('aeroelastic.Cp_out', ref=-1.)
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
        if control_opt['flaps']['te_flap_end']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_end', lower=control_opt['flaps']['te_flap_end']['min_end'],
                                                            upper=control_opt['flaps']['te_flap_end']['max_end'])
        if control_opt['flaps']['te_flap_ext']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_ext', lower=control_opt['flaps']['te_flap_ext']['minimum'],
                                                            upper=control_opt['flaps']['te_flap_ext']['maximum'])
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

        if blade_constr["strains_spar_cap_ss"]["flag"]:
            # Remove generic WISDEM one
            name = 'rs.constr.constr_max_strainU_spar'
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
                name = 'rs.constr.constr_max_strainL_spar'
                if name in wt_opt.model._responses:
                    wt_opt.model._responses.pop( name )
                if name in wt_opt.model._static_responses:
                    wt_opt.model._static_responses.pop( name )
                indices_strains_spar_cap_ps = range(blade_constr["strains_spar_cap_ps"]["index_start"], blade_constr["strains_spar_cap_ps"]["index_end"])
                wt_opt.model.add_constraint("rlds_post.constr.constr_max_strainL_spar", indices = indices_strains_spar_cap_ps, upper=1.0)

        control_constraints = self.opt['constraints']['control']
        if control_constraints['flap_control']['flag']:
            if self.modeling['Level3']['flag'] != True:
                exit('Please turn on the call to OpenFAST if you are trying to optimize trailing edge flaps.')
            wt_opt.model.add_constraint('sse_tune.tune_rosco.flptune_coeff1',
                lower = control_constraints['flap_control']['min'],
                upper = control_constraints['flap_control']['max'])
            wt_opt.model.add_constraint('sse_tune.tune_rosco.flptune_coeff2', 
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


    def set_initial_weis(self, wt_opt):

        if self.modeling["flags"]["blade"]:
            blade_constr = self.opt["constraints"]["blade"]
            wt_opt["rlds_post.constr.max_strainU_spar"] = blade_constr["strains_spar_cap_ss"]["max"]
            wt_opt["rlds_post.constr.max_strainL_spar"] = blade_constr["strains_spar_cap_ps"]["max"]
            wt_opt["stall_check_of.stall_margin"] = blade_constr["stall"]["margin"] * 180.0 / np.pi
            wt_opt["tcons_post.max_allowable_td_ratio"] = blade_constr["tip_deflection"]["margin"]

        return wt_opt