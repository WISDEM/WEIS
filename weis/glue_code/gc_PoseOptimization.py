import numpy as np
import openmdao.api as om
import os

class PoseOptimization(object):
    def __init__(self, modeling_options, analysis_options):
        self.modeling    = modeling_options
        self.opt         = analysis_options
        self.blade_opt   = self.opt['design_variables']['blade']
        self.tower_opt   = self.opt['design_variables']['tower']
        self.control_opt = self.opt['design_variables']['control']

        
    def get_number_design_variables(self):
        # Determine the number of design variables
        n_DV = 0
        if self.blade_opt['aero_shape']['twist']['flag']:
            n_DV += self.blade_opt['aero_shape']['twist']['n_opt'] - 2
        if self.blade_opt['aero_shape']['chord']['flag']:    
            n_DV += self.blade_opt['aero_shape']['chord']['n_opt'] - 3            
        if self.blade_opt['aero_shape']['af_positions']['flag']:
            n_DV += self.modeling['blade']['n_af_span'] - self.blade_opt['aero_shape']['af_positions']['af_start'] - 1
        if self.blade_opt['structure']['spar_cap_ss']['flag']:
            n_DV += self.blade_opt['structure']['spar_cap_ss']['n_opt'] - 2
        if self.blade_opt['structure']['spar_cap_ps']['flag'] and not self.blade_opt['structure']['spar_cap_ps']['equal_to_suction']:
            n_DV += self.blade_opt['structure']['spar_cap_ps']['n_opt'] - 2
        if self.opt['design_variables']['control']['tsr']['flag']:
            n_DV += 1
        if self.opt['design_variables']['control']['servo']['pitch_control']['flag']:
            n_DV += 2
        if self.opt['design_variables']['control']['servo']['torque_control']['flag']:
            n_DV += 2
        if self.opt['design_variables']['control']['servo']['flap_control']['flag']:
            n_DV += 2
        if self.opt['design_variables']['control']['flaps']['te_flap_end']['flag']:
            n_DV += self.modeling['blade']['n_te_flaps']
        if self.opt['design_variables']['control']['flaps']['te_flap_ext']['flag']:
            n_DV += self.modeling['blade']['n_te_flaps']
        if self.tower_opt['outer_diameter']['flag']:
            n_DV += self.modeling['tower']['n_height']
        if self.tower_opt['layer_thickness']['flag']:
            n_DV += (self.modeling['tower']['n_height'] - 1) * self.modeling['tower']['n_layers']
        
        if self.opt['driver']['form'] == 'central':
            n_DV *= 2

        return n_DV

    
    def _get_step_size(self):
        # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
        return (1.e-6 if not 'step_size' in self.opt['driver'] else self.opt['driver']['step_size'])

    
    def set_driver(self, wt_opt):
        folder_output = self.opt['general']['folder_output']

        step_size = self._get_step_size()

        # Solver has specific meaning in OpenMDAO
        wt_opt.model.approx_totals(method='fd', step=step_size, form=self.opt['driver']['form'])

        # Set optimization solver and options. First, Scipy's SLSQP
        if self.opt['driver']['solver'] == 'SLSQP':
            wt_opt.driver  = om.ScipyOptimizeDriver()
            wt_opt.driver.options['optimizer'] = self.opt['driver']['solver']
            wt_opt.driver.options['tol']       = self.opt['driver']['tol']
            wt_opt.driver.options['maxiter']   = self.opt['driver']['max_iter']

        # The next two optimization methods require pyOptSparse.
        elif self.opt['driver']['solver'] == 'CONMIN':
            try:
                from openmdao.api import pyOptSparseDriver
            except:
                exit('You requested the optimization solver CONMIN, but you have not installed the pyOptSparseDriver. Please do so and rerun.')
            wt_opt.driver = pyOptSparseDriver()
            wt_opt.driver.options['optimizer'] = self.opt['driver']['solver']
            wt_opt.driver.opt_settings['ITMAX']= self.opt['driver']['max_iter']

        elif self.opt['driver']['solver'] == 'SNOPT':
            try:
                from openmdao.api import pyOptSparseDriver
            except:
                exit('You requested the optimization solver SNOPT which requires pyOptSparse to be installed, but it cannot be found. Please install pyOptSparse and rerun.')
            wt_opt.driver = pyOptSparseDriver()
            try:    
                wt_opt.driver.options['optimizer']                       = self.opt['driver']['solver']
            except:
                exit('You requested the optimization solver SNOPT, but you have not installed it within the pyOptSparseDriver. Please do so and rerun.')
            wt_opt.driver.opt_settings['Major optimality tolerance']  = float(self.opt['driver']['tol'])
            wt_opt.driver.opt_settings['Major iterations limit']      = int(self.opt['driver']['max_major_iter'])
            wt_opt.driver.opt_settings['Iterations limit']            = int(self.opt['driver']['max_minor_iter'])
            wt_opt.driver.opt_settings['Major feasibility tolerance'] = float(self.opt['driver']['tol'])
            wt_opt.driver.opt_settings['Summary file']                = os.path.join(folder_output, 'SNOPT_Summary_file.txt')
            wt_opt.driver.opt_settings['Print file']                  = os.path.join(folder_output, 'SNOPT_Print_file.txt')
            if 'hist_file_name' in self.opt['driver']:
                wt_opt.driver.hist_file = self.opt['driver']['hist_file_name']
            if 'verify_level' in self.opt['driver']:
                wt_opt.driver.opt_settings['Verify level'] = self.opt['driver']['verify_level']
            # wt_opt.driver.declare_coloring()  
            if 'hotstart_file' in self.opt['driver']:
                wt_opt.driver.hotstart_file = self.opt['driver']['hotstart_file']

        else:
            exit('The optimizer ' + self.opt['driver']['solver'] + 'is not yet supported!')
                
        return wt_opt

    
    def set_objective(self, wt_opt):

        # Set merit figure. Each objective has its own scaling.
        if self.opt['merit_figure'] == 'AEP':
            wt_opt.model.add_objective('rp.AEP', ref = -1.e6)
        elif self.opt['merit_figure'] == 'blade_mass':
            wt_opt.model.add_objective('re.precomp.blade_mass', ref = 1.e4)
        elif self.opt['merit_figure'] == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', ref = 0.1)
        elif self.opt['merit_figure'] == 'blade_tip_deflection':
            wt_opt.model.add_objective('tcons_post.tip_deflection_ratio')
        elif self.opt['merit_figure'] == 'tower_mass':
            wt_opt.model.add_objective('towerse.tower_mass')
        elif self.opt['merit_figure'] == 'tower_cost':
            wt_opt.model.add_objective('tcc.tower_cost')
        elif self.opt['merit_figure'] == 'Cp':
            if self.modeling['WISDEM']['RotorSE']:
                wt_opt.model.add_objective('rp.powercurve.Cp_regII', ref = -1.)
            else:
                wt_opt.model.add_objective('ccblade.CP', ref = -1.)
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
            exit('The merit figure ' + self.opt['merit_figure'] + ' is not supported.')
                
        return wt_opt

    
    def set_design_variables(self, wt_opt, wt_init):

        # Set optimization design variables.

        if self.blade_opt['aero_shape']['twist']['flag']:
            indices        = range(2, self.blade_opt['aero_shape']['twist']['n_opt'])
            wt_opt.model.add_design_var('blade.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)

        chord_options = self.blade_opt['aero_shape']['chord']
        if chord_options['flag']:
            indices  = range(3, chord_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.chord_opt_gain', indices = indices, lower=chord_options['min_gain'], upper=chord_options['max_gain'])

        if self.blade_opt['aero_shape']['af_positions']['flag']:
            n_af = self.modeling['blade']['n_af_span']
            indices  = range(self.blade_opt['aero_shape']['af_positions']['af_start'],n_af - 1)
            af_pos_init = wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['grid']
            step_size   = self._get_step_size()
            lb_af    = np.zeros(n_af)
            ub_af    = np.zeros(n_af)
            for i in range(1,indices[0]):
                lb_af[i]    = ub_af[i] = af_pos_init[i]
            for i in indices:
                lb_af[i]    = 0.5*(af_pos_init[i-1] + af_pos_init[i]) + step_size
                ub_af[i]    = 0.5*(af_pos_init[i+1] + af_pos_init[i]) - step_size
            lb_af[-1] = ub_af[-1] = 1.
            wt_opt.model.add_design_var('blade.opt_var.af_position', indices = indices, lower=lb_af[indices], upper=ub_af[indices])

        spar_cap_ss_options = self.blade_opt['structure']['spar_cap_ss']
        if spar_cap_ss_options['flag']:
            indices  = range(1,spar_cap_ss_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ss_opt_gain', indices = indices, lower=spar_cap_ss_options['min_gain'], upper=spar_cap_ss_options['max_gain'])

        # Only add the pressure side design variables if we do set
        # `equal_to_suction` as False in the optimization yaml.
        spar_cap_ps_options = self.blade_opt['structure']['spar_cap_ps']
        if spar_cap_ps_options['flag'] and not spar_cap_ps_options['equal_to_suction']:
            indices  = range(1, spar_cap_ps_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ps_opt_gain', indices = indices, lower=spar_cap_ps_options['min_gain'], upper=spar_cap_ps_options['max_gain'])

        if self.opt['design_variables']['control']['flaps']['te_flap_end']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_end', 
            lower=self.opt['design_variables']['control']['flaps']['te_flap_end']['minimum'], 
            upper=self.opt['design_variables']['control']['flaps']['maximum'], 
            ref=1e2)
        if self.opt['design_variables']['control']['flaps']['te_flap_ext']['flag']:
            wt_opt.model.add_design_var('dac_ivc.te_flap_ext', 
                                lower=self.opt['design_variables']['control']['flaps']['te_flap_ext']['min_ext'], 
                                upper=self.opt['design_variables']['control']['flaps']['te_flap_ext']['max_ext'],
                                ref=1e2)

        if self.tower_opt['outer_diameter']['flag']:
            wt_opt.model.add_design_var('tower.diameter', lower=self.tower_opt['outer_diameter']['lower_bound'], upper=self.tower_opt['outer_diameter']['upper_bound'], ref=5.)

        if self.tower_opt['layer_thickness']['flag']:
            wt_opt.model.add_design_var('tower.layer_thickness', lower=self.tower_opt['layer_thickness']['lower_bound'], upper=self.tower_opt['layer_thickness']['upper_bound'], ref=1e-2)

        # -- Control -- 
        if self.control_opt['tsr']['flag']:
            wt_opt.model.add_design_var('opt_var.tsr_opt_gain', lower=self.control_opt['tsr']['min_gain'], 
                                                                upper=self.control_opt['tsr']['max_gain'])
        if self.control_opt['servo']['pitch_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.PC_omega', lower=self.control_opt['servo']['pitch_control']['omega_min'], 
                                                            upper=self.control_opt['servo']['pitch_control']['omega_max'])
            wt_opt.model.add_design_var('tune_rosco_ivc.PC_zeta', lower=self.control_opt['servo']['pitch_control']['zeta_min'], 
                                                           upper=self.control_opt['servo']['pitch_control']['zeta_max'])
        if self.control_opt['servo']['torque_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.VS_omega', lower=self.control_opt['servo']['torque_control']['omega_min'], 
                                                            upper=self.control_opt['servo']['torque_control']['omega_max'])
            wt_opt.model.add_design_var('tune_rosco_ivc.VS_zeta', lower=self.control_opt['servo']['torque_control']['zeta_min'], 
                                                           upper=self.control_opt['servo']['torque_control']['zeta_max'])
        if self.control_opt['servo']['ipc_control']['flag']:
            wt_opt.model.add_design_var('tune_rosco_ivc.IPC_Ki1p', lower=self.control_opt['servo']['ipc_control']['Ki_min'],
                                                            upper=self.control_opt['servo']['ipc_control']['Ki_max'],
                                                            ref=1.e-7)
        if 'flap_control' in self.control_opt['servo']:
            if self.control_opt['servo']['flap_control']['flag']:
                wt_opt.model.add_design_var('tune_rosco_ivc.Flp_omega', 
                                    lower=self.control_opt['servo']['flap_control']['omega_min'], 
                                    upper=self.control_opt['servo']['flap_control']['omega_max'])
                wt_opt.model.add_design_var('tune_rosco_ivc.Flp_zeta', 
                                    lower=self.control_opt['servo']['flap_control']['zeta_min'], 
                                    upper=self.control_opt['servo']['flap_control']['zeta_max'])
        
        return wt_opt

    
    def set_constraints(self, wt_opt):

        # Set non-linear constraints
        blade_constraints = self.opt['constraints']['blade']
        if blade_constraints['strains_spar_cap_ss']['flag']:
            if self.blade_opt['structure']['spar_cap_ss']['flag']:
                wt_opt.model.add_constraint('rs.constr.constr_max_strainU_spar', upper= 1.0)
            else:
                print('WARNING: the strains of the suction-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['strains_spar_cap_ps']['flag']:
            if self.blade_opt['structure']['spar_cap_ps']['flag'] or self.blade_opt['structure']['spar_cap_ps']['equal_to_suction']:
                wt_opt.model.add_constraint('rs.constr.constr_max_strainL_spar', upper= 1.0)
            else:
                print('WARNING: the strains of the pressure-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['stall']['flag']:
            if self.blade_opt['aero_shape']['twist']['flag']:
                wt_opt.model.add_constraint('stall_check.no_stall_constraint', upper= 1.0) 
            else:
                print('WARNING: the margin to stall is set to be constrained, but twist is not an active design variable. The constraint is not enforced.')

        if blade_constraints['tip_deflection']['flag']:
            if self.blade_opt['structure']['spar_cap_ss']['flag'] or self.blade_opt['structure']['spar_cap_ps']['flag']:
                wt_opt.model.add_constraint('tcons_post.tip_deflection_ratio', upper=1.)
            else:
                print('WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['chord']['flag']:
            if self.blade_opt['aero_shape']['chord']['flag']:
                wt_opt.model.add_constraint('blade.pa.max_chord_constr', upper= 1.0)
            else:
                print('WARNING: the max chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced.')

        if blade_constraints['frequency']['flap_above_3P']:
            if self.blade_opt['structure']['spar_cap_ss']['flag'] or self.blade_opt['structure']['spar_cap_ps']['flag']:
                wt_opt.model.add_constraint('rs.constr.constr_flap_f_margin', upper= 0.0)
            else:
                print('WARNING: the blade flap frequencies are set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['frequency']['edge_above_3P']:
            wt_opt.model.add_constraint('rs.constr.constr_edge_f_margin', upper= 0.0)

        # if blade_constraints['frequency']['flap_below_3P']:
        #     wt_opt.model.add_constraint('rs.constr.constr_flap_f_below_3P', upper= 1.0)

        # if blade_constraints['frequency']['edge_below_3P']:
        #     wt_opt.model.add_constraint('rs.constr.constr_edge_f_below_3P', upper= 1.0)

        # if blade_constraints['frequency']['flap_above_3P'] and blade_constraints['frequency']['flap_below_3P']:
        #     exit('The blade flap frequency is constrained to be both above and below 3P. Please check the constraint flags.')

        # if blade_constraints['frequency']['edge_above_3P'] and blade_constraints['frequency']['edge_below_3P']:
        #     exit('The blade edge frequency is constrained to be both above and below 3P. Please check the constraint flags.')

        if blade_constraints['rail_transport']['flag']:
            if blade_constraints['rail_transport']['8_axle']:
                wt_opt.model.add_constraint('re.rail.constr_LV_8axle_horiz',   lower = 0.8, upper= 1.0)
                wt_opt.model.add_constraint('re.rail.constr_strainPS',         upper= 1.0)
                wt_opt.model.add_constraint('re.rail.constr_strainSS',         upper= 1.0)
            elif blade_constraints['rail_transport']['4_axle']:
                wt_opt.model.add_constraint('re.rail.constr_LV_4axle_horiz', upper= 1.0)
            else:
                exit('You have activated the rail transport constraint module. Please define whether you want to model 4- or 8-axle flatcars.')

        if self.opt['constraints']['blade']['moment_coefficient']['flag']:
            wt_opt.model.add_constraint('ccblade.CM', lower= self.opt['constraints']['blade']['moment_coefficient']['min'], upper= self.opt['constraints']['blade']['moment_coefficient']['max'])
        if self.opt['constraints']['blade']['match_cl_cd']['flag_cl'] or self.opt['constraints']['blade']['match_cl_cd']['flag_cd']:
            data_target = np.loadtxt(self.opt['constraints']['blade']['match_cl_cd']['filename'])
            eta_opt     = np.linspace(0., 1., self.opt['design_variables']['blade']['aero_shape']['twist']['n_opt'])
            target_cl   = np.interp(eta_opt, data_target[:,0], data_target[:,3])
            target_cd   = np.interp(eta_opt, data_target[:,0], data_target[:,4])
            eps_cl = 1.e-2
            if self.opt['constraints']['blade']['match_cl_cd']['flag_cl']:
                wt_opt.model.add_constraint('ccblade.cl_n_opt', lower = target_cl-eps_cl, upper = target_cl+eps_cl)
            if self.opt['constraints']['blade']['match_cl_cd']['flag_cd']:
                wt_opt.model.add_constraint('ccblade.cd_n_opt', lower = target_cd-eps_cl, upper = target_cd+eps_cl)
        if self.opt['constraints']['blade']['match_L_D']['flag_L'] or self.opt['constraints']['blade']['match_L_D']['flag_D']:
            data_target = np.loadtxt(self.opt['constraints']['blade']['match_L_D']['filename'])
            eta_opt     = np.linspace(0., 1., self.opt['design_variables']['blade']['aero_shape']['twist']['n_opt'])
            target_L   = np.interp(eta_opt, data_target[:,0], data_target[:,7])
            target_D   = np.interp(eta_opt, data_target[:,0], data_target[:,8])
        eps_L  = 1.e+2
        if self.opt['constraints']['blade']['match_L_D']['flag_L']:
            wt_opt.model.add_constraint('ccblade.L_n_opt', lower = target_L-eps_L, upper = target_L+eps_L)
        if self.opt['constraints']['blade']['match_L_D']['flag_D']:
            wt_opt.model.add_constraint('ccblade.D_n_opt', lower = target_D-eps_L, upper = target_D+eps_L)

        tower_constraints = self.opt['constraints']['tower']
        if tower_constraints['height_constraint']['flag']:
            wt_opt.model.add_constraint('towerse.height_constraint',
                lower=tower_constraints['height_constraint']['lower_bound'],
                upper=tower_constraints['height_constraint']['upper_bound'])

        if tower_constraints['stress']['flag']:
            wt_opt.model.add_constraint('towerse.post.stress', upper=1.0)

        if tower_constraints['global_buckling']['flag']:
            wt_opt.model.add_constraint('towerse.post.global_buckling', upper=1.0)

        if tower_constraints['shell_buckling']['flag']:
            wt_opt.model.add_constraint('towerse.post.shell_buckling', upper=1.0)

        if tower_constraints['d_to_t']['flag']:
            wt_opt.model.add_constraint('towerse.constr_d_to_t', upper=0.0)

        if tower_constraints['taper']['flag']:
            wt_opt.model.add_constraint('towerse.constr_taper', lower=0.0)

        if tower_constraints['slope']['flag']:
            wt_opt.model.add_constraint('towerse.slope', upper=1.0)

        if tower_constraints['frequency_1']['flag']:
            wt_opt.model.add_constraint('towerse.tower.f1',
                lower=tower_constraints['frequency_1']['lower_bound'],
                upper=tower_constraints['frequency_1']['upper_bound'])

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

    
    def set_recorders(self, wt_opt):
        folder_output = self.opt['general']['folder_output']

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.opt['recorder']['flag']:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.opt['recorder']['file_name']))
            wt_opt.driver.add_recorder(recorder)
            wt_opt.add_recorder(recorder)

            wt_opt.driver.recording_options['excludes'] = ['*_df']
            wt_opt.driver.recording_options['record_constraints'] = True 
            wt_opt.driver.recording_options['record_desvars'] = True 
            wt_opt.driver.recording_options['record_objectives'] = True
        
        return wt_opt


    def set_initial(self, wt_opt, wt_init):
        
        wt_opt['blade.opt_var.s_opt_twist']   = np.linspace(0., 1., self.blade_opt['aero_shape']['twist']['n_opt'])
        if self.blade_opt['aero_shape']['twist']['flag']:
            init_twist_opt = np.interp(wt_opt['blade.opt_var.s_opt_twist'], wt_init['components']['blade']['outer_shape_bem']['twist']['grid'], wt_init['components']['blade']['outer_shape_bem']['twist']['values'])
            lb_twist = np.array(self.blade_opt['aero_shape']['twist']['lower_bound'])
            ub_twist = np.array(self.blade_opt['aero_shape']['twist']['upper_bound'])
            wt_opt['blade.opt_var.twist_opt_gain']    = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
            if max(wt_opt['blade.opt_var.twist_opt_gain']) > 1. or min(wt_opt['blade.opt_var.twist_opt_gain']) < 0.:
                print('Warning: the initial twist violates the upper or lower bounds of the twist design variables.')
                
        blade_constraints = self.opt['constraints']['blade']
        wt_opt['blade.opt_var.s_opt_chord']  = np.linspace(0., 1., self.blade_opt['aero_shape']['chord']['n_opt'])
        wt_opt['blade.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., self.blade_opt['structure']['spar_cap_ss']['n_opt'])
        wt_opt['blade.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., self.blade_opt['structure']['spar_cap_ps']['n_opt'])
        wt_opt['rs.constr.max_strainU_spar'] = blade_constraints['strains_spar_cap_ss']['max']
        wt_opt['rs.constr.max_strainL_spar'] = blade_constraints['strains_spar_cap_ps']['max']
        wt_opt['stall_check.stall_margin'] = blade_constraints['stall']['margin'] * 180. / np.pi
        
        return wt_opt


    def set_restart(self, wt_opt):
        if 'warmstart_file' in self.opt['driver']:
            
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict
            db = SqliteDict(self.opt['driver']['warmstart_file'])

            # Grab the last iteration's design variables
            last_key = db['last']
            desvars = db[last_key]['xuser']
            
            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars
            
            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list['output']
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]
                
                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]['scaler']
                adder = design_vars[prom_key]['adder']
                
                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0
                
                scaled_dv = desvars[key] / scaler - adder
                
                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if 'twist_opt_gain' in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv
                    
        return wt_opt
