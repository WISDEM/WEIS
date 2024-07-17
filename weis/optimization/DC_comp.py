import numpy as np
import os
import time
import re
import pickle as pkl
import openmdao.api as om
# from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import KRG
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS


sm_file = '/home/elenaf3/Desktop/DC_studies_code/WEIS/weis/optimization/log_opt.smt'
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
ex_dir = "/home/elenaf3/Desktop/DC_studies_code/WEIS/examples/15_RAFT_Studies/"
fname_wt_input         = os.path.join(ex_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = ex_dir + "modeling_options_level1_doe.yaml"
fname_analysis_options = ex_dir + "analysis_options_level1_doe.yaml"
wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
wt_init, modeling_options, opt_options = wt_initial.get_input_data()

# WTSM = WindTurbineSMOpt()
# WTSM.read_sm(sm_file, modeling_options, opt_options) 

# class WindTurbineSMOpt():

# def __init__(self):
# self._sm_loaded = False

# self.scipy_methods = [
scipy_methods = [
            "SLSQP",
            "Nelder-Mead",
            "COBYLA",
        ]
# self.pyoptsparse_methods = [
pyoptsparse_methods = [
            "SNOPT",
            "CONMIN",
            "NSGA2",
        ]
        

# def read_sm(self, sm_file, modeling_options, opt_options):
# read sm
try:
    sm2 = None
    with open(sm_file, "rb") as f:
       sm2 = pkl.load(f)

except:
    print('Unable to read surrogate model file: {:}.'.format(sm_file))
    raise Exception('Unable to read surrogate model file: {:}.'.format(sm_file))

model = sm2[0]
num_samples = opt_options['driver']['design_of_experiments']['num_samples']


# identify available inputs and outputs provided by the sm
avail_inputs = model['inputs']
avail_outputs_str = model['outputs']
avail_outputs_str = {
  # "bounds": np.zeros((2,len(sm2))),
  "keys": list(len(sm2)*[None]),
  # "vals": np.zeros((num_samples,1))
  }

avail_outputs_str1 = []
for sm_ind in range(len(sm2)):
    # avail_outputs_str['bounds'][:,sm_ind]=sm2[sm_ind]['outputs']['bounds']
    avail_outputs_str['keys'][sm_ind]=sm2[sm_ind]['outputs']['keys']
    avail_outputs_str1.append(sm2[sm_ind]['outputs']['keys'].split('+'))
    
avail_outputs_str_keys = '+'.join(avail_outputs_str['keys'])
avail_outputs = model['outputs']
avail_outputs['keys'] = avail_outputs_str_keys.split('+')


avail_input_keys_ref = list(set(avail_inputs['keys']).intersection([
                'floating.member_main_column:outer_diameter',
                'floating.member_column1:outer_diameter',
                'floating.member_column2:outer_diameter',
                'floating.member_column3:outer_diameter',
                'floating.member_Y_pontoon_upper1:outer_diameter',
                'floating.member_Y_pontoon_upper2:outer_diameter',
                'floating.member_Y_pontoon_upper3:outer_diameter',
                'floating.member_Y_pontoon_lower1:outer_diameter',
                'floating.member_Y_pontoon_lower2:outer_diameter',
                'floating.member_Y_pontoon_lower3:outer_diameter',
                'configuration.rotor_diameter_user',
                'tune_rosco_ivc.Kp_float',
                'tune_rosco_ivc.ps_percent',
                'tune_rosco_ivc.omega_pc',
                'tune_rosco_ivc.ptfm_freq',
                'tune_rosco_ivc.zeta_pc',
                'floating.jointdv_0',
                'floating.jointdv_1',
                'floating.jointdv_2',
                'floating.memgrp0.outer_diameter_in',
                'floating.memgrp1.outer_diameter_in',
                'floating.memgrp2.outer_diameter_in',
                'floating.memgrp3.outer_diameter_in',
                
            ]))
avail_input_keys_ref.sort()

if not (len(avail_input_keys_ref)==len(avail_inputs['keys'])):   
    raise Exception('Some of the sm inputs are not available')

avail_output_keys_ref = list(set(avail_outputs['keys']).intersection([
                'tune_rosco_ivc.ps_percent',
                'tune_rosco_ivc.omega_pc',
                'tune_rosco_ivc.zeta_pc',
                'tune_rosco_ivc.Kp_float',
                'tune_rosco_ivc.ptfm_freq',
                'tune_rosco_ivc.omega_vs',
                'tune_rosco_ivc.zeta_vs',
                'configuration.rotor_diameter_user',
                'towerse.tower.fore_aft_freqs', # 123
                'towerse.tower.side_side_freqs', # 123
                'towerse.tower.torsion_freqs', # 123
                'towerse.tower.top_deflection',
                'floatingse.platform_base_F', # xyz
                'floatingse.platform_base_M', # xyz
                'floating.member_main_column:joint1', # xyz
                'floating.member_main_column:joint2', # xyz
                'floating.member_column1:joint1', # xyz
                'floating.member_column1:joint2', # xyz
                'floating.member_column2:joint1', # xyz
                'floating.member_column2:joint2', # xyz
                'floating.member_column3:joint1', # xyz
                'floating.member_column3:joint2', # xyz
                'floating.jointdv_0', # keel z-location
                'floating.jointdv_1', # freeboard z-location
                'floating.jointdv_2', # column123 r-location
                'raft.Max_Offset', # Maximum distance in surge/sway direction [m]
                'raft.heave_avg', # Average heave over all cases [m]
                'raft.Max_PtfmPitch', # Maximum platform pitch over all cases [deg]
                'raft.Std_PtfmPitch', # Average platform pitch std. over all cases [deg]
                'rigid_body_periods', # Rigid body natural period [s]
                'raft.heave_period', # Heave natural period [s]
                'raft.pitch_period', # Pitch natural period [s]
                'raft.roll_period', # Roll natural period [s]
                'raft.surge_period', # Surge natural period [s]
                'raft.sway_period', # Sway natural period [s]
                'raft.yaw_period', # Yaw natural period [s]
                'raft.max_nac_accel', # Maximum nacelle accelleration over all cases [m/s**2]
                'raft.max_tower_base', # Maximum tower base moment over all cases [N*m]
                'raft.platform_total_center_of_mass', # xyz
                'raft.platform_displacement',
                'raft.platform_mass', # Platform mass
                'tcons.tip_deflection_ratio', # Blade tip deflection ratio (constrained to be <=1.0)
                'financese.lcoe', # WEIS LCOE from FinanceSE
                'rotorse.rp.AEP', # WISDEM AEP from RotorSE
                'rotorse.blade_mass', # Blade mass
                #'towerse.tower_mass', # Tower mass
                'fixedse.structural_mass', # System structural mass for fixed foundation turbines
                'floatingse.system_structural_mass', # System structural mass for floating turbines
                'floatingse.platform_mass', # Platform mass from FloatingSE
                'floatingse.platform_cost', # Platform cost
                #'floatingse.mooring_mass', # Mooring mass
                #'floatingse.mooring_cost', # Mooring cost
                'floatingse.structural_frequencies', 
            ]))
avail_output_keys_ref.sort()


# %%
# requested dv and outputs for opt
DCA_req = {
  "DesignVars": ["floating.member_column1:outer_diameter", "tune_rosco_ivc.Kp_float"],
  "DesignParms": ["floating.memgrp0.outer_diameter_in", "tune_rosco_ivc.ps_percent"],
  "objective": "floatingse.platform_mass",
  "constraints":["raft.Max_PtfmPitch"], 
}

req_dv = DCA_req["DesignVars"]
req_design_parms = DCA_req["DesignParms"]
req_objective = [DCA_req["objective"]]
req_constraints = DCA_req["constraints"]
req_outputs = req_objective + req_constraints
req_inputs = req_dv + req_design_parms

output_keys_ref = list(set(req_outputs).intersection(avail_output_keys_ref))
objective_key = list(set(req_objective).intersection(avail_output_keys_ref))
constraints_key = list(set(req_constraints).intersection(avail_output_keys_ref))
opt_dv_key = list(set(req_dv).intersection(avail_input_keys_ref))
opt_params_key = list(set(req_design_parms).intersection(avail_input_keys_ref))

class SM_Comp(om.ExplicitComponent):


    def setup(self):
        
        for k in range(len(avail_input_keys_ref)):   #add all sm inputs, add their values
            
            if avail_input_keys_ref[k] == 'floating.jointdv_0':
                self.add_input('floating.jointdv_0')
                
            elif avail_input_keys_ref[k]  == "floating.jointdv_1":
                self.add_input('floating.jointdv_1')
                
            elif avail_input_keys_ref[k]  == "floating.jointdv_2":
                self.add_input('floating.jointdv_2')
                
            elif avail_input_keys_ref[k]  == "floating.memgrp0.outer_diameter_in":
                self.add_input('floating.memgrp0.outer_diameter_in')
                
            elif avail_input_keys_ref[k]  == "floating.memgrp1.outer_diameter_in":
                self.add_input('floating.memgrp1.outer_diameter_in')
                
            elif avail_input_keys_ref[k]  == "floating.memgrp2.outer_diameter_in":
                self.add_input('floating.memgrp2.outer_diameter_in')
                
            elif avail_input_keys_ref[k]  == "floating.memgrp3.outer_diameter_in":
                self.add_input('floating.memgrp3.outer_diameter_in')
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.ps_percent":
                self.add_input('tune_rosco_ivc.ps_percent')
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.omega_pc":
                self.add_input('tune_rosco_ivc.omega_pc')
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.ptfm_freq":
                self.add_input('tune_rosco_ivc.ptfm_freq')
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.zeta_pc":
                self.add_input('tune_rosco_ivc.zeta_pc')
                
            elif avail_input_keys_ref[k]  == "configuration.rotor_diameter_user":
                self.add_input('configuration.rotor_diameter_user')
                
            elif avail_input_keys_ref[k]  == "floating.member_main_column:outer_diameter":
                self.add_input('floating.member_main_column:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_column1:outer_diameter":
                self.add_input('floating.member_column1:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_column2:outer_diameter":
                self.add_input('floating.member_column2:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_column3:outer_diameter":
                self.add_input('floating.member_column3:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                self.add_input('floating.member_Y_pontoon_upper1:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                self.add_input('floating.member_Y_pontoon_upper2:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                self.add_input('floating.member_Y_pontoon_upper3:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                self.add_input('floating.member_Y_pontoon_lower1:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                self.add_input('floating.member_Y_pontoon_lower2:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                self.add_input('floating.member_Y_pontoon_lower3:outer_diameter')
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.Kp_float":
                self.add_input('tune_rosco_ivc.Kp_float')
            else:
                print("Please choose correct input")


        for k in range(len(output_keys_ref)):
            
            if output_keys_ref[k] == 'raft.Max_PtfmPitch':
                self.add_output('raft.Max_PtfmPitch')
                
            elif output_keys_ref[k]  == "raft.Std_PtfmPitch":
                self.add_output('raft.Std_PtfmPitch')
                
            elif output_keys_ref[k]  == "raft.max_nac_accel":
                self.add_output('raft.max_nac_accel')
                
            elif output_keys_ref[k]  == "floatingse.structural_frequencies":
                self.add_output('floatingse.structural_frequencies')
                
            elif output_keys_ref[k]  == "floatingse.platform_mass":
                self.add_output('floatingse.platform_mass')

            else:
                print("Please choose correct constraint")
        

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')
        
    def run_predict(self, predict_output, predict_input):
        
        output_indx = avail_outputs_str1.index(predict_output)
        output_value, variance = sm2[output_indx]['surrogate'].predict(np.array([predict_input]))
        return output_value

    def compute(self, inputs, outputs):

        predict_input = []
        for k in range(len(avail_input_keys_ref)):   #add all sm inputs
            
            if avail_input_keys_ref[k] == 'floating.jointdv_0':
                floating_jointdv_0 = inputs['floating.jointdv_0']
                predict_input.append(floating_jointdv_0)
                
            elif avail_input_keys_ref[k]  == "floating.jointdv_1":
                floating_jointdv_1 = inputs['floating.jointdv_1']
                predict_input.append(floating_jointdv_1)
                
            elif avail_input_keys_ref[k]  == "floating.jointdv_2":
                floating_jointdv_2 = inputs['floating.jointdv_2']
                predict_input.append(floating_jointdv_2)
                
            elif avail_input_keys_ref[k]  == "floating.memgrp0.outer_diameter_in":
                floating_memgrp0_outer_diameter_in = inputs['floating.memgrp0.outer_diameter_in']
                predict_input.append(floating_memgrp0_outer_diameter_in)
                
            elif avail_input_keys_ref[k]  == "floating.memgrp1.outer_diameter_in":
                floating_memgrp1_outer_diameter_in = inputs['floating.memgrp1.outer_diameter_in']
                predict_input.append(floating_memgrp1_outer_diameter_in)
                
            elif avail_input_keys_ref[k]  == "floating.memgrp2.outer_diameter_in":
                floating_memgrp2_outer_diameter_in = inputs['floating.memgrp2.outer_diameter_in']
                predict_input.append(floating_memgrp2_outer_diameter_in)
                
            elif avail_input_keys_ref[k]  == "floating.memgrp3.outer_diameter_in":
                floating_memgrp3_outer_diameter_in = inputs['floating.memgrp3.outer_diameter_in']
                predict_input.append(floating_memgrp3_outer_diameter_in)
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.ps_percent":
                tune_rosco_ivc_ps_percent = inputs['tune_rosco_ivc.ps_percent']
                predict_input.append(tune_rosco_ivc_ps_percent)
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.omega_pc":
                tune_rosco_ivc_omega_pc = inputs['tune_rosco_ivc.omega_pc']
                predict_input.append(tune_rosco_ivc_omega_pc)
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.ptfm_freq":
                tune_rosco_ivc_ptfm_freq = inputs['tune_rosco_ivc.ptfm_freq']
                predict_input.append(tune_rosco_ivc_ptfm_freq)
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.zeta_pc":
                tune_rosco_ivc_zeta_pc = inputs['tune_rosco_ivc.zeta_pc']
                predict_input.append(tune_rosco_ivc_zeta_pc)
                
            elif avail_input_keys_ref[k]  == "configuration.rotor_diameter_user":
                configuration_rotor_diameter_user = inputs['configuration.rotor_diameter_user']
                predict_input.append(configuration_rotor_diameter_user)
            
            elif avail_input_keys_ref[k]  == "floating.member_main_column:outer_diameter":
                floating_member_main_column_outer_diameter = inputs['floating.member_main_column:outer_diameter']
                predict_input.append(floating_member_main_column_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_column1:outer_diameter":
                floating_member_column1_outer_diameter = inputs['floating.member_column1:outer_diameter']
                predict_input.append(floating_member_column1_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_column2:outer_diameter":
                floating_member_column2_outer_diameter = inputs['floating.member_column2:outer_diameter']
                predict_input.append(floating_member_column2_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_column3:outer_diameter":
                floating_member_column3_outer_diameter = inputs['floating.member_column3:outer_diameter']
                predict_input.append(floating_member_column3_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                floating_member_Y_pontoon_upper1_outer_diameter = inputs['floating.member_Y_pontoon_upper1:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper1_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                floating_member_Y_pontoon_upper2_outer_diameter = inputs['floating.member_Y_pontoon_upper2:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper2_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                floating_member_Y_pontoon_upper3_outer_diameter = inputs['floating.member_Y_pontoon_upper3:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper3_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                floating_member_Y_pontoon_lower1_outer_diameter = inputs['floating.member_Y_pontoon_lower1:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower1_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                floating_member_Y_pontoon_lower2_outer_diameter = inputs['floating.member_Y_pontoon_lower2:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower2_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                floating_member_Y_pontoon_lower3_outer_diameter = inputs['floating.member_Y_pontoon_lower3:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower3_outer_diameter)
                
            elif avail_input_keys_ref[k]  == "tune_rosco_ivc.Kp_float":
                tune_rosco_ivc_Kp_float = inputs['tune_rosco_ivc.Kp_float']
                predict_input.append(tune_rosco_ivc_Kp_float)
                
            else:
                print("Please choose correct input")
                        
        predict_input_flat = [arr.flatten()[0] for arr in predict_input]
        for k in range(len(output_keys_ref)):
            
            if output_keys_ref[k] == 'raft.Max_PtfmPitch':
                outputs['raft.Max_PtfmPitch'] = self.run_predict(['raft.Max_PtfmPitch'], predict_input_flat) 
                
            elif output_keys_ref[k]  == "raft.Std_PtfmPitch":
                outputs['raft.Std_PtfmPitch'] = self.run_predict(['raft.Std_PtfmPitch'], predict_input_flat)
                
            elif output_keys_ref[k]  == "raft.max_nac_accel":
                outputs['raft.max_nac_accel'] = self.run_predict(['raft.max_nac_accel'], predict_input_flat)
                
            elif output_keys_ref[k]  == "floatingse.structural_frequencies":
                outputs['floatingse.structural_frequencies'] = self.run_predict(['floatingse.structural_frequencies'], predict_input_flat)
            
            elif output_keys_ref[k]  == "floatingse.platform_mass":
                outputs['floatingse.platform_mass'] = self.run_predict(['floatingse.platform_mass'], predict_input_flat)
                
            else:
                print("Please choose correct output")


# from WISDEM/wisdem/glue_code/gc_PoseOptimization.py
# def _set_optimizer_properties(self, prob, options_keys=[], opt_settings_keys=[], mapped_keys={}):
def _set_optimizer_properties(prob, options_keys=[], opt_settings_keys=[], mapped_keys={}):

        """
        Set the optimizer properties, both the `driver.options` and
        `driver.opt_settings`. See OpenMDAO documentation on drivers
        to determine which settings are set by either options or
        opt_settings.

        Parameters
        ----------
        wt_opt : OpenMDAO problem object
            The wind turbine problem object.
        options_keys : list
            List of keys for driver options to be set.
        opt_settings_keys: list
            List of keys for driver opt_settings to be set.
        mapped_keys: dict
            Key pairs where the yaml name differs from what's expected
            by the driver. Specifically, the key is what's given in the yaml
            and the value is what's expected by the driver.

        Returns
        -------
        wt_opt : OpenMDAO problem object
            The updated wind turbine problem object with driver settings applied.
        """

        sm_opt_options = opt_options["driver"]["surrogate_based_optimization"]

        # Loop through all of the options provided and set them in the OM driver object
        for key in options_keys:
            if key in sm_opt_options:
                if key in mapped_keys:
                    prob.driver.options[mapped_keys[key]] = sm_opt_options[key]
                else:
                    prob.driver.options[key] = sm_opt_options[key]

        # Loop through all of the opt_settings provided and set them in the OM driver object
        for key in opt_settings_keys:
            if key in sm_opt_options:
                if key in mapped_keys:
                    prob.driver.opt_settings[mapped_keys[key]] = sm_opt_options[key]
                else:
                    prob.driver.opt_settings[key] = sm_opt_options[key]

        return prob


prob = om.Problem()
model = prob.model
# model.add_subsystem('model', subsys=SM_Comp(), promotes_inputs=avail_inputs['keys'])
model.add_subsystem('model', subsys=SM_Comp(), promotes_inputs=opt_dv_key, promotes_outputs=output_keys_ref)
solver = opt_options['driver']['surrogate_based_optimization']['solver']

# Set optimization solver and options. First, Scipy's SLSQP and COBYLA
# if solver in self.scipy_methods:
if solver in scipy_methods:

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = solver
    
    options_keys = ["tol", "max_iter", "disp"]
    opt_settings_keys = ["rhobeg", "catol", "adaptive"]
    mapped_keys = {"max_iter": "maxiter"}
    # prob = self._set_optimizer_properties(prob, options_keys, opt_settings_keys, mapped_keys)
    prob = _set_optimizer_properties(prob, options_keys, opt_settings_keys, mapped_keys)

# The next two optimization methods require pyOptSparse.
# elif solver in self.pyoptsparse_methods:
elif solver in pyoptsparse_methods:
    try:
        from openmdao.api import pyOptSparseDriver
    except:
        raise ImportError(
            f"You requested the optimization solver {opt_options['solver']}, but you have not installed pyOptSparse. Please do so and rerun."
        )
    prob.driver = pyOptSparseDriver()
    try:
        prob.driver.options["optimizer"] = solver
    except:
        raise ImportError(
            f"You requested the optimization solver {solver}, but you have not installed it within pyOptSparse. Please build {solver} and rerun."
        )

    # Most of the pyOptSparse options have special syntax when setting them,
    # so here we set them by hand instead of using `_set_optimizer_properties` for SNOPT and CONMIN.
    if solver == "CONMIN":
        prob.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

    if solver == "NSGA2":
        opt_settings_keys = [
            "PopSize",
            "maxGen",
            "pCross_real",
            "pMut_real",
            "eta_c",
            "eta_m",
            "pCross_bin",
            "pMut_bin",
            "PrintOut",
            "seed",
            "xinit",
        ]
        prob = _set_optimizer_properties(prob, opt_settings_keys=opt_settings_keys)

    elif solver == "SNOPT":
        prob.driver.opt_settings["Major optimality tolerance"] = float(opt_options["tol"])
        prob.driver.opt_settings["Major iterations limit"] = int(opt_options["max_major_iter"])
        prob.driver.opt_settings["Iterations limit"] = int(opt_options["max_minor_iter"])
        prob.driver.opt_settings["Major feasibility tolerance"] = float(opt_options["tol"])
        if "time_limit" in opt_options:
            prob.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
        folder_output = opt_options["general"]["folder_output"]
        prob.driver.opt_settings["Summary file"] = os.path.join(folder_output, "SNOPT_Summary_file.txt")
        prob.driver.opt_settings["Print file"] = os.path.join(folder_output, "SNOPT_Print_file.txt")
        if "hist_file_name" in opt_options:
            prob.driver.hist_file = opt_options["hist_file_name"]
        if "verify_level" in opt_options:
            prob.driver.opt_settings["Verify level"] = opt_options["verify_level"]
        else:
            prob.driver.opt_settings["Verify level"] = -1
    if "hotstart_file" in opt_options:
        prob.driver.hotstart_file = opt_options["hotstart_file"]

def get_bounds(avail_inputs,dv_key):
    dv_indx = avail_inputs['keys'].index(dv_key)
    lb = avail_inputs['bounds'][0,dv_indx]
    ub = avail_inputs['bounds'][1,dv_indx]
    return lb, ub

# add dv
for k in range(len(opt_dv_key)):
    
    if opt_dv_key[k] == 'floating.jointdv_0':
        lb, ub = get_bounds(avail_inputs,'floating.jointdv_0')
        model.add_design_var('floating.jointdv_0', lower=lb, upper=ub)
        model.set_input_defaults('floating.jointdv_0', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.jointdv_1":
        lb, ub = get_bounds(avail_inputs,'floating.jointdv_1')
        model.add_design_var('floating.jointdv_1', lower=lb, upper=ub)
        model.set_input_defaults('floating.jointdv_1', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.jointdv_2":
        lb, ub = get_bounds(avail_inputs,'floating.jointdv_2')
        model.add_design_var('floating.jointdv_2', lower=lb, upper=ub)
        model.set_input_defaults('floating.jointdv_2', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.memgrp0.outer_diameter_in":
        lb, ub = get_bounds(avail_inputs,'floating.memgrp0.outer_diameter_in')
        model.add_design_var('floating.memgrp0.outer_diameter_in', lower=lb, upper=ub)
        model.set_input_defaults('floating.memgrp0.outer_diameter_in', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.memgrp1.outer_diameter_in":
        lb, ub = get_bounds(avail_inputs,'floating.memgrp1.outer_diameter_in')
        model.add_design_var('floating.memgrp1.outer_diameter_in', lower=lb, upper=ub)
        model.set_input_defaults('floating.memgrp1.outer_diameter_in', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.memgrp2.outer_diameter_in":
        lb, ub = get_bounds(avail_inputs,'floating.memgrp2.outer_diameter_in')
        model.add_design_var('floating.memgrp2.outer_diameter_in', lower=lb, upper=ub)
        model.set_input_defaults('floating.memgrp2.outer_diameter_in', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.memgrp3.outer_diameter_in":
        lb, ub = get_bounds(avail_inputs,'floating.memgrp3.outer_diameter_in')
        model.add_design_var('floating.memgrp3.outer_diameter_in', lower=lb, upper=ub)
        model.set_input_defaults('floating.memgrp3.outer_diameter_in', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "tune_rosco_ivc.ps_percent":
        lb, ub = get_bounds(avail_inputs,'tune_rosco_ivc.ps_percent')
        model.add_design_var('tune_rosco_ivc.ps_percent', lower=lb, upper=ub)
        model.set_input_defaults('tune_rosco_ivc.ps_percent', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "tune_rosco_ivc.omega_pc":
        lb, ub = get_bounds(avail_inputs,'tune_rosco_ivc.omega_pc')
        model.add_design_var('tune_rosco_ivc.omega_pc', lower=lb, upper=ub)
        model.set_input_defaults('tune_rosco_ivc.omega_pc', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "tune_rosco_ivc.ptfm_freq":
        lb, ub = get_bounds(avail_inputs,'tune_rosco_ivc.ptfm_freq')
        model.add_design_var('tune_rosco_ivc.ptfm_freq', lower=lb, upper=ub)
        model.set_input_defaults('tune_rosco_ivc.ptfm_freq', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "tune_rosco_ivc.zeta_pc":
        lb, ub = get_bounds(avail_inputs,'tune_rosco_ivc.zeta_pc')
        model.add_design_var('tune_rosco_ivc.zeta_pc', lower=lb, upper=ub)
        model.set_input_defaults('tune_rosco_ivc.zeta_pc', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "configuration.rotor_diameter_user":
        lb, ub = get_bounds(avail_inputs,'configuration.rotor_diameter_user')
        model.add_design_var('configuration.rotor_diameter_user', lower=lb, upper=ub)
        model.set_input_defaults('configuration.rotor_diameter_user', val=(lb+ub)/2)
    
    elif opt_dv_key[k]  == "floating.member_main_column:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_main_column:outer_diameter')
        model.add_design_var('floating.member_main_column:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_main_column:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_column1:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_column1:outer_diameter')
        model.add_design_var('floating.member_column1:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_column1:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_column2:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_column2:outer_diameter')
        model.add_design_var('floating.member_column2:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_column2:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_column3:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_column3:outer_diameter')
        model.add_design_var('floating.member_column3:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_column3:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_upper1:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_upper1:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_upper1:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_upper2:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_upper2:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_upper2:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_upper3:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_upper3:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_upper3:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_lower1:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_lower1:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_lower1:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_lower2:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_lower2:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_lower2:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
        lb, ub = get_bounds(avail_inputs,'floating.member_Y_pontoon_lower3:outer_diameter')
        model.add_design_var('floating.member_Y_pontoon_lower3:outer_diameter', lower=lb, upper=ub)
        model.set_input_defaults('floating.member_Y_pontoon_lower3:outer_diameter', val=(lb+ub)/2)
        
    elif opt_dv_key[k]  == "tune_rosco_ivc.Kp_float":
        lb, ub = get_bounds(avail_inputs,'tune_rosco_ivc.Kp_float')
        model.add_design_var('tune_rosco_ivc.Kp_float', lower=lb, upper=ub)
        model.set_input_defaults('tune_rosco_ivc.Kp_float', val=(lb+ub)/2)
    else:
        print("Please choose correct design variable")


for k in range(len(opt_params_key)):  #need to get the values from Saeid's code, the sm inputs that are not dvs or opt parameters also need vals
    
    if opt_params_key[k] == 'floating.jointdv_0':
        model.set_input_defaults('floating.jointdv_0', val=6778.0)
        
    elif opt_params_key[k]  == "floating.jointdv_1":
        model.set_input_defaults('floating.jointdv_1', val=6778.0)
        
    elif opt_params_key[k]  == "floating.jointdv_2":
        model.set_input_defaults('floating.jointdv_2', val=6778.0)
        
    elif opt_params_key[k]  == "floating.memgrp0.outer_diameter_in":
        model.set_input_defaults('floating.memgrp0.outer_diameter_in', val=6778.0)
        
    elif opt_params_key[k]  == "floating.memgrp1.outer_diameter_in":
        model.set_input_defaults('floating.memgrp1.outer_diameter_in', val=6778.0)
        
    elif opt_params_key[k]  == "floating.memgrp2.outer_diameter_in":
        model.set_input_defaults('floating.memgrp2.outer_diameter_in', val=6778.0)
        
    elif opt_params_key[k]  == "floating.memgrp3.outer_diameter_in":
        model.set_input_defaults('floating.memgrp3.outer_diameter_in', val=6778.0)
        
    elif opt_params_key[k]  == "tune_rosco_ivc.ps_percent":
        model.set_input_defaults('tune_rosco_ivc.ps_percent', val=6778.0)
        
    elif opt_params_key[k]  == "tune_rosco_ivc.omega_pc":
        model.set_input_defaults('tune_rosco_ivc.omega_pc', val=6778.0)
        
    elif opt_params_key[k]  == "tune_rosco_ivc.ptfm_freq":
        model.set_input_defaults('tune_rosco_ivc.ptfm_freq', val=6778.0)
        
    elif opt_params_key[k]  == "tune_rosco_ivc.zeta_pc":
        model.set_input_defaults('tune_rosco_ivc.zeta_pc', val=6778.0)
        
    elif opt_params_key[k]  == "configuration.rotor_diameter_user":
        model.set_input_defaults('configuration.rotor_diameter_user', val=6778.0)
    
    elif opt_params_key[k]  == "floating.member_main_column:outer_diameter":
        model.set_input_defaults('floating.member_main_column:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_column1:outer_diameter":
        model.set_input_defaults('floating.member_column1:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_column2:outer_diameter":
        model.set_input_defaults('floating.member_column2:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_column3:outer_diameter":
        model.set_input_defaults('floating.member_column3:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_upper1:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_upper2:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_upper3:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_lower1:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_lower2:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
        model.set_input_defaults('floating.member_Y_pontoon_lower3:outer_diameter', val=6778.0)
        
    elif opt_params_key[k]  == "tune_rosco_ivc.Kp_float":
        model.set_input_defaults('tune_rosco_ivc.Kp_float', val=6778.0)
        
    else:
        print("Please choose correct parameter")

for k in range(len(constraints_key)):  #need to get the lower and upper values from Saeid's code
    
    if constraints_key[k] == 'raft.Max_PtfmPitch':
        prob.model.add_constraint('raft.Max_PtfmPitch', lower = 0, upper=2.5)
        
    elif constraints_key[k]  == "raft.Std_PtfmPitch":
        prob.model.add_constraint('raft.Std_PtfmPitch', lower = 0, upper=2.5)
        
    elif constraints_key[k]  == "raft.max_nac_accel":
        prob.model.add_constraint('raft.max_nac_accel', lower = 0, upper=2.5)
        
    elif constraints_key[k]  == "floatingse.structural_frequencies":
        prob.model.add_constraint('floatingse.structural_frequencies', lower = 0, upper=2.5)

    else:
        print("Please choose correct constraint")


prob.model.add_objective(objective_key[0])


# Setup the problem
prob.setup()

# Execute the model with the given inputs
prob.run_model()

# %%
#get opt data
objective_st = prob.get_val(objective_key[0])