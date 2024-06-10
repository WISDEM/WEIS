import openmdao.api as om

# dict input

dict_input = {
  "DesignVars": ["floating.jointdv_0", "floating.jointdv_1"],
  "DesignParms": ["floating.memgrp0.outer_diameter_in", "tune_rosco_ivc.ps_percent"],
  "objective": "AEP",
  "constraints":"raft.Max_PtfmPitch", 
  "outputs":["raft.Max_PtfmPitch","AEP"]
}

Design_Vars = dict_input["DesignVars"]
Design_Parms = dict_input["DesignParms"]
Objective = dict_input["objective"]
Constraints = dict_input["constraints"]
Outputs = dict_input["outputs"]

prob = om.Problem()
model = prob.model


# model.add_subsystem('model', subsys=f_DOE(), promotes_inputs=[Design_Vars, Design_Parms], promotes_outputs=[Objective, Outputs])

prob.driver = om.ScipyOptimizeDriver() #we can change this and add settings

# need to add ref scaling
for k in range(len(Design_Vars)):
    
    if Design_Vars[k] == 'floating.jointdv_0':
        model.add_design_var('floating.jointdv_0', lower=0, upper=28.5)
        model.set_input_defaults('floating.jointdv_0', val=6778.0)
        
    elif Design_Vars[k]  == "floating.jointdv_1":
        model.add_design_var('floating.jointdv_1', lower=0, upper=28.5)
        model.set_input_defaults('floating.jointdv_1', val=6778.0)
        
    elif Design_Vars[k]  == "floating.jointdv_2":
        model.add_design_var('floating.jointdv_2', lower=0, upper=28.5)
        model.set_input_defaults('floating.jointdv_2', val=6778.0)
        
    elif Design_Vars[k]  == "floating.memgrp0.outer_diameter_in":
        model.add_design_var('floating.memgrp0.outer_diameter_in', lower=0, upper=28.5)
        model.set_input_defaults('floating.memgrp0.outer_diameter_in', val=6778.0)
        
    elif Design_Vars[k]  == "floating.memgrp1.outer_diameter_in":
        model.add_design_var('floating.memgrp1.outer_diameter_in', lower=0, upper=28.5)
        model.set_input_defaults('floating.memgrp1.outer_diameter_in', val=6778.0)
        
    elif Design_Vars[k]  == "floating.memgrp2.outer_diameter_in":
        model.add_design_var('floating.memgrp2.outer_diameter_in', lower=0, upper=28.5)
        model.set_input_defaults('floating.memgrp2.outer_diameter_in', val=6778.0)
        
    elif Design_Vars[k]  == "floating.memgrp3.outer_diameter_in":
        model.add_design_var('floating.memgrp3.outer_diameter_in', lower=0, upper=28.5)
        model.set_input_defaults('floating.memgrp3.outer_diameter_in', val=6778.0)
        
    elif Design_Vars[k]  == "tune_rosco_ivc.ps_percent":
        model.add_design_var('tune_rosco_ivc.ps_percent', lower=0, upper=28.5)
        model.set_input_defaults('tune_rosco_ivc.ps_percent', val=6778.0)
        
    elif Design_Vars[k]  == "tune_rosco_ivc.omega_pc":
        model.add_design_var('tune_rosco_ivc.omega_pc', lower=0, upper=28.5)
        model.set_input_defaults('tune_rosco_ivc.omega_pc', val=6778.0)
        
    elif Design_Vars[k]  == "configuration.rotor_diameter_user":
        model.add_design_var('configuration.rotor_diameter_user', lower=0, upper=28.5)
        model.set_input_defaults('configuration.rotor_diameter_user', val=6778.0)
    else:
        print("Please choose correct design variable")


for k in range(len(Design_Parms)):
    
    if Design_Parms[k] == 'floating.jointdv_0':
        model.set_input_defaults('floating.jointdv_0', val=6778.0)
        
    elif Design_Parms[k]  == "floating.jointdv_1":
        model.set_input_defaults('floating.jointdv_1', val=6778.0)
        
    elif Design_Parms[k]  == "floating.jointdv_2":
        model.set_input_defaults('floating.jointdv_2', val=6778.0)
        
    elif Design_Parms[k]  == "floating.memgrp0.outer_diameter_in":
        model.set_input_defaults('floating.memgrp0.outer_diameter_in', val=6778.0)
        
    elif Design_Parms[k]  == "floating.memgrp1.outer_diameter_in":
        model.set_input_defaults('floating.memgrp1.outer_diameter_in', val=6778.0)
        
    elif Design_Parms[k]  == "floating.memgrp2.outer_diameter_in":
        model.set_input_defaults('floating.memgrp2.outer_diameter_in', val=6778.0)
        
    elif Design_Parms[k]  == "floating.memgrp3.outer_diameter_in":
        model.set_input_defaults('floating.memgrp3.outer_diameter_in', val=6778.0)
        
    elif Design_Parms[k]  == "tune_rosco_ivc.ps_percent":
        model.set_input_defaults('tune_rosco_ivc.ps_percent', val=6778.0)
        
    elif Design_Parms[k]  == "tune_rosco_ivc.omega_pc":
        model.set_input_defaults('tune_rosco_ivc.omega_pc', val=6778.0)
        
    elif Design_Parms[k]  == "configuration.rotor_diameter_user":
        model.set_input_defaults('configuration.rotor_diameter_user', val=6778.0)
    else:
        print("Please choose correct parameter")

for k in range(len(Constraints)):
    
    if Constraints[k] == 'raft.Max_PtfmPitch':
        prob.model.add_constraint('raft.Max_PtfmPitch', lower = 0, upper=2.5)
        
    elif Design_Parms[k]  == "raft.Std_PtfmPitch":
        prob.model.add_constraint('raft.Std_PtfmPitch', lower = 0, upper=2.5)
        
    elif Constraints[k]  == "raft.max_nac_accel":
        prob.model.add_constraint('raft.max_nac_accel', lower = 0, upper=2.5)
        
    elif Constraints[k]  == "floatingse.structural_frequencies":
        prob.model.add_constraint('floatingse.structural_frequencies', lower = 0, upper=2.5)

    else:
        print("Please choose correct constraint")


prob.model.add_objective(Objective)


# # Setup the problem
# prob.setup()

# # Execute the model with the given inputs
# prob.run_model()


#get opt data