from ROSCO_toolbox.inputs.validation import load_rosco_yaml
from wisdem.inputs import load_yaml

def assign_ROSCO_values(wt_opt, modeling_options, opt_options):
    # ROSCO tuning parameters
    # Apply tuning yaml input if available

    if modeling_options['ROSCO']['tuning_yaml']:  # default is empty
        inps = load_rosco_yaml(modeling_options['ROSCO']['tuning_yaml'])  # tuning yaml validated in here
        rosco_init_options         = inps['controller_params']

        # Apply changes in modeling options, should have already been validated
        modopts_no_defaults = load_yaml(modeling_options['fname_input_modeling'])  
        for option, value in modopts_no_defaults['ROSCO'].items():
            rosco_init_options[option] = value
    else:
        rosco_init_options = modeling_options['ROSCO']

    # Pitch regulation
    wt_opt['tune_rosco_ivc.omega_pc']      = rosco_init_options['omega_pc']
    wt_opt['tune_rosco_ivc.zeta_pc']       = rosco_init_options['zeta_pc']
    # if not (len(rosco_init_options['omega_pc']) == \
    #     len(rosco_init_options['zeta_pc']) == \
    #     len(rosco_init_options['U_pc'])):
    #     raise Exception('omega_pc, zeta_pc, and U_pc must have the same number of elements in the modeling options')

    # Torque control
    wt_opt['tune_rosco_ivc.omega_vs']      = rosco_init_options['omega_vs']
    wt_opt['tune_rosco_ivc.zeta_vs']       = rosco_init_options['zeta_vs']
    
    # Flap control params
    if rosco_init_options['Flp_Mode'] > 0:
        try:
            wt_opt['tune_rosco_ivc.flp_kp_norm']      = rosco_init_options['flp_kp_norm']
            wt_opt['tune_rosco_ivc.flp_tau']       = rosco_init_options['flp_tau']
        except:
            raise Exception('If Flp_Mode > 0, you must set flp_kp_norm, flp_tau in the modeling options')

    # IPC 
    if rosco_init_options['IPC_ControlMode']:
        wt_opt['tune_rosco_ivc.IPC_Kp1p'] = rosco_init_options['IPC_Kp1p']
        wt_opt['tune_rosco_ivc.IPC_Ki1p'] = rosco_init_options['IPC_Ki1p']
    
    # Robust controller tuning
    if opt_options['design_variables']['control']['servo']['pitch_control']['stability_margin']['flag']:
        wt_opt['tune_rosco_ivc.stability_margin'] = rosco_init_options['linmodel_tuning']['stability_margin']
        wt_opt['tune_rosco_ivc.omega_pc_max'] = rosco_init_options['linmodel_tuning']['omega_pc']['max']
    # other optional parameters
    wt_opt['tune_rosco_ivc.max_pitch']     = rosco_init_options['max_pitch']
    wt_opt['tune_rosco_ivc.min_pitch']     = rosco_init_options['min_pitch']
    wt_opt['tune_rosco_ivc.vs_minspd']     = rosco_init_options['vs_minspd']
    wt_opt['tune_rosco_ivc.ss_vsgain']     = rosco_init_options['ss_vsgain']
    wt_opt['tune_rosco_ivc.ss_pcgain']     = rosco_init_options['ss_pcgain']
    wt_opt['tune_rosco_ivc.ps_percent']    = rosco_init_options['ps_percent']
    
    if rosco_init_options['Fl_Mode']:
        try:
            # wt_opt['tune_rosco_ivc.twr_freq']      = rosco_init_options['twr_freq']
            wt_opt['tune_rosco_ivc.ptfm_freq']     = rosco_init_options['ptfm_freq']
            if 'Kp_float' in rosco_init_options:
                wt_opt['tune_rosco_ivc.Kp_float']      = rosco_init_options['Kp_float']
        except:
            raise Exception('If Fl_Mode > 0, you must set twr_freq and ptfm_freq in modeling options')
        
    # Check for proper Flp_Mode, print warning
    if modeling_options['WISDEM']['RotorSE']['n_tab'] > 1 and rosco_init_options['Flp_Mode'] == 0:
            raise Exception('A distributed aerodynamic control device is specified in the geometry yaml, but Flp_Mode is zero in the modeling options.')
    if modeling_options['WISDEM']['RotorSE']['n_tab'] == 1 and rosco_init_options['Flp_Mode'] > 0:
            raise Exception('Flp_Mode is non zero in the modeling options, but no distributed aerodynamic control device is specified in the geometry yaml.')

    return wt_opt
