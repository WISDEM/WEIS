def assign_ROSCO_values(wt_opt, modeling_options, opt_options):

    rosco_init_options = modeling_options['ROSCO']

    # Required inputs
    wt_opt['tune_rosco_ivc.max_pitch']     = rosco_init_options['max_pitch']
    wt_opt['tune_rosco_ivc.min_pitch']     = rosco_init_options['min_pitch']

    # Robust controller tuning
    if opt_options['design_variables']['control']['servo']['pitch_control']['stability_margin']['flag']:
        wt_opt['tune_rosco_ivc.stability_margin'] = rosco_init_options['linmodel_tuning']['stability_margin']
        wt_opt['tune_rosco_ivc.omega_pc_max'] = rosco_init_options['linmodel_tuning']['omega_pc']['max']        
    
    # Generic input variables
    rosco_tuning_dvs = opt_options['design_variables']['control']['rosco_tuning']
    for dv in rosco_tuning_dvs:
        wt_opt[f"tune_rosco_ivc.{dv['name']}"] = dv['start']
        
    # DISCON inputs (ROSCO)
    discon_dvs = opt_options['design_variables']['control']['discon']
    for dv in discon_dvs:
        wt_opt[f"tune_rosco_ivc.discon:{dv['name']}"] = dv['start']
    
    # Check for proper Flp_Mode, print warning
    if modeling_options['WISDEM']['RotorSE']['n_tab'] > 1 and rosco_init_options['Flp_Mode'] == 0:
            raise Exception('A distributed aerodynamic control device is specified in the geometry yaml, but Flp_Mode is zero in the modeling options.')
    if modeling_options['WISDEM']['RotorSE']['n_tab'] == 1 and rosco_init_options['Flp_Mode'] > 0:
            raise Exception('Flp_Mode is non zero in the modeling options, but no distributed aerodynamic control device is specified in the geometry yaml.')

    return wt_opt
