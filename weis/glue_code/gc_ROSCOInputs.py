def assign_ROSCO_values(wt_opt, modeling_options, opt_options):

    rosco_init_options = modeling_options["ROSCO"]

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
    
    # Optional parameters
    optional_params = [
         'max_pitch',
         'min_pitch',
         'vs_minspd',
         'ss_vsgain',
         'ss_pcgain',
         'ps_percent',
    ]
    for param in optional_params:
        if param in rosco_init_options:
            wt_opt[f'tune_rosco_ivc.{param}'] = rosco_init_options[param]
    
    if rosco_init_options["Fl_Mode"]:
        try:
            # wt_opt["tune_rosco_ivc.twr_freq"]      = rosco_init_options["twr_freq"]
            wt_opt["tune_rosco_ivc.ptfm_freq"]     = rosco_init_options["ptfm_freq"]
            if "Kp_float" in rosco_init_options:
                wt_opt["tune_rosco_ivc.Kp_float"]      = rosco_init_options["Kp_float"]
        except:
            raise Exception("If Fl_Mode > 0, you must set twr_freq and ptfm_freq in modeling options")

    # Check for proper Flp_Mode, print warning
    #if modeling_options["WISDEM"]["RotorSE"]["n_tab"] > 1 and rosco_init_options["Flp_Mode"] == 0:
    #        raise Exception("A distributed aerodynamic control device is specified in the geometry yaml, but Flp_Mode is zero in the modeling options.")
    if rosco_init_options["Flp_Mode"] > 0:
        raise Exception("Flp_Mode is non zero in the modeling options, but no distributed aerodynamic control device is allowed in the geometry yaml. anymore")

    return wt_opt
