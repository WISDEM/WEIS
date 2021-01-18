def assign_ROSCO_values(wt_opt, modeling_options, control):
    # ROSCO tuning parameters
    wt_opt['tune_rosco_ivc.PC_omega']      = control['pitch']['PC_omega']
    wt_opt['tune_rosco_ivc.PC_zeta']       = control['pitch']['PC_zeta']
    wt_opt['tune_rosco_ivc.VS_omega']      = control['torque']['VS_omega']
    wt_opt['tune_rosco_ivc.VS_zeta']       = control['torque']['VS_zeta']
    if modeling_options['Level3']['ROSCO']['Flp_Mode'] > 0:
        wt_opt['tune_rosco_ivc.Flp_omega']      = control['dac']['Flp_omega']
        wt_opt['tune_rosco_ivc.Flp_zeta']       = control['dac']['Flp_zeta']
    if 'IPC' in control.keys():
        wt_opt['tune_rosco_ivc.IPC_KI']      = control['IPC']['IPC_gain_1P']
    # # other optional parameters
    wt_opt['tune_rosco_ivc.max_pitch']     = control['pitch']['max_pitch']
    wt_opt['tune_rosco_ivc.min_pitch']     = control['pitch']['min_pitch']
    wt_opt['tune_rosco_ivc.vs_minspd']     = control['torque']['VS_minspd']
    wt_opt['tune_rosco_ivc.ss_vsgain']     = control['setpoint_smooth']['ss_vsgain']
    wt_opt['tune_rosco_ivc.ss_pcgain']     = control['setpoint_smooth']['ss_pcgain']
    wt_opt['tune_rosco_ivc.ps_percent']    = control['pitch']['ps_percent']
    # Check for proper Flp_Mode, print warning
    if modeling_options['WISDEM']['RotorSE']['n_tab'] > 1 and modeling_options['Level3']['ROSCO']['Flp_Mode'] == 0:
            raise Exception('A distributed aerodynamic control device is specified in the geometry yaml, but Flp_Mode is zero in the modeling options.')
    if modeling_options['WISDEM']['RotorSE']['n_tab'] == 1 and modeling_options['Level3']['ROSCO']['Flp_Mode'] > 0:
            raise Exception('Flp_Mode is non zero in the modeling options, but no distributed aerodynamic control device is specified in the geometry yaml.')

    return wt_opt
