from weis.inputs.validation import load_modeling_yaml

from rosco.toolbox.inputs.validation import load_rosco_yaml, write_rosco_yaml

from rosco.toolbox.ofTools.fast_io.update_discons import update_discons



original_tuning_yaml = '/Users/dzalkind/Projects/USFLOWT/USFLOWT_repo/ROSCO/USFLOWT_ROSCO.yaml'
output_modeling_yaml = '/Users/dzalkind/Projects/USFLOWT/USFLOWT_repo/WEIS/outputs/2_ROSCO_opt/1_long_run/iea15mw-modeling.yaml'

new_rosco_yaml = '/Users/dzalkind/Projects/USFLOWT/USFLOWT_repo/ROSCO/USFLOWT_ROSCO_opt.yaml'

rosco_opts = load_rosco_yaml(original_tuning_yaml)
mod_opts = load_modeling_yaml(output_modeling_yaml)


rosco_opts['controller_params'].update(mod_opts['ROSCO'])
write_rosco_yaml(rosco_opts,new_rosco_yaml)


# paths relative to Tune_Case/ and Test_Case/
map_rel = {
    new_rosco_yaml: '/Users/dzalkind/Projects/USFLOWT/USFLOWT_repo/OpenFAST/USFLOWT_Definition/Controller/USFLOWT_DISCON_opt.IN',
}

update_discons(map_rel)


