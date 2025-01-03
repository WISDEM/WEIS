import os
from weis.aeroelasticse.openmdao_openfast import FASTLoadCases
import weis.inputs as sch
from rosco import discon_lib_path
from wisdem.inputs.validation import load_yaml, write_yaml, _validate, simple_types, DefaultValidatingDraft7Validator


this_dir = os.path.dirname( os.path.realpath(__file__) )

def load_testbench_yaml(filename):
    # Will eventually move this somewhere else for clarity
    options = load_yaml(filename)

    # TODO: validate against schema

    return options


def main():


    input_file = 'testbench_options.yaml'
    testbench_options = load_testbench_yaml(os.path.join(this_dir,input_file))

    #### NOTHING BELOW HERE SHOULD CHANGE FOR THE USER


    # Configure modeling options for FASTLoadCases 
    modopt_file = os.path.join(this_dir,input_file)
    modeling_options = sch.load_modeling_yaml(modopt_file)
    modeling_options['General']['openfast_configuration']['fst_vt'] = {}
    modeling_options['fname_input_modeling'] = modopt_file
    modeling_options['materials'] = {}

    # Unpack inputs to openmdao_openfast
    inputs = {
        'V_cutin':          testbench_options['Turbine_Info']['wind_speed_cut_in'],
        'V_cutout':         testbench_options['Turbine_Info']['wind_speed_cut_out'],
        'Vrated':           testbench_options['Turbine_Info']['wind_speed_rated'],
        'hub_height':       testbench_options['Turbine_Info']['hub_height'],
        'Rtip':             testbench_options['Turbine_Info']['rotor_radius'],
        'shearExp': 0.14,   # Not used, hard code
    }

    discrete_inputs = {
        'turbine_class': testbench_options['Turbine_Info']['turbine_class'],
        'turbulence_class': testbench_options['Turbine_Info']['turbulence_class'],
    }

    modeling_options['flags'] = testbench_options['Turbine_Info']['flags']  # Used to determine output channels in OpenFAST


    # Usually, in WEIS, we populate fst_vt['DISCON'] with previously tuned ROSCO.  
    # This will just use the DISCON referenced by the OpenFAST model. 
    # TODO: should we do any tuning or specify a DISCON?
    modeling_options['ROSCO']['flag'] = False   

    OFmgmt = modeling_options['General']['openfast_configuration']
    OFmgmt['cores'] = modeling_options['Testbench_Options']['n_cores']
    OFmgmt['path2dll'] = discon_lib_path

    # Set default directories, relative to testbench options
    OFmgmt['OF_run_fst'] = testbench_options['Testbench_Options']['output_filebase']
    OFmgmt['OF_run_dir'] = os.path.join(os.path.dirname(modopt_file), testbench_options['Testbench_Options']['output_directory'])
    modeling_options['Level3']['openfast_dir'] = os.path.join(os.path.dirname(modopt_file),testbench_options['Level3']['openfast_dir'])

    flc = FASTLoadCases()
    flc.options['modeling_options'] = modeling_options
    flc.n_blades = 3
    flc.of_inumber = 0

    flc.setup_directories()
    fst_vt = flc.create_fst_vt(inputs, discrete_inputs)

    summary_stats, extreme_table, DELs, Damage, case_list, case_name, chan_time, dlc_generator  = flc.run_FAST(inputs, discrete_inputs, fst_vt)


    # Post-processing here
    print('here')

if __name__=='__main__':
    main()