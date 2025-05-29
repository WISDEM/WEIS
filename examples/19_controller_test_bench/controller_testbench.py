import os, sys
from weis.aeroelasticse.openmdao_openfast import FASTLoadCases
import weis.inputs as sch
from rosco import discon_lib_path
from openmdao.utils.mpi  import MPI
from weis.dlc_driver.dlc_generator    import DLCGenerator
import numpy as np
import logging
import shutil

if MPI:
    from weis.glue_code.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop


this_dir = os.path.dirname( os.path.realpath(__file__) )
logger = logging.getLogger("wisdem/weis")


def load_testbench_yaml(filename):
    # TODO: add testbench schema to modeling schema
    
    
    # Will eventually move this somewhere else for clarity
    options = sch.load_modeling_yaml(filename)


    return options


def main():


    input_file = 'testbench_options_5mw.yaml'
    testbench_options = load_testbench_yaml(os.path.join(this_dir,input_file))

    #### NOTHING BELOW HERE SHOULD CHANGE FOR THE USER
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Configure modeling options for FASTLoadCases 
    modopt_file = os.path.join(this_dir,input_file)
    testbench_options['General']['openfast_configuration']['fst_vt'] = {}
    testbench_options['fname_input_modeling'] = modopt_file
    testbench_options['materials'] = {}

    # Unpack testbench inputs for openmdao_openfast
    inputs = {
        'V_cutin':          testbench_options['Turbine_Info']['wind_speed_cut_in'],
        'V_cutout':         testbench_options['Turbine_Info']['wind_speed_cut_out'],
        'Vrated':           testbench_options['Turbine_Info']['wind_speed_rated'],
        'hub_height':       testbench_options['Turbine_Info']['hub_height'],
        'Rtip':             testbench_options['Turbine_Info']['rotor_radius'],
        'shearExp': 0.14,   # Not used, hard code
        'lifetime': 25.,   # Not used, hard code
    }
    inputs= {key: np.array([value]) for key, value in inputs.items()}   # make all

    discrete_inputs = {
        'turbine_class': testbench_options['Turbine_Info']['turbine_class'],
        'turbulence_class': testbench_options['Turbine_Info']['turbulence_class'],
    }

    testbench_options['flags'] = testbench_options['Turbine_Info']['flags']  # Used to determine output channels in OpenFAST


    # Usually, in WEIS, we populate fst_vt['DISCON'] with previously tuned ROSCO.  
    # This will just use the DISCON referenced by the OpenFAST model. 
    # TODO: should we do any tuning or specify a DISCON?
    testbench_options['ROSCO']['flag'] = False   

    # OpenFAST: should always use pre-existing input set
    testbench_options['OpenFAST']['flag'] = True
    testbench_options['OpenFAST']['from_openfast'] = True

    OFmgmt = testbench_options['General']['openfast_configuration']
    OFmgmt['cores'] = testbench_options['Testbench_Options'].get('n_cores', 1)
    OFmgmt['use_exe'] = True
    OFmgmt['allow_fails'] = True    

    OFmgmt['FAST_exe'] = testbench_options['Testbench_Options'].get('FAST_exe',shutil.which('openfast'))
    OFmgmt['turbsim_exe'] = testbench_options['Testbench_Options'].get('turbsim_exe',shutil.which('turbsim'))

    OFmgmt['write_stdout'] = testbench_options['OpenFAST'].get('write_stdout', False)
    
    # Controller inputs
    if 'path2dll' in testbench_options['Controller']:
        OFmgmt['path2dll'] = testbench_options['Controller']['path2dll']
    else:
        logger.warning('No path2dll specified in testbench_options.yaml. Using default rosco path to dll.')

    if 'DISCON_in' in testbench_options['Controller']:
        OFmgmt['DISCON_in'] = testbench_options['Controller']['DISCON_in']
        OFmgmt['DISCON_in'] = os.path.join(os.path.dirname(modopt_file), OFmgmt['DISCON_in'])
        if not os.path.isfile(OFmgmt['DISCON_in']):
            raise FileNotFoundError(f"DISCON_in file not found: {OFmgmt['DISCON_in']}")
    else:
        logger.warning('No DISCON_in specified in testbench_options.yaml. Using DISCON input defined in OpenFAST input set.')

    # Set default directories relative to testbench options
    OFmgmt['OF_run_fst'] = 'testbench'
    OFmgmt['OF_run_dir'] = os.path.join(os.path.dirname(modopt_file), testbench_options['Testbench_Options']['output_directory'])
    testbench_options['OpenFAST']['openfast_dir'] = os.path.join(os.path.dirname(modopt_file),testbench_options['OpenFAST']['openfast_dir'])

    # Postprocessing options (map to OFMgmt)
    OFmgmt['postprocessing'] = testbench_options['PostProcessing']

    # Figure out how many cases we're running
    dlc_generator = DLCGenerator(
        metocean = testbench_options['DLC_driver']['metocean_conditions'], 
        dlc_driver_options = testbench_options['DLC_driver'],
        )
    DLCs = testbench_options['DLC_driver']['DLCs']
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)
    n_OF_runs = dlc_generator.n_cases

    if MPI:
        opt_options = {}
        opt_options['driver'] = {}
        opt_options['driver']['design_of_experiments'] = {}
        opt_options['driver']['design_of_experiments']['flag'] = False

        available_cores = MPI.COMM_WORLD.Get_size()
        n_parallel_OFruns = min([available_cores - 1, n_OF_runs])
        comm_map_down, comm_map_up, _ = map_comm_heirarchical(1, n_parallel_OFruns)

        OFmgmt['mpi_run'] = True
        OFmgmt['mpi_comm_map_down'] = comm_map_down

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # other ranks wait for later
        if rank in comm_map_up.keys():
            subprocessor_loop(comm_map_up)

    else:
        opt_options = {}
        rank = 0

    # Make FASTLoadCases
    if rank == 0:
        logging.info('Running controller testbench with input file: %s', input_file)
        logger.info('Using DISCON_in: %s', OFmgmt['DISCON_in'])
        sys.stdout.flush()
        flc = FASTLoadCases()
        flc.options['modeling_options'] = testbench_options
        flc.options['opt_options'] = opt_options
        flc.n_blades = 3
        flc.of_inumber = -1

        flc.setup_directories()
        flc.modopt_dir = os.path.dirname(flc.options['modeling_options']['fname_input_modeling'])

        fst_vt = flc.create_fst_vt(inputs, discrete_inputs)

        case_list, case_name, dlc_generator  = flc.run_FAST(inputs, discrete_inputs, fst_vt)

        # Post-processing here
        outputs = {}
        discrete_outputs = {}
        flc.post_process(case_list, case_name, dlc_generator, inputs, discrete_inputs, outputs, discrete_outputs)


    # Close signal to subprocessors
    if rank == 0 and MPI:
        subprocessor_stop(comm_map_down)
    sys.stdout.flush()

if __name__=='__main__':
    main()
