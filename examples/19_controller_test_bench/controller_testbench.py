import os, sys
from weis.aeroelasticse.openmdao_openfast import FASTLoadCases
import weis.inputs as sch
from rosco import discon_lib_path
from openmdao.utils.mpi  import MPI
from weis.dlc_driver.dlc_generator    import DLCGenerator

if MPI:
    from weis.glue_code.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop


this_dir = os.path.dirname( os.path.realpath(__file__) )

def load_testbench_yaml(filename):
    # TODO: add testbench schema to modeling schema
    
    
    # Will eventually move this somewhere else for clarity
    options = sch.load_modeling_yaml(filename)


    return options


def main():


    input_file = 'testbench_options_lite.yaml'
    testbench_options = load_testbench_yaml(os.path.join(this_dir,input_file))

    #### NOTHING BELOW HERE SHOULD CHANGE FOR THE USER


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
    }

    discrete_inputs = {
        'turbine_class': testbench_options['Turbine_Info']['turbine_class'],
        'turbulence_class': testbench_options['Turbine_Info']['turbulence_class'],
    }

    testbench_options['flags'] = testbench_options['Turbine_Info']['flags']  # Used to determine output channels in OpenFAST


    # Usually, in WEIS, we populate fst_vt['DISCON'] with previously tuned ROSCO.  
    # This will just use the DISCON referenced by the OpenFAST model. 
    # TODO: should we do any tuning or specify a DISCON?
    testbench_options['ROSCO']['flag'] = False   

    OFmgmt = testbench_options['General']['openfast_configuration']
    OFmgmt['cores'] = testbench_options['Testbench_Options']['n_cores']
    OFmgmt['path2dll'] = discon_lib_path

    # Set default directories relative to testbench options
    OFmgmt['OF_run_fst'] = testbench_options['Testbench_Options']['output_filebase']
    OFmgmt['OF_run_dir'] = os.path.join(os.path.dirname(modopt_file), testbench_options['Testbench_Options']['output_directory'])
    testbench_options['Level3']['openfast_dir'] = os.path.join(os.path.dirname(modopt_file),testbench_options['Level3']['openfast_dir'])


    if MPI:
        opt_options = {}
        opt_options['driver'] = {}
        opt_options['driver']['design_of_experiments'] = {}
        opt_options['driver']['design_of_experiments']['flag'] = False

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
        print('Running FASTLoadCases')
        sys.stdout.flush()
        flc = FASTLoadCases()
        flc.options['modeling_options'] = testbench_options
        flc.options['opt_options'] = opt_options
        flc.n_blades = 3
        flc.of_inumber = -1

        flc.setup_directories()
        fst_vt = flc.create_fst_vt(inputs, discrete_inputs)

        summary_stats, extreme_table, DELs, Damage, case_list, case_name, chan_time, dlc_generator  = flc.run_FAST(inputs, discrete_inputs, fst_vt)

        # Post-processing here
        outputs = {}
        discrete_outputs = {}
        flc.post_process(summary_stats, extreme_table, DELs, Damage, case_list, case_name, dlc_generator, chan_time, inputs, discrete_inputs, outputs, discrete_outputs)

        print('here')

    # Close signal to subprocessors
    if rank == 0 and MPI:
        subprocessor_stop(comm_map_down)
    sys.stdout.flush()

if __name__=='__main__':
    main()