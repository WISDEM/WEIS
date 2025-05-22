import os
import os.path as osp
import copy, logging
import numpy as np

from rosco import discon_lib_path
import weis.inputs as sch
from openfast_io.FAST_reader import InputReader_OpenFAST
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from weis.dlc_driver.dlc_generator    import DLCGenerator
from openmdao.utils.mpi import MPI
from rosco.toolbox.inputs.validation import load_rosco_yaml
from wisdem.inputs import load_yaml

logger = logging.getLogger("wisdem/weis")

def update_options(options,override):
    for key, value in override.items():
        if isinstance(value, dict) and key in options:
            update_options(options[key], value)
        elif key in options:
            options[key] = value
        else:
            raise Exception(f'Error updating option overrides. {key} is not part of {options.keys()}')

class WindTurbineOntologyPythonWEIS(WindTurbineOntologyPython):
    # Pure python class inheriting the class WindTurbineOntologyPython from WISDEM
    # and adding the WEIS options, namely the paths to the WEIS submodules
    # (OpenFAST, ROSCO, TurbSim, XFoil) and initializing the control parameters.

    def __init__(
            self, 
            fname_input_wt, 
            fname_input_modeling, 
            fname_input_analysis,
            modeling_override = None,
            analysis_override = None,
            ):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.modeling_options['fname_input_modeling'] = fname_input_modeling
        self.wt_init          = sch.load_geometry_yaml(fname_input_wt)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        self.analysis_options['fname_input_analysis'] = fname_input_analysis

        # Update options to maintain some backwards compatibility
        self.backwards_compatibility()

        if modeling_override:
            update_options(self.modeling_options, modeling_override)
            sch.re_validate_modeling(self.modeling_options)
                
        
        if analysis_override:
            update_options(self.analysis_options, analysis_override)
            sch.re_validate_analysis(self.analysis_options)

        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_openmdao_vectors_control()
        self.set_weis_data()
        self.set_opt_flags()

    def set_weis_data(self):

        # Directory of modeling option input, if we want to use it for relative paths
        mod_opt_dir = osp.dirname(self.modeling_options['fname_input_modeling'])
        ana_opt_dir = osp.dirname(self.analysis_options['fname_input_analysis'])

        # OpenFAST prefixes
        if self.modeling_options['General']['openfast_configuration']['OF_run_fst'] in ['','None','NONE','none']:
            self.modeling_options['General']['openfast_configuration']['OF_run_fst'] = 'weis_job'
            
        if self.modeling_options['General']['openfast_configuration']['OF_run_dir'] in ['','None','NONE','none']:
            self.modeling_options['General']['openfast_configuration']['OF_run_dir'] = osp.join(
                ana_opt_dir,        # If it's a relative path, will be relative to analysis folder_output directory
                self.analysis_options['general']['folder_output'], 
                'openfast_runs'
                )

        # BEM dir, all levels
        base_run_dir = os.path.join(mod_opt_dir,self.modeling_options['General']['openfast_configuration']['OF_run_dir'])
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            bemDir = osp.join(base_run_dir,'rank_%000d'%int(rank),'BEM')
        else:
            bemDir = osp.join(base_run_dir,'BEM')

        self.modeling_options["RAFT"]['BEM_dir'] = bemDir
        if MPI:
            # If running MPI, RAFT won't be able to save designs in parallel
            self.modeling_options["RAFT"]['save_designs'] = False


        # Openfast
        if self.modeling_options['OpenFAST_Linear']['flag'] or self.modeling_options['OpenFAST']['flag']:
            fast = InputReader_OpenFAST()
            self.modeling_options['General']['openfast_configuration']['fst_vt'] = {}
            self.modeling_options['General']['openfast_configuration']['fst_vt']['outlist'] = fast.fst_vt['outlist']

                
            # User-defined control dylib (path2dll)
            path2dll = self.modeling_options['General']['openfast_configuration']['path2dll']
            if path2dll == 'none':   #Default option, use above
                self.modeling_options['General']['openfast_configuration']['path2dll'] = discon_lib_path
            else:
                if not osp.isabs(path2dll):  # make relative path absolute
                    self.modeling_options['General']['openfast_configuration']['path2dll'] = \
                        osp.join(osp.dirname(self.options['modeling_options']['fname_input_modeling']), path2dll)
            path2dll = self.modeling_options['General']['openfast_configuration']['path2dll']
            if not osp.exists( path2dll ):
                raise NameError("Cannot find DISCON library: "+path2dll)

            # Activate HAMS in RAFT if requested for OpenFAST
            if self.modeling_options["flags"]["offshore"] or self.modeling_options["OpenFAST"]["from_openfast"]:
                if self.modeling_options["RAFT"]["potential_model_override"] == 2:
                    self.modeling_options["OpenFAST"]["HydroDyn"]["PotMod"] = 1
                elif ( (self.modeling_options["RAFT"]["potential_model_override"] == 0) and
                       (len(self.modeling_options["RAFT"]["potential_bem_members"]) > 0) ):
                    self.modeling_options["OpenFAST"]["HydroDyn"]["PotMod"] = 1
                elif self.modeling_options["RAFT"]["potential_model_override"] == 1:
                    self.modeling_options["OpenFAST"]["HydroDyn"]["PotMod"] = 0
                else:
                    # Keep user defined value of PotMod
                    pass

                if self.modeling_options["OpenFAST"]["HydroDyn"]["PotMod"] == 1:

                    # If user requested PotMod but didn't specify any override or members, just run everything (potential_model_override = 2)
                    if ( (self.modeling_options["RAFT"]["potential_model_override"] == 0) and
                       (len(self.modeling_options["RAFT"]["potential_bem_members"]) == 0) ):
                        self.modeling_options["RAFT"]["potential_model_override"] = 2
                        
                    cwd = os.getcwd()
                    weis_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
                    potpath = self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"].replace('.hst','').replace('.12','').replace('.3','').replace('.1','')
                    if ( (len(potpath) == 0) or (potpath.lower() in ['unused','default','none']) ):
                        
                        self.modeling_options['RAFT']['flag'] = True
                        self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"] = osp.join(bemDir,'Output','Wamit_format','Buoy')
                        

                    else:
                        if self.modeling_options['RAFT']['runPyHAMS']:
                            print('Found existing potential model: {}\n    - Trying to use this instead of running PyHAMS.'.format(potpath))
                            self.modeling_options['RAFT']['runPyHAMS'] = False
                        if osp.exists( potpath+'.1' ):
                            self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"] = osp.realpath(potpath)
                        elif osp.exists( osp.join(cwd, potpath+'.1') ):
                            self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"] = osp.realpath( osp.join(cwd, potpath) )
                        elif osp.exists( osp.join(weis_dir, potpath+'.1') ):
                            self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"] = osp.realpath( osp.join(weis_dir, potpath) )
                        elif osp.exists( osp.join(mod_opt_dir, potpath+'.1') ):
                            self.modeling_options["OpenFAST"]["HydroDyn"]["PotFile"] = osp.realpath( osp.join(mod_opt_dir, potpath) )
                        else:
                            raise Exception(f'No valid Wamit-style output found for specified PotFile option, {potpath}.1')

        # OpenFAST dir
        if self.modeling_options["OpenFAST"]["from_openfast"]:
            if not osp.isabs(self.modeling_options['OpenFAST']['openfast_dir']):
                # Make relative to modeling options input
                self.modeling_options['OpenFAST']['openfast_dir'] = osp.realpath(osp.join(
                    mod_opt_dir, self.modeling_options['OpenFAST']['openfast_dir'] ))
        
        # BEM dir, all levels
        base_run_dir = os.path.join(mod_opt_dir,self.modeling_options['General']['openfast_configuration']['OF_run_dir'])
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            bemDir = osp.join(base_run_dir,'rank_%000d'%int(rank),'BEM')
        else:
            bemDir = osp.join(base_run_dir,'BEM')

        self.modeling_options["Level1"]['BEM_dir'] = bemDir
        if MPI:
            # If running MPI, RAFT won't be able to save designs in parallel
            self.modeling_options["Level1"]['save_designs'] = False
        
        # RAFT
        if self.modeling_options["flags"]["floating"]:
            bool_init = True if self.modeling_options["RAFT"]["potential_model_override"]==2 else False
            self.modeling_options["RAFT"]["model_potential"] = [bool_init] * self.modeling_options["floating"]["members"]["n_members"]

            if self.modeling_options["RAFT"]["potential_model_override"] == 0:
                for k in self.modeling_options["RAFT"]["potential_bem_members"]:
                    idx = self.modeling_options["floating"]["members"]["name"].index(k)
                    self.modeling_options["RAFT"]["model_potential"][idx] = True
        elif self.modeling_options["flags"]["offshore"]:
            self.modeling_options["RAFT"]["model_potential"] = [False]*1000
            
        # ROSCO
        self.modeling_options['ROSCO']['flag'] = (self.modeling_options['RAFT']['flag'] or
                                                  self.modeling_options['OpenFAST_Linear']['flag'] or
                                                  self.modeling_options['OpenFAST']['flag'])
        
        if self.modeling_options['ROSCO']['tuning_yaml'] != 'none':  # default is empty
            # Make path absolute if not, relative to modeling options input
            if not osp.isabs(self.modeling_options['ROSCO']['tuning_yaml']):
                self.modeling_options['ROSCO']['tuning_yaml'] = osp.realpath(osp.join(
                    mod_opt_dir, self.modeling_options['ROSCO']['tuning_yaml'] ))
                
        # Apply tuning yaml input if available, this needs to be here for sizing tune_rosco_ivc
        if os.path.split(self.modeling_options['ROSCO']['tuning_yaml'])[1] != 'none':  # default is none
            inps = load_rosco_yaml(self.modeling_options['ROSCO']['tuning_yaml'])  # tuning yaml validated in here
            self.modeling_options['ROSCO'].update(inps['controller_params'])

            # Apply changes in modeling options, should have already been validated
            modopts_no_defaults = load_yaml(self.modeling_options['fname_input_modeling'])  
            skip_options = ['tuning_yaml']  # Options to skip loading, tuning_yaml path has been updated, don't overwrite
            for option, value in modopts_no_defaults['ROSCO'].items():
                if option not in skip_options:
                    self.modeling_options['ROSCO'][option] = value
        
        # XFoil
        if not osp.isfile(self.modeling_options['OpenFAST']["xfoil"]["path"]) and self.modeling_options['ROSCO']['Flp_Mode']:
            raise Exception("A distributed aerodynamic control device is defined in the geometry yaml, but the path to XFoil in the modeling options is not defined correctly")

        # Compute the number of DLCs that will be run
        DLCs = self.modeling_options['DLC_driver']['DLCs']
        # Initialize the DLC generator
        cut_in = self.wt_init['control']['supervisory']['Vin']
        cut_out = self.wt_init['control']['supervisory']['Vout']
        metocean = self.modeling_options['DLC_driver']['metocean_conditions']
        dlc_driver_options = self.modeling_options['DLC_driver']
        dlc_generator = DLCGenerator(cut_in, cut_out, dlc_driver_options=dlc_driver_options, metocean=metocean)
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)
        self.modeling_options['DLC_driver']['n_cases'] = dlc_generator.n_cases
        
        # Determine wind speeds that will be used to calculate AEP (using DLC AEP or 1.1)
        DLCs = [i_dlc['DLC'] for i_dlc in self.modeling_options['DLC_driver']['DLCs']]
        if 'AEP' in DLCs:
            DLC_label_for_AEP = 'AEP'
        else:
            DLC_label_for_AEP = '1.1'
        dlc_aep_ws = [c.URef for c in dlc_generator.cases if c.label == DLC_label_for_AEP]
        self.modeling_options['DLC_driver']['n_ws_aep'] = len(np.unique(dlc_aep_ws))

        # TMD modeling
        self.modeling_options['flags']['TMDs'] = False
        if 'TMDs' in self.wt_init:
            if self.modeling_options['OpenFAST']['flag']:
                self.modeling_options['flags']['TMDs'] = True
            else:
                raise Exception("TMDs in Levels 1 and 2 are not supported yet")


    def set_openmdao_vectors_control(self):
        # Distributed aerodynamic control devices along blade
        self.modeling_options['WISDEM']['RotorSE']['n_te_flaps']      = 0
        if 'aerodynamic_control' in self.wt_init['components']['blade']:
            if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                self.modeling_options['WISDEM']['RotorSE']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                self.modeling_options['WISDEM']['RotorSE']['n_tab']   = 3
            else:
                raise Exception('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')

        if 'TMDs' in self.wt_init:
            n_TMDs = len(self.wt_init['TMDs'])
            self.modeling_options['TMDs'] = {}
            self.modeling_options['TMDs']['n_TMDs']                 = n_TMDs
            # TODO: come back and check how many of these need to be modeling options
            self.modeling_options['TMDs']['name']                   = [tmd['name'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['component']              = [tmd['component'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['location']               = [tmd['location'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['mass']                   = [tmd['mass'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['stiffness']              = [tmd['stiffness'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['damping']                = [tmd['damping'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['natural_frequency']      = [tmd['natural_frequency'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['damping_ratio']          = [tmd['damping_ratio'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['X_DOF']                  = [tmd['X_DOF'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['Y_DOF']                  = [tmd['Y_DOF'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['Z_DOF']                  = [tmd['Z_DOF'] for  tmd in self.wt_init['TMDs']]
            self.modeling_options['TMDs']['preload_spring']         = [tmd['preload_spring'] for  tmd in self.wt_init['TMDs']]

            # Check that TMD locations map to somewhere valid (tower or platform member)
            self.modeling_options['TMDs']['num_tower_TMDs'] = 0
            self.modeling_options['TMDs']['num_ptfm_TMDs']  = 0
            
            for i_TMD, component in enumerate(self.modeling_options['TMDs']['component']):
                if self.modeling_options['flags']['floating'] and component in self.modeling_options['floating']['members']['name']:
                    self.modeling_options['TMDs']['num_ptfm_TMDs'] += 1
                elif component == 'tower':
                    self.modeling_options['TMDs']['num_tower_TMDs'] += 1
                else:
                    raise Exception('Invalid TMD component mapping for {} on {}'.format(
                        self.modeling_options['TMDs']['name'][i_TMD],component))      

            # Set TMD group  mapping: list of length n_groups, with i_TMDs in each group
            # Loop through TMD names, assign to own group if not in an analysis group
            if 'TMDs' in self.analysis_options['design_variables']:
                tmd_group_map = []
                tmd_names = self.modeling_options['TMDs']['name']
                
                for i_group, tmd_group in enumerate(self.analysis_options['design_variables']['TMDs']['groups']):
                    tmds_in_group_i = [tmd_names.index(tmd_name) for tmd_name in tmd_group['names']]

                    tmd_group_map.append(tmds_in_group_i)
                
                self.modeling_options['TMDs']['group_mapping'] = tmd_group_map

    def update_ontology_control(self, wt_opt):
        # Update controller
        if self.modeling_options['flags']['control']:
            self.wt_init['control']['pitch']['omega_pc'] = wt_opt['tune_rosco_ivc.omega_pc']
            self.wt_init['control']['pitch']['zeta_pc']  = wt_opt['tune_rosco_ivc.zeta_pc']
            self.wt_init['control']['torque']['omega_vs'] = float(wt_opt['tune_rosco_ivc.omega_vs'])
            self.wt_init['control']['torque']['zeta_vs']  = float(wt_opt['tune_rosco_ivc.zeta_vs'])
            self.wt_init['control']['pitch']['Kp_float']  = float(wt_opt['tune_rosco_ivc.Kp_float'])
            self.wt_init['control']['pitch']['ptfm_freq']  = float(wt_opt['tune_rosco_ivc.ptfm_freq'])
            self.wt_init['control']['IPC']['IPC_Ki_1P'] = float(wt_opt['tune_rosco_ivc.IPC_Kp1p'])
            self.wt_init['control']['IPC']['IPC_Kp_1P'] = float(wt_opt['tune_rosco_ivc.IPC_Ki1p'])
            if self.modeling_options['ROSCO']['Flp_Mode'] > 0:
                self.wt_init['control']['dac']['flp_kp_norm']= float(wt_opt['tune_rosco_ivc.flp_kp_norm'])
                self.wt_init['control']['dac']['flp_tau'] = float(wt_opt['tune_rosco_ivc.flp_tau'])


    def write_options(self, fname_output):
        # Override the WISDEM version to ensure that the WEIS options files are written instead
        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)


    def backwards_compatibility(self):

        modopts_no_defaults = load_yaml(self.modeling_options['fname_input_modeling'])

        if 'Level1' in modopts_no_defaults:
            self.modeling_options['RAFT'] = copy.deepcopy(self.modeling_options['Level1'])
            logger.warning('Level1 is no longer a WEIS modeling option.  Please use RAFT instead.  Level1 will be depreciated in a future release.')

        if 'Level2' in modopts_no_defaults:
            self.modeling_options['OpenFAST_Linear'] = copy.deepcopy(self.modeling_options['Level2'])
            logger.warning('Level2 is no longer a WEIS modeling option.  Please use OpenFAST_Linear instead.  Level2 will be depreciated in a future release.')

        if 'Level3' in modopts_no_defaults:
            self.modeling_options['OpenFAST'] = copy.deepcopy(self.modeling_options['Level3'])
            logger.warning('Level3 is no longer a WEIS modeling option.  Please use OpenFAST instead.  Level3 will be depreciated in a future release.')


