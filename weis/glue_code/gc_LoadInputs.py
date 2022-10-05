import os
import os.path as osp
import platform
import multiprocessing as mp
import weis.inputs as sch
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from weis.dlc_driver.dlc_generator    import DLCGenerator
from wisdem.commonse.mpi_tools              import MPI

class WindTurbineOntologyPythonWEIS(WindTurbineOntologyPython):
    # Pure python class inheriting the class WindTurbineOntologyPython from WISDEM
    # and adding the WEIS options, namely the paths to the WEIS submodules
    # (OpenFAST, ROSCO, TurbSim, XFoil) and initializing the control parameters.

    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.modeling_options['fname_input_modeling'] = fname_input_modeling
        self.wt_init          = sch.load_geometry_yaml(fname_input_wt)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)

        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_openmdao_vectors_control()
        self.set_weis_data()
        self.set_opt_flags()

    def set_weis_data(self):

        # BEM dir, all levels
        base_run_dir = self.modeling_options['General']['openfast_configuration']['OF_run_dir']
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            bemDir = os.path.join(base_run_dir,'rank_%000d'%int(rank),'BEM')
        else:
            bemDir = os.path.join(base_run_dir,'BEM')

        self.modeling_options["Level1"]['BEM_dir'] = bemDir
        if MPI:
            # If running MPI, RAFT won't be able to save designs in parallel
            self.modeling_options["Level1"]['save_designs'] = False
        # Openfast
        if self.modeling_options['Level2']['flag'] or self.modeling_options['Level3']['flag']:
            fast = InputReader_OpenFAST()
            self.modeling_options['General']['openfast_configuration']['fst_vt'] = {}
            self.modeling_options['General']['openfast_configuration']['fst_vt']['outlist'] = fast.fst_vt['outlist']

            # OpenFAST prefixes
            if self.modeling_options['General']['openfast_configuration']['OF_run_fst'] in ['','None','NONE','none']:
                self.modeling_options['General']['openfast_configuration']['OF_run_fst'] = 'weis_job'
                
            if self.modeling_options['General']['openfast_configuration']['OF_run_dir'] in ['','None','NONE','none']:
                self.modeling_options['General']['openfast_configuration']['OF_run_dir'] = osp.join(os.getcwd(), 'openfast_runs')
                
            # Find the path to the WEIS controller
            weis_dir = osp.dirname( osp.dirname( osp.dirname( osp.realpath(__file__) ) ) )
            if platform.system() == 'Windows':
                path2dll = osp.join(weis_dir, 'local','lib','libdiscon.dll')
            elif platform.system() == 'Darwin':
                path2dll = osp.join(weis_dir, 'local','lib','libdiscon.dylib')
            else:
                path2dll = osp.join(weis_dir, 'local','lib','libdiscon.so')

            # User-defined control dylib (path2dll)
            if self.modeling_options['General']['openfast_configuration']['path2dll'] == 'none':   #Default option, use above
                self.modeling_options['General']['openfast_configuration']['path2dll'] = path2dll
            else:
                if not os.path.isabs(self.modeling_options['General']['openfast_configuration']['path2dll']):  # make relative path absolute
                    self.modeling_options['General']['openfast_configuration']['path2dll'] = \
                        os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']), FASTpref['file_management']['FAST_lib'])

            # Activate HAMS in Level1 if requested for Level 2 or 3
            if self.modeling_options["flags"]["offshore"] or self.modeling_options["Level3"]["from_openfast"]:
                if self.modeling_options["Level1"]["potential_model_override"] == 2:
                    self.modeling_options["Level3"]["HydroDyn"]["PotMod"] = 1
                elif ( (self.modeling_options["Level1"]["potential_model_override"] == 0) and
                       (len(self.modeling_options["Level1"]["potential_bem_members"]) > 0) ):
                    self.modeling_options["Level3"]["HydroDyn"]["PotMod"] = 1
                elif self.modeling_options["Level1"]["potential_model_override"] == 1:
                    self.modeling_options["Level3"]["HydroDyn"]["PotMod"] = 0
                else:
                    # Keep user defined value of PotMod
                    pass

                if self.modeling_options["Level3"]["HydroDyn"]["PotMod"] == 1:

                    # If user requested PotMod but didn't specify any override or members, just run everything
                    if ( (self.modeling_options["Level1"]["potential_model_override"] == 0) and
                       (len(self.modeling_options["Level1"]["potential_bem_members"]) == 0) ):
                        self.modeling_options["Level1"]["potential_model_override"] == 2
                        
                    cwd = os.getcwd()
                    weis_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
                    potpath = self.modeling_options["Level3"]["HydroDyn"]["PotFile"].replace('.hst','').replace('.12','').replace('.3','').replace('.1','')
                    if ( (len(potpath) == 0) or (potpath.lower() in ['unused','default','none']) ):
                        
                        self.modeling_options['Level1']['flag'] = True
                        self.modeling_options["Level3"]["HydroDyn"]["PotFile"] = osp.join(cwd, bemDir,'Output','Wamit_format','Buoy')
                        

                    else:
                        if self.modeling_options['Level1']['runPyHAMS']:
                            print('Found existing potential model: {}\n    - Trying to use this instead of running PyHAMS.'.format(potpath))
                            self.modeling_options['Level1']['runPyHAMS'] = False
                        if osp.exists( potpath+'.1' ):
                            self.modeling_options["Level3"]["HydroDyn"]["PotFile"] = osp.realpath(potpath)
                        elif osp.exists( osp.join(cwd, potpath+'.1') ):
                            self.modeling_options["Level3"]["HydroDyn"]["PotFile"] = osp.realpath( osp.join(cwd, potpath) )
                        elif osp.exists( osp.join(weis_dir, potpath+'.1') ):
                            self.modeling_options["Level3"]["HydroDyn"]["PotFile"] = osp.realpath( osp.join(weis_dir, potpath) )
                        else:
                            raise Exception(f'No valid Wamit-style output found for specified PotFile option, {potpath}.1')

        # RAFT
        if self.modeling_options["flags"]["floating"]:
            bool_init = True if self.modeling_options["Level1"]["potential_model_override"]==2 else False
            self.modeling_options["Level1"]["model_potential"] = [bool_init] * self.modeling_options["floating"]["members"]["n_members"]

            if self.modeling_options["Level1"]["potential_model_override"] == 0:
                for k in self.modeling_options["Level1"]["potential_bem_members"]:
                    idx = self.modeling_options["floating"]["members"]["name"].index(k)
                    self.modeling_options["Level1"]["model_potential"][idx] = True
        elif self.modeling_options["flags"]["offshore"]:
            self.modeling_options["Level1"]["model_potential"] = [False]*1000
            
        # ROSCO
        self.modeling_options['ROSCO']['flag'] = (self.modeling_options['Level1']['flag'] or
                                                  self.modeling_options['Level2']['flag'] or
                                                  self.modeling_options['Level3']['flag'])
        
        # XFoil
        if not osp.isfile(self.modeling_options['Level3']["xfoil"]["path"]) and self.modeling_options['ROSCO']['Flp_Mode']:
            raise Exception("A distributed aerodynamic control device is defined in the geometry yaml, but the path to XFoil in the modeling options is not defined correctly")

        # Compute the number of DLCs that will be run
        DLCs = self.modeling_options['DLC_driver']['DLCs']
        # Initialize the DLC generator
        cut_in = self.wt_init['control']['supervisory']['Vin']
        cut_out = self.wt_init['control']['supervisory']['Vout']
        metocean = self.modeling_options['DLC_driver']['metocean_conditions']
        dlc_generator = DLCGenerator(cut_in, cut_out, metocean=metocean)
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)
        self.modeling_options['DLC_driver']['n_cases'] = dlc_generator.n_cases
        if hasattr(dlc_generator,'n_ws_dlc11'):
            self.modeling_options['DLC_driver']['n_ws_dlc11'] = dlc_generator.n_ws_dlc11
        else:
            self.modeling_options['DLC_driver']['n_ws_dlc11'] = 0

        self.modeling_options['flags']['TMDs'] = False
        if 'TMDs' in self.wt_init:
            if self.modeling_options['Level3']['flag']:
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
