import os
import os.path as osp
import platform
import weis.inputs as sch
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from weis.dlc_driver.dlc_generator    import DLCGenerator

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
        # Openfast
        if self.modeling_options['Level2']['flag'] or self.modeling_options['Level3']['flag']:
            fast = InputReader_OpenFAST()
            self.modeling_options['DLC_driver']['openfast_file_management']['fst_vt'] = {}
            self.modeling_options['DLC_driver']['openfast_file_management']['fst_vt']['outlist'] = fast.fst_vt['outlist']

            # Find the path to the WEIS controller
            run_dir = osp.dirname( osp.dirname( osp.dirname( osp.realpath(__file__) ) ) )
            if platform.system() == 'Windows':
                path2dll = osp.join(run_dir, 'local','lib','libdiscon.dll')
            elif platform.system() == 'Darwin':
                path2dll = osp.join(run_dir, 'local','lib','libdiscon.dylib')
            else:
                path2dll = osp.join(run_dir, 'local','lib','libdiscon.so')
            self.modeling_options['DLC_driver']['openfast_file_management']['path2dll'] = path2dll

            # Activate HAMS in Level1 if requested for Level 2 or 3
            if self.modeling_options["flags"]["offshore"]:
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
                        self.modeling_options["Level3"]["HydroDyn"]["PotFile"] = osp.join(cwd, 'BEM','Output','Wamit_format','Buoy')

                    else:
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
        dlc_generator = DLCGenerator(cut_in, cut_out)
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)
        self.modeling_options['DLC_driver']['n_cases'] = dlc_generator.n_cases
        self.modeling_options['DLC_driver']['n_ws_dlc11'] = dlc_generator.n_ws_dlc11


    def set_openmdao_vectors_control(self):
        # Distributed aerodynamic control devices along blade
        self.modeling_options['WISDEM']['RotorSE']['n_te_flaps']      = 0
        if 'aerodynamic_control' in self.wt_init['components']['blade']:
            if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                self.modeling_options['WISDEM']['RotorSE']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                self.modeling_options['WISDEM']['RotorSE']['n_tab']   = 3
            else:
                raise Exception('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')

    def update_ontology_control(self, wt_opt):
        # Update controller
        if self.modeling_options['flags']['control']:
            self.wt_init['control']['pitch']['PC_omega'] = float(wt_opt['tune_rosco_ivc.PC_omega'])
            self.wt_init['control']['pitch']['PC_zeta']  = float(wt_opt['tune_rosco_ivc.PC_zeta'])
            self.wt_init['control']['torque']['VS_omega'] = float(wt_opt['tune_rosco_ivc.VS_omega'])
            self.wt_init['control']['torque']['VS_zeta']  = float(wt_opt['tune_rosco_ivc.VS_zeta'])
            if self.modeling_options['ROSCO']['Flp_Mode'] > 0:
                self.wt_init['control']['dac']['Flp_omega']= float(wt_opt['tune_rosco_ivc.Flp_omega'])
                self.wt_init['control']['dac']['Flp_zeta'] = float(wt_opt['tune_rosco_ivc.Flp_zeta'])
            if 'IPC' in self.wt_init['control'].keys():
                self.wt_init['control']['IPC']['IPC_gain_1P'] = float(wt_opt['tune_rosco_ivc.IPC_Ki1p'])


    def write_options(self, fname_output):
        # Override the WISDEM version to ensure that the WEIS options files are written instead
        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)
