import os
import platform
import weis.inputs as sch
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython


class WindTurbineOntologyPythonWEIS(WindTurbineOntologyPython):
    # Pure python class inheriting the class WindTurbineOntologyPython from WISDEM and adding the WEIS options, namely the paths to the WEIS submodules (OpenFAST, ROSCO, TurbSim, XFoil) and initializing the control parameters.
    
    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.modeling_options['fname_input_modeling'] = fname_input_modeling
        self.wt_init          = sch.load_geometry_yaml(fname_input_wt)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        
        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_openmdao_vectors_control()
        self.set_openfast_data()
        self.set_opt_flags()
    
    def set_openfast_data(self):
        # Openfast
        if self.modeling_options['Level3']['flag'] == True:
            fast                = InputReader_OpenFAST(FAST_ver='OpenFAST')
            self.modeling_options['openfast']['fst_vt'] = {}
            self.modeling_options['openfast']['fst_vt']['outlist'] = fast.fst_vt['outlist']

            if self.modeling_options['openfast']['file_management']['FAST_directory'] != 'none':
                # Load Input OpenFAST model variable values
                fast.FAST_InputFile = self.modeling_options['openfast']['file_management']['FAST_InputFile']
                if os.path.isabs(self.modeling_options['openfast']['file_management']['FAST_directory']):
                    fast.FAST_directory = self.modeling_options['openfast']['file_management']['FAST_directory']
                else:
                    fast.FAST_directory = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['openfast']['file_management']['FAST_directory'])

            # Find the controller
            run_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
            if platform.system() == 'Windows':
                path2dll = os.path.join(run_dir, 'local/lib/libdiscon.dll')
            elif platform.system() == 'Darwin':
                path2dll = os.path.join(run_dir, 'local/lib/libdiscon.dylib')
            else:
                path2dll = os.path.join(run_dir, 'local/lib/libdiscon.so')
            if self.modeling_options['openfast']['file_management']['path2dll'] == 'none':
                self.modeling_options['openfast']['file_management']['path2dll'] = path2dll
                
            if os.path.isabs(self.modeling_options['openfast']['file_management']['path2dll']) == False:
                self.modeling_options['openfast']['file_management']['path2dll'] = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['openfast']['file_management']['path2dll'])

            if self.modeling_options['openfast']['file_management']['FAST_directory'] != 'none':
                fast.path2dll = self.modeling_options['openfast']['file_management']['path2dll']
                fast.execute()
            
            if self.modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2 and self.modeling_options['openfast']['dlc_settings']['run_power_curve'] == False and self.modeling_options['openfast']['dlc_settings']['run_IEC'] == False:
                raise Exception('WEIS is set to run OpenFAST, but both flags for power curve and IEC cases are set to False among the modeling options. Set at least one of the two to True to proceed.')
        
        # XFoil
        if not os.path.isfile(self.modeling_options["xfoil"]["path"]) and self.modeling_options['Level3']['ROSCO']['Flp_Mode']:
            raise Exception("A distributed aerodynamic control device is defined in the geometry yaml, but the path to XFoil in the modeling options is not defined correctly")

            
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
            if self.modeling_options['Level3']['ROSCO']['Flp_Mode'] > 0:
                self.wt_init['control']['dac']['Flp_omega']= float(wt_opt['tune_rosco_ivc.Flp_omega'])
                self.wt_init['control']['dac']['Flp_zeta'] = float(wt_opt['tune_rosco_ivc.Flp_zeta'])
            if 'IPC' in self.wt_init['control'].keys():
                self.wt_init['control']['IPC']['IPC_gain_1P'] = float(wt_opt['tune_rosco_ivc.IPC_Ki1p'])

        
    def write_options(self, fname_output):
        # Override the WISDEM version to ensure that the WEIS options files are written instead
        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)
            
