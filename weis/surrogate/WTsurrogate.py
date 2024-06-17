import numpy as np
import csv
import sys
import os
import yaml # temporary
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import KRG

class WindTurbineDOE2SM():

    def __init__(self):
        pass

    def read_doe(self, sql_file, modeling_options, opt_options):

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # construct data structure
        # sql_files = [f for f in os.listdir(os.curdir) if os.path.splitext(f)[1][:4] == '.sql']
        # sql_file = sql_files[0]
        #
        # sql_file = 'log_opt.sql_0'
        # with open('refturb_output-analysis.yaml', 'rt') as fid:
        #     opt_options = yaml.safe_load(fid)
        # with open('refturb_output-modeling.yaml', 'rt') as fid:
        #     modeling_options = yaml.safe_load(fid)
        # rank = 0

        cr = om.CaseReader(sql_file)
        cases = cr.list_cases('driver')

        if (not MPI) or (MPI and rank == 0):
            case = cases[0]
            inputs = cr.get_case(case).inputs
            outputs = cr.get_case(case).outputs
            input_keys_ref = list(set(inputs.keys()).intersection([
                'floating.member_main_column:outer_diameter',
                'floating.member_column1:outer_diameter',
                'floating.member_column2:outer_diameter',
                'floating.member_column3:outer_diameter',
                'floating.member_Y_pontoon_upper1:outer_diameter',
                'floating.member_Y_pontoon_upper2:outer_diameter',
                'floating.member_Y_pontoon_upper3:outer_diameter',
                'floating.member_Y_pontoon_lower1:outer_diameter',
                'floating.member_Y_pontoon_lower2:outer_diameter',
                'floating.member_Y_pontoon_lower3:outer_diameter',
            ]))
            input_keys_ref.sort()
            output_keys_ref = list(set(outputs.keys()).intersection([
                'tune_rosco_ivc.ps_percent',
                'tune_rosco_ivc.omega_pc',
                'tune_rosco_ivc.zeta_pc',
                'tune_rosco_ivc.Kp_float',
                'tune_rosco_ivc.ptfm_freq',
                'tune_rosco_ivc.omega_vs',
                'tune_rosco_ivc.zeta_vs',
                'configuration.rotor_diameter_user',
                'towerse.tower.fore_aft_freqs', # 123
                'towerse.tower.side_side_freqs', # 123
                'towerse.tower.torsion_freqs', # 123
                'towerse.tower.top_deflection',
                'floatingse.platform_base_F', # xyz
                'floatingse.platform_base_M', # xyz
                'floating.member_main_column:joint1', # xyz
                'floating.member_main_column:joint2', # xyz
                'floating.member_column1:joint1', # xyz
                'floating.member_column1:joint2', # xyz
                'floating.member_column2:joint1', # xyz
                'floating.member_column2:joint2', # xyz
                'floating.member_column3:joint1', # xyz
                'floating.member_column3:joint2', # xyz
                'floating.jointdv_0', # keel z-location
                'floating.jointdv_1', # freeboard z-location
                'floating.jointdv_2', # column123 r-location
                'raft.Max_Offset', # Maximum distance in surge/sway direction [m]
                'raft.heave_avg', # Average heave over all cases [m]
                'raft.Max_PtfmPitch', # Maximum platform pitch over all cases [deg]
                'raft.Std_PtfmPitch', # Average platform pitch std. over all cases [deg]
                'rigid_body_periods', # Rigid body natural period [s]
                'raft.heave_period', # Heave natural period [s]
                'raft.pitch_period', # Pitch natural period [s]
                'raft.roll_period', # Roll natural period [s]
                'raft.surge_period', # Surge natural period [s]
                'raft.sway_period', # Sway natural period [s]
                'raft.yaw_period', # Yaw natural period [s]
                'raft.max_nac_accel', # Maximum nacelle accelleration over all cases [m/s**2]
                'raft.max_tower_base', # Maximum tower base moment over all cases [N*m]
                'raft.platform_total_center_of_mass', # xyz
                'raft.platform_displacement',
                'raft.platform_mass', # Platform mass
                'tcons.tip_deflection_ratio', # Blade tip deflection ratio (constrained to be <=1.0)
                'financese.lcoe', # WEIS LCOE from FinanceSE
                'rotorse.rp.AEP', # WISDEM AEP from RotorSE
                'rotorse.blade_mass', # Blade mass
                'towerse.tower_mass', # Tower mass
                'fixedse.structural_mass', # System structural mass for fixed foundation turbines
                'floatingse.system_structural_mass', # System structural mass for floating turbines
                'floatingse.platform_mass', # Platform mass from FloatingSE
                'floatingse.platform_cost', # Platform cost
                'floatingse.mooring_mass', # Mooring mass
                'floatingse.mooring_cost', # Mooring cost
                'floatingse.structural_frequencies', 
            ]))
            output_keys_ref.sort()
        else:
            input_keys_ref = None
            output_keys_ref = None
        if MPI:
            input_keys_ref = MPI.COMM_WORLD.bcast(input_keys_ref, root=0)
            output_keys_ref = MPI.COMM_WORLD.bcast(output_keys_ref, root=0)

        # Retrieve values to construct dataset list
        dataset = []

        for case in cases:
            inputs = cr.get_case(case).inputs
            outputs = cr.get_case(case).outputs
            var_keys = []
            var_values = []
            for key in input_keys_ref:
                if len(inputs[key]) == 1:
                    var_keys.append(key)
                    var_values.append(inputs[key][0])
                else:
                    for idx in range(len(np.squeeze(inputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        var_values.append(np.squeeze(inputs[key])[idx])
            for key in output_keys_ref:
                if len(outputs[key]) == 1:
                    var_keys.append(key)
                    var_values.append(outputs[key][0])
                else:
                    for idx in range(len(np.squeeze(outputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        var_values.append(np.squeeze(outputs[key])[idx])
            dataset.append(var_values)

        #print('type(dataset) = {:}'.format(type(dataset)))
        #print('len(dataset) = {:}'.format(len(dataset)))
        #print('dataset = ')
        #print(dataset)

        # Gather data values
        if MPI:
            dataset = MPI.COMM_WORLD.gather(dataset, root=0)
            MPI.COMM_WORLD.Barrier()
            if rank==0:
                print(dataset)
            MPI.COMM_WORLD.Barrier()
            dataset = [dp for proc in dataset for dp in proc]


        if rank==0:
            with open('test.csv', 'wt', newline='') as fid:
                dwriter = csv.writer(fid, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for idx in range(len(dataset)):
                    dwriter.writerow(dataset[idx])


        ## DVs and select inputs and outputs from opt_options
        #if opt_options['design_variables']['control']['ps_percent']['flag']:
        #    output_keys.append('tune_rosco_ivc.ps_percent')



class WindTurbineModel(om.ExplicitComponent):

    def setup(self):

        # inputs (DVs, fixed parameters)
        
        pass 
