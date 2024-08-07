import numpy as np
import csv
import os
import time
import re
import pickle as pkl
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import SGP

class WindTurbineDOE2SM():
    # Read DOE sql files, Process data, Create and train surrogate models, and Save them in smt file.

    def __init__(self):
        self._doe_loaded = False
        self._sm_trained = False

    def read_doe(self, sql_file, modeling_options, opt_options):
        # Read DOE sql files. If MPI, read them in parallel.

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

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
            input_keys_dv = self._identify_dv(input_keys_ref, opt_options, inputs, outputs)
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
                #'towerse.tower_mass', # Tower mass
                'fixedse.structural_mass', # System structural mass for fixed foundation turbines
                'floatingse.system_structural_mass', # System structural mass for floating turbines
                'floatingse.platform_mass', # Platform mass from FloatingSE
                'floatingse.platform_cost', # Platform cost
                #'floatingse.mooring_mass', # Mooring mass
                #'floatingse.mooring_cost', # Mooring cost
                'floatingse.structural_frequencies', 
            ]))
            output_keys_ref.sort()
            output_keys_dv = self._identify_dv(output_keys_ref, opt_options, inputs, outputs)
        else:
            input_keys_ref = None
            input_keys_dv = None
            output_keys_ref = None
            output_keys_dv = None
        if MPI:
            input_keys_ref = MPI.COMM_WORLD.bcast(input_keys_ref, root=0)
            input_keys_dv = MPI.COMM_WORLD.bcast(input_keys_dv, root=0)
            output_keys_ref = MPI.COMM_WORLD.bcast(output_keys_ref, root=0)
            output_keys_dv = MPI.COMM_WORLD.bcast(output_keys_dv, root=0)

        # Retrieve values to construct dataset list
        dataset = []

        for case in cases:
            inputs = cr.get_case(case).inputs
            outputs = cr.get_case(case).outputs
            var_keys = []
            var_dv = []
            var_values = []
            for key in input_keys_ref:
                if len(inputs[key]) == 1:
                    var_keys.append(key)
                    try:
                        dvidx = input_keys_dv[input_keys_ref.index(key)]
                        if ((type(dvidx) == bool) and (dvidx == False)) or \
                                ((type(dvidx) == list) and (len(dvidx) == 0)):
                            var_dv.append(False)
                        elif ((type(dvidx) == list) and (len(dvidx) == 1)):
                            var_dv.append(True)
                        else:
                            raise Exception
                    except:
                        var_dv.append(False)
                    var_values.append(inputs[key][0])
                else:
                    for idx in range(len(np.squeeze(inputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        try:
                            dvidx = input_keys_dv[input_keys_ref.index(key)]
                            if ((type(dvidx) == bool) and (dvidx == False)) or \
                                    ((type(dvidx) == list) and (len(dvidx) == 0)):
                                var_dv.append(False)
                            elif ((type(dvidx) == list) and (len(dvidx) > 0)):
                                for jdx in range(len(dvidx)):
                                    var_dv_append = False
                                    if idx == dvidx[jdx]:
                                        var_dv_append = True
                                var_dv.append(var_dv_append)
                            else:
                                raise Exception
                        except:
                            var_dv.append(False)
                        var_values.append(np.squeeze(inputs[key])[idx])
            for key in output_keys_ref:
                if len(outputs[key]) == 1:
                    var_keys.append(key)
                    try:
                        dvidx = output_keys_dv[output_keys_ref.index(key)]
                        if ((type(dvidx) == bool) and (dvidx == False)) or \
                                ((type(dvidx) == list) and (len(dvidx) == 0)):
                            var_dv.append(False)
                        elif ((type(dvidx) == list) and (len(dvidx) == 1)):
                            var_dv.append(True)
                        else:
                            raise Exception
                    except:
                        var_dv.append(False)
                    var_values.append(outputs[key][0])
                else:
                    for idx in range(len(np.squeeze(outputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        try:
                            dvidx = output_keys_dv[output_keys_ref.index(key)]
                            if ((type(dvidx) == bool) and (dvidx == False)) or \
                                    ((type(dvidx) == list) and (len(dvidx) == 0)):
                                var_dv.append(False)
                            elif ((type(dvidx) == list) and (len(dvidx) > 0)):
                                for jdx in range(len(dvidx)):
                                    var_dv_append = False
                                    if idx == dvidx[jdx]:
                                        var_dv_append = True
                                var_dv.append(var_dv_append)
                            else:
                                raise Exception
                        except:
                            var_dv.append(False)
                        var_values.append(np.squeeze(outputs[key])[idx])
            dataset.append(var_values)

        # Gather data values
        if MPI:
            dataset = MPI.COMM_WORLD.gather(dataset, root=0)
        else:
            dataset = [dataset]
        if rank == 0:
            dataset = np.array([dp for proc in dataset for dp in proc])
            # Remove duplicated columns
            flag = dataset.shape[1]*[True]
            for idx in range(1,dataset.shape[1]):
                for jdx in range(idx):
                    if (np.array_equal(dataset[:,jdx], dataset[:,idx])):
                        flag[idx] = False
                        if flag[jdx] == True:
                            var_keys[jdx] += ('+' + var_keys[idx])
            data_vals = np.zeros(shape=(dataset.shape[0],0))
            data_dv = []
            data_keys = []
            for idx in range(dataset.shape[1]):
                if flag[idx]:
                    data_vals = np.concatenate(
                            (data_vals, dataset[:,idx].reshape(len(dataset[:,idx]),1)),
                            axis=1)
                    data_dv.append(var_dv[idx])
                    data_keys.append(var_keys[idx])
        else:
            dataset = None
            data_vals = None
            data_dv = None
            data_keys = None

        # Store data
        self.data_vals = data_vals
        self.data_dv = data_dv
        self.data_keys = data_keys
        self._doe_loaded = True

        if rank==0:
            fname = os.path.join(
                opt_options['general']['folder_output'],
                os.path.splitext(opt_options['recorder']['file_name'])[0]+'_doedata.pkl'
            )
            print('Saving DOE data into: {:}'.format(fname))
            with open(fname, 'wb') as fid:
                pkl.dump(
                    {
                        'data_vals': data_vals,
                        'data_dv': data_dv,
                        'data_keys': data_keys,
                    },
                    fid, protocol=5,
                )
            fname = os.path.join(
                opt_options['general']['folder_output'],
                os.path.splitext(opt_options['recorder']['file_name'])[0]+'_doedata.csv'
            )
            print('Saving DOE data into {:}'.format(fname))
            with open(fname, 'wt', newline='') as fid:
                dwriter = csv.writer(fid, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                dwriter.writerow(data_keys)
                dwriter.writerow(data_dv)
                for idx in range(data_vals.shape[0]):
                    if np.iscomplex(data_vals[idx,:]).any():
                        raise Exception('Complex number detected from idx = {:}, key = {:}'.format(idx, data_keys[idx]))
                    else:
                        dwriter.writerow(data_vals[idx,:])



    def _identify_dv(self, keys, opt_options, inputs, outputs):
        # Identify design variables to set them as input parameters in the surrogate model
        # if they are flagged as DV in the analysis_options.yaml file

        dvflag = len(keys)*[False]

        if opt_options['design_variables']['control']['ps_percent']['flag']:
            for key in ['tune_rosco_ivc.ps_percent']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['omega']['flag']:
            for key in ['tune_rosco_ivc.omega_pc']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['zeta']['flag']:
            for key in ['tune_rosco_ivc.zeta_pc']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['Kp_float']['flag']:
            for key in ['tune_rosco_ivc.Kp_float']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['ptfm_freq']['flag']:
            for key in ['tune_rosco_ivc.ptfm_freq']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['torque_control']['omega']['flag']:
            for key in ['tune_rosco_ivc.omega_vs']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['torque_control']['zeta']['flag']:
            for key in ['tune_rosco_ivc.zeta_vs']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['floating']['members']['flag']:
            groups = opt_options['design_variables']['floating']['members']['groups']
            for group in groups:
                names = group['names']
                for key in keys:
                    for name in names:
                        txt = name + ':outer_diameter'
                        ltxt = len(txt)
                        if key[-ltxt:] == txt:
                            try:
                                idx = keys.index(key)
                                if group['diameter']['constant']:
                                    jdx = 0
                                else:
                                    jdx = True
                                if not dvflag[idx]:
                                    dvflag[idx] = [jdx]
                                else:
                                    dvflag[idx].append(jdx)
                            except:
                                pass

        if opt_options['design_variables']['floating']['joints']['flag']:
            key_prefix = 'floating.jointdv_'
            for key in keys:
                if key[:len(key_prefix)] == key_prefix:
                    try:
                        idx = keys.index(key)
                        if not dvflag[idx]:
                            dvflag[idx] = [0]
                        else:
                            dvflag[idx].append(0)
                    except:
                        pass

        if opt_options['design_variables']['rotor_diameter']['flag']:
            for key in ['configuration.rotor_diameter_user']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        return dvflag


    def write_sm(self, sm_filename):
        # Write trained surrogate models in the smt file in pickle format

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else:
            rank = 0
            n_cores = 1

        # File writing only on rank=0
        if rank == 0:
            if not self._doe_loaded:
                raise Exception('DOE data needs to be loaded first.')
            if not self._sm_trained:
                raise Exception('SM needs to be trained before saving to file.')

            try:
                with open(sm_filename, 'wb') as fid:
                    pkl.dump(self.dataset_list, fid, protocol=5)
            except:
                print('Unable to write surrogate model file: {:}.'.format(sm_filename))
                raise Exception('Unable to write surrogate model file: {:}.'.format(sm_filename))


    def train_sm(self):
        # Surrogate model training

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else:
            rank = 0
            n_cores = 1

        # Prepare dataset columns individually (and split if MPI)
        if (not MPI) or (MPI and rank == 0):
            data_vals = self.data_vals
            data_dv = self.data_dv
            data_keys = self.data_keys

            # Number of design variables (n_dv), output variables (n_out), and overall (n_var)
            n_dv = data_dv.count(True)
            n_out = len(data_dv) - n_dv
            n_var = n_dv + n_out
            n_data = data_vals.shape[0]
            # Lists of indexes for design variables (i_dv) and output variables (i_out)
            i_dv = [i for i, e in enumerate(data_dv) if e == True]
            i_out = [i for i, e in enumerate(data_dv) if not e == True]

            # Data arrays and their list
            data_arr_dv = np.zeros(shape=(n_data, n_dv), dtype=float)
            data_lst_out = list(n_out*[None])
            data_dv_keys = list(n_dv*[None])
            data_out_keys = list(n_out*[None])
            data_dv_bounds = list(n_dv*[None])
            data_out_bounds = list(n_out*[None])
            # DVs
            for idx in range(len(i_dv)):
                jdx = i_dv[idx]
                vals = data_vals[:,jdx]
                bounds = [min(vals), max(vals)]
                if ((bounds[1] - bounds[0]) < 10.0*np.finfo(np.float64).eps):
                    bounds[1] = 1.0
                    bounds[0] = 0.0
                data_arr_dv[:,idx] = (vals - bounds[0])/(bounds[1] - bounds[0])
                data_dv_keys[idx] = data_keys[jdx]
                data_dv_bounds[idx] = bounds
            data_dv_bounds = np.array(data_dv_bounds).T
            # Output variables
            for idx in range(len(i_out)):
                jdx = i_out[idx]
                vals = np.array(data_vals[:,jdx]).reshape(n_data, 1)
                bounds = [min(vals)[0], max(vals)[0]]
                if ((bounds[1] - bounds[0]) < 10.0*np.finfo(np.float64).eps):
                    bounds[1] = 1.0
                    bounds[0] = 0.0
                data_lst_out[idx] = (vals - bounds[0])/(bounds[1] - bounds[0])
                data_out_keys[idx] = data_keys[jdx]
                data_out_bounds[idx] = np.array(bounds)

            # Construct data structure for parallel distribution
            dataset_list = []
            for idx in range(len(data_out_keys)):
                data_entry = {
                    'inputs': {
                        'keys': data_dv_keys,
                        'vals': data_arr_dv,
                        'bounds': data_dv_bounds,
                    },
                    'outputs': {
                        'keys': data_out_keys[idx],
                        'vals': data_lst_out[idx],
                        'bounds': data_out_bounds[idx],
                    },
                    'surrogate': None
                }
                dataset_list.append(data_entry)
            
            # Distribute for parallel training
            if MPI:
                dataset_lists = list(self._split_list_chunks(dataset_list, n_cores))
                if len(dataset_lists) < n_cores:
                    for _ in range(0, n_cores - len(dataset_lists)):
                        dataset_lists.append([])
            else:
                dataset_lists = [dataset_list]
        else:
            dataset_list = []
            dataset_lists = []
        # Scatter data to train
        if MPI:
            MPI.COMM_WORLD.barrier()
            dataset_list = MPI.COMM_WORLD.scatter(dataset_lists, root=0)

        # Train SM
        for data_entry in dataset_list:
            t = time.time()
            try:
                outvalarr = data_entry['outputs']['vals']
                outvalavg = np.mean(outvalarr)
                # If response is constant
                if np.all(np.isclose(outvalarr, outvalavg, atol=np.abs(outvalavg)*1.0e-3)):
                    outvalconst = True
                    R_squared = 1.0
                else:
                    outvalconst = False
                    n_data = data_entry['inputs']['vals'].shape[0]
                    # If number of sample data is enough
                    if n_data > 30:
                        n_data_80 = int(0.8*float(n_data))
                        # Train the validation SM
                        smval = SGP_WT(eval_noise=True, print_global=False)
                        smval.set_training_values(
                            data_entry['inputs']['vals'][:n_data_80,:], data_entry['outputs']['vals'][:n_data_80,:])
                        smval.train()
                        smval.set_bounds(
                            data_entry['inputs']['bounds'], data_entry['outputs']['bounds'])
                        smval.keys_in = data_entry['inputs']['keys']
                        for idx in range(len(smval.keys_in)):
                            s = smval.keys_in[idx]
                            i = re.search('[^+]*', s).span()
                            smval.keys_in[idx] = s[i[0]:i[1]]
                        # Validate SM
                        x_val = data_entry['inputs']['vals'][n_data_80:,:]
                        y_val_true = data_entry['outputs']['vals'][n_data_80:,:]
                        lb_val_in = np.tile(smval.bounds_in[0,:], (x_val.shape[0], 1))
                        ub_val_in = np.tile(smval.bounds_in[1,:], (x_val.shape[0], 1))
                        lb_val_out = np.tile(smval.bounds_out[0], (x_val.shape[0], 1))
                        ub_val_out = np.tile(smval.bounds_out[1], (x_val.shape[0], 1))
                        x_val = lb_val_in + (ub_val_in - lb_val_in)*x_val
                        y_val_true = lb_val_out + (ub_val_out - lb_val_out)*y_val_true
                        y_val_pred, v_val_pred = smval.predict(x_val)
                        SSE = np.sum((y_val_pred - y_val_true)**2)
                        SST = np.sum((y_val_true - np.mean(y_val_true))**2)
                        R_squared = 1.0 - SSE/SST
                        print('rank {:}, R_squared: {:}'.format(rank, R_squared))
                        if R_squared <= 0.25:
                            outvalconst = True
                            outvalavg = np.mean(outvalarr)
                            R_squared = 0.0
                    else:
                        R_squared = None
                # Train the full SM
                data_entry['surrogate'] = SGP_WT(eval_noise=True, print_global=False)
                data_entry['surrogate'].set_training_values(
                        data_entry['inputs']['vals'], data_entry['outputs']['vals'])
                if outvalconst:
                    data_entry['surrogate'].constant = True
                    data_entry['surrogate'].constant_value = outvalavg
                else:
                    data_entry['surrogate'].train()
                data_entry['surrogate'].set_bounds(
                        data_entry['inputs']['bounds'], data_entry['outputs']['bounds'])
                data_entry['surrogate'].keys_in = data_entry['inputs']['keys']
                for idx in range(len(data_entry['surrogate'].keys_in)):
                    s = data_entry['surrogate'].keys_in[idx]
                    i = re.search('[^+]*', s).span()
                    data_entry['surrogate'].keys_in[idx] = s[i[0]:i[1]]
                data_entry['surrogate'].keys_out = data_entry['outputs']['keys']
                s = data_entry['surrogate'].keys_out
                i = re.search('[^+]*', s).span()
                data_entry['surrogate'].keys_out = s[i[0]:i[1]]
                data_entry['surrogate'].R_squared = R_squared
            except:
                print('rank {:}, Surrogate model training failed.'.format(rank))
                raise Exception('rank {:}, Surrogate model training failed.'.format(rank))
            t = time.time() - t
            print('rank {:}, Surrogate model training done. Time (sec): {:}'.format(rank, t))


        # Gather trained data
        if MPI:
            MPI.COMM_WORLD.barrier()
            dataset_lists = MPI.COMM_WORLD.gather(dataset_list, root=0)
            if rank == 0:
                dataset_list = [item for items in dataset_lists for item in items]
            else:
                dataset_list = []

        self.dataset_list = dataset_list
        self._sm_trained = True



    def _split_list_chunks(self, fulllist, max_n_chunk=1, item_count=None):
        # Split items in a list into nested multiple (max_n_chunk) lists in an outer list
        # This method is useful to divide jobs for parallel surrogate model training as the number
        # of surrogate models trained are generally not equal to the number of parallel cores used.

        item_count = item_count or len(fulllist)
        n_chunks = min(item_count, max_n_chunk)
        fulllist = iter(fulllist)
        floor = item_count // n_chunks
        ceiling = floor + 1
        stepdown = item_count % n_chunks
        for x_i in range(n_chunks):
            length = ceiling if x_i < stepdown else floor
            yield [next(fulllist) for _ in range(length)]


class SGP_WT(SGP):
    # Sparse Gaussian Process surrogate model class, specifically tailored for WEIS use,
    # inherited from SGP class included in the SMT Toolbox.

    def _initialize(self):
        super()._initialize()
        self._bounds_set = False
        self.keys_in = None
        self.keys_out = None
        self.R_squared = None
        self.constant = False
        self.constant_value = 0.0

    def set_bounds(self, bounds_in, bounds_out):
        self.bounds_in = bounds_in
        self.bounds_out = bounds_out
        self._bounds_set = True

    def predict(self, x):
        # Predicts surrogate model response and variance
        # Input (x) and outputs (y_out, v_out) are denormalized (raw scale) values
        # while actual surrogate model computation is done in normalized scale.

        x_in = np.array(x)

        if not self._bounds_set:
            raise Exception('Normalizing bounds are needed before accessing surrogate model.')

        if not len(x_in.shape) == 2:
            raise Exception('Input array x needs to have shape = (:,n_dv).')

        lb_in = np.tile(self.bounds_in[0,:], (x_in.shape[0], 1))
        ub_in = np.tile(self.bounds_in[1,:], (x_in.shape[0], 1))

        lb_out = np.tile(self.bounds_out[0], (x_in.shape[0], 1))
        ub_out = np.tile(self.bounds_out[1], (x_in.shape[0], 1))

        x_in_normalized = (x_in - lb_in)/(ub_in - lb_in)
        if self.constant:
            y_out_normalized = self.constant_value*np.ones((x_in.shape[0], 1), dtype=float)
            sqrt_v_out_normalized = 0.0
        else:
            y_out_normalized = self.predict_values(x_in_normalized)
            sqrt_v_out_normalized = np.sqrt(self.predict_variances(x_in_normalized))

        y_out = lb_out + (ub_out - lb_out)*y_out_normalized
        sqrt_v_out = (ub_out - lb_out)*sqrt_v_out_normalized
        v_out = sqrt_v_out**2

        return y_out, v_out # values and variances


