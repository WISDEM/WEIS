'''
Various functions for help visualizing WEIS outputs
'''
from openfast_io.FileTools import load_yaml
import weis.inputs as sch
import pandas as pd
import numpy as np
import openmdao.api as om
import glob
import json
import multiprocessing as mp
import plotly.graph_objects as go
import os
import io
import yaml
import re
import socket
from dash import html
#from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
import pickle
import raft
#from raft.helpers import *

import vtk
import dash_vtk
from dash_vtk.utils import to_mesh_state
import pyvista as pv
import plotly

try:
    import ruamel_yaml as ry
except Exception:
    try:
        import ruamel.yaml as ry
    except Exception:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except:
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def checkPort(port, host="0.0.0.0"):
    # check port availability and then close the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind((host, port))
        result = True
    except:
        result = False

    sock.close()
    return result


def parse_yaml(file_path):
    '''
    Parse the data contents in dictionary format
    '''
    # TODO: Encountering the error for parsing hyperlink data - either skip or add that?
    #       load_yaml doesn't work as well..
    # print('Reading the input yaml file..')
    try:
        # dict = load_yaml(file_path, 1)
        with io.open(file_path, 'r') as stream:
            dict = yaml.safe_load(stream)
            # try:
            #     dict = yaml.safe_load(stream)
            # except yaml.YAMLError as exc:
            #     from subprocess import call
            #     call(["yamllint", "-f", "parsable", file_path])
        
        dict['yamlPath'] = file_path
        # print('input file dict:\n', dict)
        return dict
    
    except FileNotFoundError:
        print('Could not locate the input yaml file..')
        exit()
    
    except Exception as e:
        print(e)
        exit()


def dict_to_html(data, out_html_list, level):
    '''
    Show the nested dictionary data to html
    '''
    
    for k1, v1 in data.items():
        if not k1 in ['dirs', 'files']:
            if not isinstance(v1, list) and not isinstance(v1, dict):
                out_html_list.append(html.H6(f'{"---"*level}{k1}: {v1}'))
                continue
            
            out_html_list.append(html.H6(f'{"---"*level}{k1}'))
        
        if isinstance(v1, list):
            out_html_list.append(html.Div([
                                    html.H6(f'{"---"*(level+1)}{i}') for i in v1]))
            
        elif isinstance(v1, dict):
            out_html_list = dict_to_html(v1, out_html_list, level+1)
    

    return out_html_list


def read_cm(cm_file):
    """
    Function originally from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Parameters
    __________
    cm_file : The file path for case matrix

    Returns
    _______
    cm : The dataframe of case matrix
    dlc_inds : The indices dictionary indicating where corresponding dlc is used for each run
    """
    cm_dict = load_yaml(cm_file, package=1)
    cnames = []
    for c in list(cm_dict.keys()):
        if isinstance(c, ry.comments.CommentedKeySeq):
            cnames.append(tuple(c))
        else:
            cnames.append(c)
    cm = pd.DataFrame(cm_dict, columns=cnames)

    return cm

def parse_contents(data):
    """
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/09_design_of_experiments/postprocess_results.py
    """
    collected_data = {}
    for key in data.keys():
        if key not in collected_data.keys():
            collected_data[key] = []

        for key_idx, _ in enumerate(data[key]):
            if isinstance(data[key][key_idx], int):
                collected_data[key].append(np.array(data[key][key_idx]))
            elif len(data[key][key_idx]) == 1:
                try:
                    collected_data[key].append(np.array(data[key][key_idx][0]))
                except:
                    collected_data[key].append(np.array(data[key][key_idx]))
            else:
                collected_data[key].append(np.array(data[key][key_idx]))

    df = pd.DataFrame.from_dict(collected_data)

    return df


def load_vars_file(fn_vars):
    """
    load a json file of problem variables as output from WEIS (as problem_vars.json)

    parameters:
    -----------
    fn_vars: str
        a filename to read

    returns:
    --------
    vars : dict[dict]
        a dictionary of dictionaries holding the problem_vars from WEIS
    """

    with open(fn_vars, "r") as fjson:
        # unpack in a useful form
        vars = {k: dict(v) for k, v in json.load(fjson).items()}
    return vars


def compare_om_data(
    dataOM_1,
    dataOM_2,
    fn_1="data 1",
    fn_2="data 2",
    verbose=False,
):
    """
    compare openmdao data dictionaries to find the in-common (and not) keys

    args:
        dataOM_1: dict
            an openmdao data dictionary
        dataOM_2: dict
            an openmdao data dictionary
        fn_1: str (optional)
            display name for the first data dictionary
        fn_2: str (optional)
            display name for the second data dictionary
        verbose : bool (optional, default: False)
            if we want to print what's happening

    returns:
        keys_all: set
            intersection (i.e. common) keys between the two OM data dictionaries
        diff_keys_12: set
            directional difference of keys between first and second OM data dicts
        diff_keys_21: set
            directional difference of keys between second and first OM data dicts
    """

    diff_keys_12 = set(dataOM_1).difference(dataOM_2)
    diff_keys_21 = set(dataOM_2).difference(dataOM_1)
    if len(diff_keys_12):
        if verbose:
            print(f"the following keys are only in {fn_1}:")
    for key_m in diff_keys_12:
        if verbose:
            print(f"\t{key_m}")
    if len(diff_keys_21):
        if verbose:
            print(f"the following keys are only in {fn_2}:")
    for key_m in diff_keys_21:
        if verbose:
            print(f"\t{key_m}")
    keys_all = set(dataOM_1).intersection(dataOM_2)
    if verbose:
        print(f"the following keys are in both {fn_1} and {fn_2}:")
    for key_m in keys_all:
        if verbose:
            print(f"\t{key_m}")

    return keys_all, diff_keys_12, diff_keys_21

def load_OMsql(
    log,
    parse_multi=False,
    meta=None,
    verbose=False,
):
    """
    load the openmdao sql file produced by a WEIS run into a dictionary

    parameters:
    -----------
        log : str
            filename of the log sql database that should be loaded
        parse_multi : bool
            switch to turn on rank/iteration parsing and storage
        meta : str
            filename of the meta log sql database that should be loaded
        verbose : bool (optional, default: False)
            if we want to print what's happening

    returns:
        rec_data: dict
            dictionary of the data recorded by openMDAO

    """

    # heads-up print
    if verbose:
        print(f"loading {log}")

    # create an openmdao reader for recorded output data
    cr = om.CaseReader(log, metadata_filename=meta)

    # create a dict for output data that's been recorded
    rec_data = {}
    # loop over the cases
    for case in cr.get_cases("driver"):
        if parse_multi:
            rankNo = case.name.split(":")[0]
            assert rankNo.startswith("rank")
            rankNo = int(rankNo[4:])
            iterNo = int(case.name.split("|")[-1])

        # for each key in the outputs
        for key in case.outputs.keys():

            if key not in rec_data:
                # if this key isn't present, create a new list
                rec_data[key] = []

            if hasattr(case[key], 'len') and len(case[key]) != 1:
                # convert to a numpy array if possible and add the data to the list
                rec_data[key].append(np.array(case[key]))
            else:
                rec_data[key].append(case[key])

        if parse_multi:
            # add rank/iter metadata
            for key in ["rank", "iter"]:  # for each key in the outputs
                if key not in rec_data:  # if this key isn't present, create a new list
                    rec_data[key] = []
            rec_data["rank"].append(rankNo)
            rec_data["iter"].append(iterNo)

    return rec_data  # return the output


def load_OMsql_multi(
    log_fmt,
    meta_in=None,
    process_multi = True,
    verbose=False,
):
    """
    load the multi-processor openmdao sql files produced by WEIS into a dict

    parameters:
    -----------
        log_fmt : str
            format string for the process-wise WEIS/OM log files
        meta_in : str (optional, default: None)
            filename string of the meta log file (will override automatic discovery)
        post_multi : bool (optional, default: True)
            postprocess in parallel using the multiprocessing library
        verbose : bool (optional, default: False)
            if we want to print what's happening

    returns:
    --------
        data_dict : dict
            dictionary of all the datapoints extracted from the WEIS/OM log files
    """

    # use glob to find the logs that match the format string
    opt_logs = sorted(
        glob.glob(log_fmt),
        key = lambda v : int(v.split("_")[-1])
            if (v.split("_")[-1] != "meta")
            else 1e8,
    )
    if len(opt_logs) < 1:
        raise FileExistsError("No output logs to postprocess!")

    # remove the "meta" log from the collection
    meta_found = None
    for idx, log in enumerate(opt_logs):
        if "meta" in log:
            meta_found = log  # save the meta file
            opt_logs.pop(idx)  # remove the meta log from the list
            break

    # handle meta logfile discovery... not sure what it actually does
    if meta_in is not None:
        meta = meta_in  # if a meta is given, override
    elif meta_found is not None:
        meta = meta_found  # if a meta is not given but one is found, use that
    else:
        meta = None  # otherwise, run without a meta

    # extract the ranks from the sql files
    sql_ranks = [ol.split("_")[-1] for ol in opt_logs]

    # run multiprocessing
    if process_multi:
        cores = mp.cpu_count()
        pool = mp.Pool(min(len(opt_logs), cores))

        # load sql file
        outdata = pool.starmap(load_OMsql, [(log, True, meta, verbose) for log in opt_logs])
        pool.close()
        pool.join()
    else: # no multiprocessing
        outdata = [load_OMsql(log, parse_multi=True, verbose=verbose, meta=meta) for log in opt_logs]

    # create a dictionary and turn it into a dataframe for convenience
    collected_data = {}
    ndarray_keys = []
    for sql_rank, data in zip(sql_ranks, outdata):
        for key in data.keys():
            if key not in collected_data.keys():
                collected_data[key] = []
            if key == "rank": # adjust the rank based on sql file rank
                data[key] = [int(sql_rank) for _ in data[key]]
            for idx_key, _ in enumerate(data[key]):
                if isinstance(data[key][idx_key], int):
                    collected_data[key].append(int(np.array(data[key][idx_key])))
                elif isinstance(data[key][idx_key], float):
                    collected_data[key].append(float(np.array(data[key][idx_key])))
                elif len(data[key][idx_key]) == 1:
                    collected_data[key].append(float(np.array(data[key][idx_key])))
                    # try:
                    #     collected_data[key].append(np.array(data[key][idx_key][0]))
                    # except:
                    #     collected_data[key].append(np.array(data[key][idx_key]))
                else:
                    collected_data[key].append(np.array(data[key][idx_key]).tolist())
                    ndarray_keys.append(key)
    df = pd.DataFrame(collected_data)

    # return a dictionary of the data that was extracted
    return df.to_dict(orient="list")


def consolidate_multi(
    dataOMmulti,
    vars_dict,
    feas_tol=1e-5,
):
    """
    load the multi-processor openmdao sql files and squash them to the
    per-iteration best-feasible result

    parameters:
    -----------
        dataOMmulti : dict
            dictionary of all the datapoints extracted from the multiprocess
            WEIS/OM log files
        vars_dict:
            experiment design variables to be analyzed
        feas_tol : float (optional)
            tolerance for feasibility analysis
    returns:
    --------
        dataOMbest_DE : dict
            dictionary of the per-iteration best-feasible simulations
    """

    dfOMmulti = pd.DataFrame(dataOMmulti)
    tfeas, cfeas = get_feasible_iterations(dataOMmulti, vars_dict, feas_tol=feas_tol)

    dfOMmulti = dfOMmulti[tfeas].reset_index()

    dataOMbest_DE = dfOMmulti.groupby("iter").apply(
        lambda grp : grp.loc[grp["floatingse.system_structural_mass"].idxmin()],
        include_groups=False,
    ).to_dict()

    for key in dataOMbest_DE.keys():
        dataOMbest_DE[key] = np.array(list(dataOMbest_DE[key].values()))

    return dataOMbest_DE


def get_feasible_iterations(
    dataOM,
    vars_dict,
    feas_tol=1e-5,
):
    """
    get iteration-wise total and per-constraint feasibility from an experiment

    args:
        dataOM: dict
            openmdao data dictionary
        vars_dict:
            experiment design variables for checking

    returns:
        total_feasibility: np.ndarray[bool]
            iteration-wise total feasibility indications
        feasibility_constraintwise: dict[np.ndarray[bool]]
            dictionary to map from constraint names to iteration-wise feasibility indications for that constraint
    """

    # assert len(vars_dict["objectives"].values()) == 1, "can't handle multi-objective... yet. -cfrontin"
    objective_name = list(vars_dict["objectives"].values())[0]["name"]

    feasibility_constraintwise = dict()
    total_feasibility = np.ones_like(np.array(dataOM[objective_name]).reshape(-1,1), dtype=bool)
    for k, v in vars_dict["constraints"].items():
        feasibility = np.ones_like(dataOM[objective_name], dtype=bool).reshape(-1, 1)
        values = np.array(dataOM[v["name"]])
        if len(values.shape) == 1:
            values = values.reshape(-1,1)
        if v.get("upper") is not None:
            feasibility = np.logical_and(feasibility, np.all(np.less_equal(values, (1+feas_tol)*v["upper"]), axis=1).reshape(-1, 1))
        if v.get("lower") is not None:
            feasibility = np.logical_and(feasibility, np.all(np.greater_equal(values, (1-feas_tol)*v["lower"]), axis=1).reshape(-1, 1))
        feasibility_constraintwise[v["name"]] = feasibility
        total_feasibility = np.logical_and(total_feasibility, feasibility)
    return total_feasibility, feasibility_constraintwise


def verify_vars(
    vars_1,
    *vars_i,
):
    """
    verifies format of DVs, constraints, objective variable file
    guarantees a list of experiments has the same variables
    adjusts unbounded constraints
    returns verified list of vars
    """

    for vars_2 in vars_i:
        if vars_2 is not None:
            for k0 in set(vars_1.keys()).union(vars_2):
                assert k0 in vars_1
                assert k0 in vars_2
                for k1 in set(vars_1[k0].keys()).union(vars_2[k0].keys()):
                    assert k1 in vars_1[k0]
                    assert k1 in vars_2[k0]
                    for k2 in set(vars_1[k0][k1].keys()).union(vars_2[k0][k1].keys()):
                        assert k2 in vars_1[k0][k1]
                        assert k2 in vars_2[k0][k1]
                        if k2 == "val":
                            continue
                        if isinstance(vars_1[k0][k1][k2], str):
                            assert vars_1[k0][k1][k2] == vars_2[k0][k1][k2]
                        elif vars_1[k0][k1][k2] is not None:
                            assert np.all(np.isclose(vars_1[k0][k1][k2], vars_2[k0][k1][k2]))
                        else:
                            assert (vars_1[k0][k1][k2] is None) and (vars_2[k0][k1][k2] is None)

    vars_unified = vars_1.copy()
    for k0 in vars_unified.keys():
        for k1 in vars_unified[k0].keys():
            if (vars_unified[k0][k1].get("lower") is not None) and (vars_unified[k0][k1].get("lower") < -1e28):
                vars_unified[k0][k1]["lower"] = -np.inf
            if (vars_unified[k0][k1].get("upper") is not None) and (vars_unified[k0][k1].get("upper") > 1e28):
                vars_unified[k0][k1]["upper"] = np.inf

    return vars_unified


def prettyprint_variables(
    keys_all,
    keys_obj,
    keys_DV,
    keys_constr,
):
    """
    print the variables we have with a prefix showing whether they are an
    objective variable (**), design variabie (--), constraint (<>), or unknown
    (??)
    """

    # print them nicely
    print()
    [
        print(
            f"** {key}"
            if key in keys_obj
            else f"-- {key}" if key in keys_DV else f"<> {key}" if key in keys_constr else f"?? {key}"
        )
        for key in keys_all
    ]
    print()

def read_per_iteration(iteration, stats_paths):

    stats_path_matched = [x for x in stats_paths if f'iteration_{iteration}' in x][0]
    iteration_path = '/'.join(stats_path_matched.split('/')[:-1])
    stats = pd.read_pickle(stats_path_matched)
    # dels = pd.read_pickle(iteration_path+'/DELs.p')
    # fst_vt = pd.read_pickle(iteration_path+'/fst_vt.p')
    print('iteration path with ', iteration, ': ', stats_path_matched)

    return stats, iteration_path


def get_timeseries_data(run_num, stats, iteration_path):
    
    stats = stats.reset_index()     # make 'index' column that has elements of 'IEA_22_Semi_00, ...'
    filename_from_stats = stats.loc[run_num, 'index'].to_string()      # filenames are not same - stats: IEA_22_Semi_83 / timeseries/: IEA_22_Semi_0_83.p
    
    # TODO: Need to clean up later with unified format..
    if filename_from_stats.split('_')[-1].startswith('0'):
        filename = ('_'.join(filename_from_stats.split('_')[:-1])+'_0_'+filename_from_stats.split('_')[-1][1:]+'.p').strip()
    else:
        filename = ('_'.join(filename_from_stats.split('_')[:-1])+'_0_'+filename_from_stats.split('_')[-1]+'.p').strip()
    
    if not os.path.exists('/'.join([iteration_path, 'timeseries', filename])):
        # examples/17_IEA22_Optimization/17_IEA22_OptStudies/of_COBYLA/openfast_runs/iteration_0/timeseries/IEA_22_Semi_0.p
        filename = ('_'.join(filename_from_stats.split('_')[2:-1])+'_'+str(int(filename_from_stats.split('_')[-1]))+'.p').strip()
    
    timeseries_path = '/'.join([iteration_path, 'timeseries', filename])
    timeseries_data = pd.read_pickle(timeseries_path)

    return filename, timeseries_data


def empty_figure():
    '''
    Draw empty figure showing nothing once initialized
    '''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def toggle(click, is_open):
    if click:
        return not is_open
    return is_open


def store_dataframes(var_files):
    dfs = {}
    for _, file_path in var_files.items():
        df = pd.read_csv(file_path, skiprows=[0,1,2,3,4,5,7], sep='\s+')
        dfs[file_path] = df.to_dict('records')
    
    return dfs


def get_file_info(file_path):
    file_name = file_path.split('/')[-1]
    file_abs_path = os.path.abspath(file_path)
    file_size = round(os.path.getsize(file_path) / (1024**2), 2)
    creation_time = os.path.getctime(file_path)
    modification_time = os.path.getmtime(file_path)

    file_info = {
        'file_name': file_name,
        'file_abs_path': file_abs_path,
        'file_size': file_size,
        'creation_time': creation_time,
        'modification_time': modification_time
    }

    return file_info


def find_file_path_from_tree(nested_dict, filename, prepath=()):
    # Works for multi-keyed files
    # Sample outputs: ('outputDirStructure', 'sample_test') ('outputDirStructure', 'sample_multi')
    for k, v in nested_dict.items():
        path = prepath + (k,)
        if v == filename:
            yield path + (v, )
        elif isinstance(v, list) and filename in v:
            yield path + (filename, )
        elif hasattr(v, 'items'):
            yield from find_file_path_from_tree(v, filename, path)

def find_iterations(nested_dict, prepath=()):
    for k, v in nested_dict.items():
        path = prepath + (k,)
        if 'iteration' in k:
            yield int(re.findall(r'\d+', k)[0])
        elif hasattr(v, 'items'):
            yield from find_iterations(v, path)


def update_yaml(input_dict, yaml_filepath):
    with open(yaml_filepath, 'w') as outfile:
        yaml.dump(input_dict, outfile, default_flow_style=False)


def read_cost_variables(labels, refturb_variables):
    # Read tcc cost-related variables from CSV file

    cost_matrix = [['Main Turbine Components', 'Cost']]

    for l in labels:
        cost_matrix.append([l, eval(refturb_variables[f'tcc.{l}_cost']['values'])[0]])
    
    return cost_matrix

def convert_dict_values_to_list(input_dict):
    return {k: [v.tolist()] if isinstance(v, np.ndarray) else v for k, v in input_dict.items()}
    
def generate_raft_img(raft_design_dir, plot_dir, log_data):
    '''
    Temporary function to visualize raft 3d plot using matplotlib.
    TODO: to build interactive 3d plot using plotly
    '''
    os.makedirs(plot_dir,exist_ok=True)

    if isinstance(log_data, list):
        log_data = convert_dict_values_to_list(log_data[0])

    opt_outs = {}
    opt_outs['max_pitch'] = np.squeeze(np.array(log_data['raft.Max_PtfmPitch']))
    n_plots = opt_outs['max_pitch'].size     # Change from len(opt_outs['max_pitch']) to solve single element np.array values
    print('n_plots: ', n_plots)
    
    matplotlib.use('agg')
    for i_plot in range(n_plots):
        # Set up subplots
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = plt.axes(projection='3d')
        
        with open(os.path.join(raft_design_dir,f'raft_design_{i_plot}.pkl'),'rb') as f:
            design = pickle.load(f)

        # TODO: Found typo on gamma value at 1_raft_opt example
        if design['turbine']['tower']['gamma'] == np.array([0.]):
            design['turbine']['tower']['gamma'] = 0.0       # Change it from array([0.])
        
        # set up the model
        model1 = raft.Model(design)
        model1.analyzeUnloaded(
            ballast= False, 
            heave_tol = 1.0
            )

        model1.fowtList[0].r6[4] = np.radians(opt_outs['max_pitch'][i_plot])
        
        _, ax = model1.plot(ax=ax)

        ax.azim = -88.63636363636361
        ax.elev = 27.662337662337674
        ax.set_xlim3d((-110.90447789470043, 102.92063063344857))
        ax.set_ylim3d((64.47420067304586, 311.37818252335893))
        ax.set_zlim3d((-88.43591080818854, -57.499893019459606))
        
        image_filename = os.path.join(plot_dir,f'ptfm_{i_plot}.png')
        plt.savefig(image_filename, bbox_inches='tight')
        print('saved ', image_filename)
        plt.close()


def remove_duplicated_legends(fig):
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    
    return fig


def set_colors():
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS
    return cols

############################
# Viz Utils for WindIO
############################
def render_meshes():
    meshes = []

    cylinder = {'center': [1,2,3], 'direction': [1,1,1], 'radius': 1, 'height': 2}
    sphere = {'center': [0,0,0], 'direction': [0,0,1], 'radius': 0.5}
    plane = {'center': [0,0,0], 'direction': [0,0,1]}
    line = {'pointa': [-0.5,0,0], 'pointb':[0.5,0,0]}
    box = {'bounds': [-1.0,1.0,-1.0,1.0,-1.0,1.0]}
    
    # Define structured points with numpy
    x = np.arange(-10, 10, 0.25)    # (80,)
    y = np.arange(-10, 10, 0.25)    # (80,)
    x, y = np.meshgrid(x, y)        # both (80, 80)
    r = np.sqrt(x**2, y**2)
    z = np.sin(r)
    points = (x, y, z)

    mesh_cylinder = render_cylinder(cylinder)
    mesh_sphere = render_sphere(sphere)
    mesh_plane = render_plane(plane)
    mesh_line = render_line(line)
    mesh_box = render_box(box)
    mesh_random = render_our_own(points)

    meshes.append(mesh_cylinder)
    meshes.append(mesh_sphere)
    meshes.append(mesh_plane)
    meshes.append(mesh_line)
    meshes.append(mesh_box)
    meshes.append(mesh_random)

    return meshes


def render_cylinder(cylinder):
    cylinder = pv.Cylinder(
        center=cylinder['center'], direction=cylinder['direction'], radius=cylinder['radius'], height=cylinder['height']
    )
    mesh_state = to_mesh_state(cylinder)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])

    return content

def render_sphere(sphere):
    sphere = pv.Sphere(
        center=sphere['center'], direction=sphere['direction'], radius=sphere['radius']
    )
    mesh_state = to_mesh_state(sphere)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])

    return content


def render_plane(plane):
    plane = pv.Plane(
        center=plane['center'], direction=plane['direction']
    )
    mesh_state = to_mesh_state(plane)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])

    return content


def render_line(line):
    line = pv.Line(
        pointa=line['pointa'], pointb=line['pointb']
    )
    mesh_state = to_mesh_state(line)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])

    return content


def render_box(box):
    box = pv.Box(
        bounds=box['bounds']
    )
    mesh_state = to_mesh_state(box)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])

    return content


def render_our_own(points):
    '''
    Create and fill the VTK Data Object with your own data using VTK library and pyvista high level api

    Reference: https://tutorial.pyvista.org/tutorial/06_vtk/b_create_vtk.html
    https://docs.pyvista.org/examples/00-load/create-tri-surface
    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.reconstruct_surface#pyvista.PolyDataFilters.reconstruct_surface
    '''

    # Join the points
    x, y, z = points
    values = np.c_[x.ravel(), y.ravel(), z.ravel()]     # (6400, 3) where each column is x, y, z coords
    coords = numpy_to_vtk(values)
    cloud = pv.PolyData(coords)
    # mesh = cloud.delaunay_2d()          # From point cloud, apply a 2D Delaunary filter to generate a 2d surface from a set of points on a plane.
    mesh = cloud.delaunay_3d()


    # Work for sin-plane but not for cylinder..
    '''
    # Join the points
    x, y, z = points
    values = np.c_[x.ravel(), y.ravel(), z.ravel()]     # (6400, 3) where each column is x, y, z coords
    coords = numpy_to_vtk(values)
    
    points = vtk.vtkPoints()
    points.SetData(coords)

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(*z.shape, 1)     # *zshape: (80 80) for sin-plane / 1000 for cylinder
    grid.SetPoints(points)

    # Add point data
    arr = numpy_to_vtk(z.ravel())
    arr.SetName("z")
    grid.GetPointData().SetScalars(arr)
    '''

    mesh_state = to_mesh_state(mesh)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            mapper={'orientationArray': 'Normals'},
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])
    
    return content

def render_our_own_delaunay(points):
    '''
    Create and fill the VTK Data Object with your own data using VTK library and pyvista high level api

    Reference: https://tutorial.pyvista.org/tutorial/06_vtk/b_create_vtk.html
    https://docs.pyvista.org/examples/00-load/create-tri-surface
    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.reconstruct_surface#pyvista.PolyDataFilters.reconstruct_surface
    '''

    # Join the points
    x, y, z = points
    values = np.c_[x.ravel(), y.ravel(), z.ravel()]     # (6400, 3) where each column is x, y, z coords
    coords = numpy_to_vtk(values)
    cloud = pv.PolyData(coords)
    # mesh = cloud.delaunay_2d()          # From point cloud, apply a 2D Delaunary filter to generate a 2d surface from a set of points on a plane.
    mesh = cloud.delaunay_3d()

    mesh_state = to_mesh_state(mesh)

    return mesh_state

def find_rows(df_dict, df_type='geometry'):

    df = pd.DataFrame(df_dict)
    
    return df[df['Type'] == df_type].to_dict('list')         # {'File Path': ['a.yaml', 'c.yaml'], 'Label': ['abc', 'ghi'], 'Type': [1, 1]}

    # return {k: v[idx] for idx in range(len(df['Type'])) for k, v in df.items() if df['Type'][idx]==df_type}



def load_geometry_data(geometry_paths):
    # Read geometry input file and load airfoils data
    # wt_options = sch.load_geometry_yaml('/projects/weis/sryu/visualization_cases/1_raft_opt/IEA-22-280-RWT.yaml')       # For HPC
    # wt_options = sch.load_geometry_yaml('/Users/sryu/Desktop/FY24/WEIS-Visualization/data/visualization_cases/1_raft_opt/IEA-22-280-RWT.yaml')       # For Local
    airfoils, geom_comps, wt_options_by_file = {}, {}, {}
    # for row in geometry_paths:
    #     wt_options = sch.load_geometry_yaml(row['File Path'])
    #     airfoils[row['Label']] = wt_options['airfoils']
    #     geom_comps[row['Label']] = wt_options['components']

    for filepath, filelabel, _ in zip(*geometry_paths.values()):
        wt_options = sch.load_geometry_yaml(filepath)
        airfoils[filelabel] = wt_options['airfoils']
        geom_comps[filelabel] = wt_options['components']
        wt_options_by_file[filelabel] = wt_options


    return airfoils, geom_comps, wt_options_by_file

###################################################
# Not needed below.. Will be deleted later
###################################################
def load_mesh(file_path):
    '''
    Read either STL or VTK file and load the 3D mesh
    '''
    # Read STL file generated by WindIO Tool
    # Sample files to test
    # file_path = '/Users/sryu/Desktop/FY24/windio2cad/nrel5mw-semi_oc4.stl'
    # file_path = '/Users/sryu/Desktop/FY24/ACDC/project1/case01/vtk/01_NREL_5MW-ED.Mode1.LinTime1.ED_BladeLn2Mesh_motion1.010.vtp'
    if file_path.endswith('.stl'):
        reader = vtk.vtkSTLReader()
    elif file_path.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    
    reader.SetFileName(file_path)
    reader.Update()

    # Get dataset and build a mesh structure to pass it as a Mesh
    dataset = reader.GetOutput()
    mesh_state = to_mesh_state(dataset)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation([
            dash_vtk.Mesh(state=mesh_state)
        ])
    ])

    return content


def render_terrain():
    '''
    This is an example of VTK rendering of point cloud. Only for reference.. Will be deleted later.
    '''
    # Get point cloud data from PyVista
    uniformGrid = pv.examples.download_crater_topo()
    subset = uniformGrid.extract_subset((500, 900, 400, 800, 0, 0), (5, 5, 1))

    # Update warp
    terrain = subset.warp_by_scalar(factor=1)
    polydata = terrain.extract_geometry()
    points = polydata.points.ravel()                        # points: [1.81750e+06 5.64600e+06 1.55213e+03 ... 1.82350e+06 5.65200e+06 1.91346e+03] / shape is (19683,) with dtype=float32
    polys = vtk_to_numpy(polydata.GetPolys().GetData())     # polys: [   4    0    1 ... 6479 6560 6559] / shape is (32000,) with dtype=int64
    elevation = polydata["scalar1of1"]
    color_range = [np.amin(elevation), np.amax(elevation)]

    content = dash_vtk.View(
        pickingModes=["hover"],
        children=[
            dash_vtk.GeometryRepresentation(
                id="vtk-representation",
                children=[
                    dash_vtk.PolyData(
                        id="vtk-polydata",
                        points=points,
                        polys=polys,
                        children=[
                            dash_vtk.PointData(
                                [
                                    dash_vtk.DataArray(
                                        id="vtk-array",
                                        registration="setScalars",
                                        name="elevation",
                                        values=elevation,
                                    )
                                ]
                            )
                        ],
                    )
                ],
                colorMapPreset="erdc_blue2green_muted",
                colorDataRange=color_range,
                property={"edgeVisibility": True},
                showCubeAxes=True,
                cubeAxesStyle={"axisLabels": ["", "", "Altitude"]},
            ),
            dash_vtk.GeometryRepresentation(
                id="pick-rep",
                actor={"visibility": False},
                children=[
                    dash_vtk.Algorithm(
                        id="pick-sphere",
                        vtkClass="vtkSphereSource",
                        state={"radius": 100},
                    )
                ],
            ),
        ],
    )
    
    return content


def render_volume():
    '''
    This is an example of a randome volume generation. Only for reference.. Will be deleted later.
    '''
    import random

    content = dash_vtk.View(
        children=dash_vtk.VolumeDataRepresentation(
            spacing=[1, 1, 1],
            dimensions=[10, 10, 10],
            origin=[0, 0, 0],
            scalars=[random.random() for _ in range(1000)],
            rescaleColorMap=False,
        )
    )
    
    return content


def render_mesh_with_grid():
    '''
    Create and fill the VTK Data Object with your own data using VTK library and pyvista high level api

    Reference: https://tutorial.pyvista.org/tutorial/06_vtk/b_create_vtk.html
    '''
    # Define structured points with numpy
    x = np.arange(-10, 10, 0.25)    # (80,)
    y = np.arange(-10, 10, 0.25)    # (80,)
    x, y = np.meshgrid(x, y)        # both (80, 80)
    r = np.sqrt(x**2, y**2)
    z = np.sin(r)

    # Join the points
    values = np.c_[x.ravel(), y.ravel(), z.ravel()]     # (6400, 3) where each column is x, y, z coords
    coords = numpy_to_vtk(values)
    
    points = vtk.vtkPoints()
    points.SetData(coords)

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(*z.shape, 1)
    grid.SetPoints(points)

    # Add point data
    arr = numpy_to_vtk(z.ravel())
    arr.SetName("z")
    grid.GetPointData().SetScalars(arr)

    mesh_state = to_mesh_state(grid)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            mapper={'orientationArray': 'Normals'},
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])
    
    return content


def render_mesh_with_faces():
    '''
    Create and fill the VTK Data Object with your own data using VTK library and pyvista high level api.
    '''
    points = np.array(
        [
            [0.0480, 0.0349, 0.9982],
            [0.0305, 0.0411, 0.9987],
            [0.0207, 0.0329, 0.9992],
            [0.0218, 0.0158, 0.9996],
            [0.0377, 0.0095, 0.9992],
            [0.0485, 0.0163, 0.9987],
            [0.0572, 0.0603, 0.9965],
            [0.0390, 0.0666, 0.9970],
            [0.0289, 0.0576, 0.9979],
            [0.0582, 0.0423, 0.9974],
            [0.0661, 0.0859, 0.9941],
            [0.0476, 0.0922, 0.9946],
            [0.0372, 0.0827, 0.9959],
            [0.0674, 0.0683, 0.9954],
        ],
    )

    face_a = [6, 0, 1, 2, 3, 4, 5]
    face_b = [6, 6, 7, 8, 1, 0, 9]
    face_c = [6, 10, 11, 12, 7, 6, 13]
    faces = np.concatenate((face_a, face_b, face_c))

    mesh = pv.PolyData(points, faces)
    mesh_state = to_mesh_state(mesh)

    content = dash_vtk.View([
        dash_vtk.GeometryRepresentation(
            children=[dash_vtk.Mesh(state=mesh_state)],
            showCubeAxes=True,      # Show origins
        )
    ])
    
    return content

