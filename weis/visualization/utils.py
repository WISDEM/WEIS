'''
Various functions for help visualizing WEIS outputs
'''
from weis.aeroelasticse.FileTools import load_yaml
import pandas as pd
import numpy as np
import openmdao.api as om
import glob
import json
import multiprocessing as mp

try:
    import ruamel_yaml as ry
except Exception:
    try:
        import ruamel.yaml as ry
    except Exception:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

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
            if len(case[key]) == 1:
                # otherwise coerce to float if possible and add the data to the list
                rec_data[key].append(float(case[key]))
            else:
                # otherwise a numpy array if possible and add the data to the list
                rec_data[key].append(np.array(case[key]))

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
