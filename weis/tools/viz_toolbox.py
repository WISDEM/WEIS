import glob
import os

import glob
import json
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om


def load_vars_file(fn_vars):
    """
    load a json file of problem variables as output from WEIS (as problem_vars.json)

    args:
    -----
    fn_vars: str
        a filename to read
    """

    with open(fn_vars, "r") as fjson:
        # unpack in a useful form
        vars = {k: dict(v) for k, v in json.load(fjson).items()}
    return vars


def load_OMsql_multi(
    log_fmt,
    verbose=False,
    meta=None,
):
    """
    load the multi-processor openmdao sql files produced by WEIS into a dict

    args:
        TO DO!

    returns:
        TO DO!
    """

    # multiprocssing?
    post_multi = True

    opt_logs = sorted(
        glob.glob(log_fmt),
        key = lambda v : int(v.split("_")[-1])
            if (v.split("_")[-1] != "meta")
            else 1e8,
    )
    if len(opt_logs) < 1:
        raise FileExistsError("No output logs to postprocess!")

    # remove the "meta" log
    for idx, log in enumerate(opt_logs):
        if "meta" in log:
            opt_logs.pop(idx)

    # sql ranks
    sql_ranks = [ol.split("_")[-1] for ol in opt_logs]

    # run multiprocessing
    if post_multi:
        cores = mp.cpu_count()
        pool = mp.Pool(min(len(opt_logs), cores))

        # load sql file
        outdata = pool.starmap(load_OMsql, [(ol, verbose, meta) for ol in opt_logs])
        pool.close()
        pool.join()
    else: # no multiprocessing
        outdata = [load_OMsql(log, verbose=verbose, meta=meta) for log in opt_logs]

    # create a dictionary and turn it into a dataframe
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
                    collected_data[key].append(np.array(data[key][idx_key]))
                elif isinstance(data[key][idx_key], float):
                    collected_data[key].append(np.array(data[key][idx_key]))
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

    # return the dataframe
    return df.to_dict(orient="list")

    # # gather logs matching the format
    # log_list = sorted(glob.glob(log_fmt), key=lambda v: int(v.split("_")[-1]) if not (".sql_meta" in v) else 1e8)
    # sql_codes = [(v.split("_")[-1]) for v in log_list]
    #
    # # run the OM sql load on each log data
    # case_data = []
    # for log in log_list:
    #     if "sql_meta" in log: continue # skip the meta file
    #     rankNo = int(log.split("_")[-1])
    #     case_data.append(load_OMsql(log, verbose=verbose, meta=meta))
    #     case_data[-1]["rank"] = [rankNo for _ in case_data[-1]["rank"]]
    #
    # # return the total dictionary
    # return case_data


def load_OMsql(
    log,
    verbose=False,
    meta=None,
):
    """
    load the openmdao sql file produced by a WEIS run into a dictionary

    args:
        log: str
            filename of the log sql database that should be loaded
        verbose: bool (optional)
            if we want to nicely print the output

    returns:
        rec_data: dict
            dictionary of the data recorded by openMDAO

    """

    if verbose:
        print(f"loading {log}")

    cr = om.CaseReader(log, metadata_filename=meta)  # openmdao reader for recorded output data

    rec_data = {}  # create a dict for output data that's been recorded
    for case in cr.get_cases("driver"):  # loop over the cases
        rankNo = case.name.split(":")[0]
        assert rankNo.startswith("rank")
        rankNo = int(rankNo[4:])
        iterNo = int(case.name.split("|")[-1])

        for key in case.outputs.keys():  # for each key in the outputs
            if key not in rec_data:  # if this key isn't present, create a new list
                rec_data[key] = []
            if len(case[key]) == 1:
                rec_data[key].append(float(case[key]))  # add the data to the list
            else:
                rec_data[key].append(np.array(case[key]))  # add the data to the list

        # add rank/iter metadata
        for key in ["rank", "iter"]:  # for each key in the outputs
            if key not in rec_data:  # if this key isn't present, create a new list
                rec_data[key] = []
        rec_data["rank"].append(rankNo)
        rec_data["iter"].append(iterNo)

    return rec_data  # return the output


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
        verbose: bool (optional)
            print estensive outputs

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


def plot_conv(
    keyset_in,
    map_dataOM_vars,
    use_casewise_feasibility=False,
    feas_tol=1e-5,
):
    """
    plot a set of keys

    args:
        keyset_in: list[str]
            list of keys to plot the convergence data for
        map_dataOM_vars: dict[str -> dict]
            map from a case of interest by name to an OM data dict to plot
        use_casewise_feasibility: bool
            if plotting a constraint should we plot feasibility w.r.t. that constraint (vs. all)

    returns:
        fig : plt.Figure
        ax : plt.Axes
    """

    if len(keyset_in) == 0:
        return

    markerstyle = "x"
    markersize = 5
    linestyle = "-"

    fig, axes = plt.subplots(
        len(keyset_in),
        1,
        sharex=True,
        figsize=(6, 0.60 * 4 * len(keyset_in)),
        squeeze=False,
        dpi=150,
    )

    has_ref_vals = type(keyset_in) == dict

    if has_ref_vals:
        key_val_map = keyset_in
        keyset = keyset_in.keys()
    else:
        keyset = keyset_in

    for imethod, method in enumerate(map_dataOM_vars.keys()):
        if imethod == 0:
            markerstyle = "o"
        elif imethod == 1:
            markerstyle = "p"
        elif imethod == 2:
            markerstyle = "s"
        else:
            markerstyle = "P"

        axes[0, 0].plot(
            [],
            [],
            markerstyle + linestyle,
            label=method,
            markersize=markersize,
            color=(0.5,0.5,0.5),
        )
        dataOM = map_dataOM_vars[method][0]
        vars = map_dataOM_vars[method][1]
        tfeas, varfeas = get_feasible_iterations(dataOM, vars, feas_tol=feas_tol)

        for idx_ax, key in enumerate(keyset):
            if key in ["rank", "iter",]: continue
            if use_casewise_feasibility and key in varfeas.keys():
                feas_val = varfeas[key]
            else:
                feas_val = tfeas  # use total feasibility

            pt0 = axes[idx_ax, 0].plot(
                np.squeeze(dataOM[key]),
                linestyle,
                label="".join(["_", method, "_"]),
                markersize=markersize,
            )
            axes[idx_ax, 0].plot(
                np.ma.array(
                    dataOM[key],
                    mask=~(
                        feas_val * np.ones(
                            (
                                1,
                                np.array(dataOM[key]).shape[1]
                                if len(np.array(dataOM[key]).shape) > 1
                                else 1
                            ),
                            dtype=bool,
                        )
                    )
                ),
                markerstyle,
                label="".join(["_", method, "_"]),
                color=pt0[-1].get_color(),
                fillstyle="full",
                markersize=markersize,
            )
            axes[idx_ax, 0].plot(
                np.ma.array(
                    dataOM[key],
                    mask=(
                        feas_val * np.ones(
                            (
                                1,
                                np.array(dataOM[key]).shape[1]
                                if len(np.array(dataOM[key]).shape) > 1
                                else 1
                            ),
                            dtype=bool,
                        )
                    )
                ),
                markerstyle,
                label="".join(["_", method, "_"]),
                color=pt0[-1].get_color(),
                fillstyle="none",
                markersize=markersize,
            )
            if has_ref_vals and (imethod == 0):
                cval = key_val_map[key]
                if cval[0] is not None:
                    axes[idx_ax, 0].plot([0, len(dataOM[key])], [cval[0], cval[0]], "b:", label="_lower bound_")
                if cval[1] is not None:
                    axes[idx_ax, 0].plot([0, len(dataOM[key])], [cval[1], cval[1]], "r:", label="_upper bound_")
            axes[idx_ax, 0].set_title(key)

    if has_ref_vals:
        axes[0, 0].plot([], [], "b:", label="lower bound")
        axes[0, 0].plot([], [], "r:", label="upper bound")
    axes[0, 0].legend()
    fig.tight_layout()

    return fig, axes


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

