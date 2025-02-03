import glob

import glob
import json
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

from weis.visualization.utils import *


def plot_conv(
    keyset_in,
    map_dataOM_vars,
    use_casewise_feasibility=False,
    feas_tol=1e-5,
    figax=None,
    alpha=None,
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


    fig, axes = figax if figax else plt.subplots(
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

    pt_imethod = []
    for imethod, method in enumerate(map_dataOM_vars.keys()):
        if imethod == 0:
            markerstyle = "o"
        elif imethod == 1:
            markerstyle = "p"
        elif imethod == 2:
            markerstyle = "s"
        else:
            markerstyle = "P"

        pt0 = axes[0, 0].plot(
            [],
            [],
            markerstyle + linestyle,
            label=method,
            markersize=markersize,
            # color=(0.5,0.5,0.5),
            alpha=alpha,
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

            axes[idx_ax, 0].plot(
                np.squeeze(dataOM[key]),
                linestyle,
                label="".join(["_", method, "_"]),
                color=pt0[-1].get_color(),
                markersize=markersize,
                alpha=alpha,
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
                alpha=alpha,
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
                alpha=alpha,
                fillstyle="none",
                markersize=markersize,
            )
            if has_ref_vals:
                cval = key_val_map[key]
                if (cval[0] is not None) and (np.log10(np.abs(cval[0])) < 18):
                    axes[idx_ax, 0].plot([0, len(dataOM[key])], [cval[0], cval[0]], "b:", label="_lower bound_")
                if (cval[1] is not None) and (np.log10(np.abs(cval[1])) < 18):
                    axes[idx_ax, 0].plot([0, len(dataOM[key])], [cval[1], cval[1]], "r:", label="_upper bound_")
            axes[idx_ax, 0].set_title(key)

    if has_ref_vals:
        axes[0, 0].plot([], [], "b:", label="lower bound")
        axes[0, 0].plot([], [], "r:", label="upper bound")
    axes[0, 0].legend()
    fig.tight_layout()

    return fig, axes

