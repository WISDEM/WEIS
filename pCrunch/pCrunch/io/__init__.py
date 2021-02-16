__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import numpy as np

from .openfast import OpenFASTAscii, OpenFASTBinary, OpenFASTOutput


def load_FAST_out(filenames, tmin=0, tmax=np.inf, **kwargs):
    """
    Load a list of OpenFAST files.

    Parameters
    ----------
    filenames : list
        List OpenFAST files to load.
    tmin : float | int, optional
        Initial line to trim output data to.
    tmax : float | int, optional

    Returns
    -------
    list
        List of OpenFASTOutput instances
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    fastout = []
    for fn in filenames:

        try:
            output = OpenFASTAscii(fn, **kwargs)
            output.read()

        except UnicodeDecodeError:
            output = OpenFASTBinary(fn, **kwargs)
            output.read()

        output.trim_data(tmin, tmax)
        fastout.append(output)

    return fastout
