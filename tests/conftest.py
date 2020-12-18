"""Shared pytest settings and fixtures."""


import os

import numpy as np
import pytest

DIR = os.path.split(__file__)[0]
DATA = os.path.join(DIR, "io", "data")


@pytest.fixture()
def data_library():

    return DATA


@pytest.fixture()
def dict_input():

    data = {
        "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "WindVxi": [7, 7, 7, 7, 7, 8, 8, 8, 8, 8],
        "WindVyi": [0] * 10,
        "WindVzi": [0] * 10,
    }

    return data


@pytest.fixture()
def array_input():

    data = {
        "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "WindVxi": [7, 7, 7, 7, 7, 8, 8, 8, 8, 8],
        "WindVyi": [0] * 10,
        "WindVzi": [0] * 10,
    }

    return (list(data.keys()), np.array(list(data.values())).T)


@pytest.fixture()
def Ascii():

    pass


@pytest.fixture()
def Binary():

    pass


@pytest.fixture()
def Output():

    pass
