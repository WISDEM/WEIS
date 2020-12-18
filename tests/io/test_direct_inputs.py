__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from pCrunch.io import OpenFASTOutput


def test_array_input(array_input):

    channels, data = array_input
    output = OpenFASTOutput(data, channels, "Test Data")

    assert output.data.shape
    assert output.channels.shape


def test_dict_input(dict_input):

    output = OpenFASTOutput.from_dict(dict_input, "Test Data")

    assert output.data.shape
    assert output.channels.shape
