__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from pCrunch.io import OpenFASTOutput


def test_array_input(array_input, magnitude_channels):

    channels, data = array_input
    output = OpenFASTOutput(
        data, channels, "Test Data", magnitude_channels=magnitude_channels
    )

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels


def test_dict_input(dict_input, magnitude_channels):

    output = OpenFASTOutput.from_dict(
        dict_input, "Test Data", magnitude_channels=magnitude_channels
    )

    assert output.data.shape
    assert output.channels.shape
    assert "Wind" in output.channels
