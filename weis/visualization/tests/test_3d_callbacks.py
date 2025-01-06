from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

# Import all of the names of callback functions to tests
from ..appServer.app.pages.visualize_windio_3d import list_labels, visualize
from ..utils import load_geometry_data

# File paths are relative to visualization/
file_table = {'File Path': ['../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_Monopile.yaml', '../../examples/05_IEA-3.4-130-RWT/IEA-3p4-130-RWT.yaml'], 'Label': ['15MW', '3,4MW'], 'Type': ['geometry', 'geometry']}

def test_list_labels_callbacks():
    _, geom_comps, _ = load_geometry_data(file_table)
    geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}
    output = list_labels(geom_comps_by_names)
    assert output == ['15MW', '3,4MW']


def test_visualize_callbacks():
    pass