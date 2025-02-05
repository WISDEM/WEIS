from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_windio_3d import list_labels, initial_loading, toggle_local_scene, update_local_header, update_local_table_content, update_local_scene_content, click_local_view
from weis.visualization.utils import load_geometry_data

# File paths are relative to app/
# This is okay for pytest, but change it into absolute paths if you want to debug
file_table = {'File Path': ['../../../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_Monopile.yaml', '../../../../examples/05_IEA-3.4-130-RWT/IEA-3p4-130-RWT.yaml', '../../../../examples/17_IEA22_Optimization/IEA-22-280-RWT-Semi.yaml'], 'Label': ['15MW', '3.4MW', '22MW'], 'Type': ['geometry', 'geometry', 'geometry']}
_, geom_comps, wt_options_by_names = load_geometry_data(file_table)
geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}


def test_list_labels_callbacks():
    output = list_labels(geom_comps_by_names)
    assert set(output) == set(['15MW', '3.4MW', '22MW'])        # Let it pass even though the order is different.. (To solve the error under python=3.9)


def test_initial_loading_callbacks():
    nClicks = 1
    geom_3d_names = ['15MW', '3.4MW', '22MW']

    output_view, output_colors = initial_loading(nClicks, geom_3d_names, wt_options_by_names)
    num_geometries = len([v for gname in geom_3d_names for v in list(wt_options_by_names[gname]['components'].keys()) if v != 'mooring']) + 1     # 1 is for axis geometry and remove duplicated field for 'floating_platform' and 'mooring'. (We just render them into a single mesh..)

    assert len(output_view.children) == num_geometries and len(output_colors) == len(geom_3d_names)
