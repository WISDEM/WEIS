import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
from bs4 import BeautifulSoup
import os

# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_windio_3d import list_labels, initial_loading, toggle_local_scene, update_local_header, update_local_table_content, update_local_scene_content, click_local_view
from weis.visualization.utils import load_geometry_data

this_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( this_dir ) ) ) ) )

file_table = {'File Path': [os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-15-240-RWT.yaml'),
                            os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-3p4-130-RWT.yaml'),
                            os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-22-280-RWT_Floater.yaml')],
              'Label': ['15MW', '3.4MW', '22MW'], 'Type': ['geometry', 'geometry', 'geometry']}
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


def test_toggle_local_scene():
    info = {'worldPosition': [0,0,0], 'displayPosition': [0,0,0], 'compositeID': 1, 'representationId': None, 'ray': [[0,0,0], [0,0,0]]}          # Random dummy info to test functionality
    output = toggle_local_scene(info=info, is_open=False)

    assert output == True

def test_update_local_header():
    info_tower = {'worldPosition': [165.9507745273005, -1.0578314024123374, 76.31877503078084], 'displayPosition': [815, 473, 0.9961089494163424], 'compositeID': 18, 'representationId': '1-tower-rep', 'ray': [[-717.3180998465665, 5.85103711285667, 27.765942056770616], [510.9722279695685, -4.2303597941686375, 95.28527787531458]]}
    output = update_local_header(info=info_tower)

    assert output == 'tower'


def test_update_local_table_content():
    geom_3d_names = ['15MW', '3.4MW', '22MW']

    info_tower = {'worldPosition': [165.9507745273005, -1.0578314024123374, 76.31877503078084], 'displayPosition': [815, 473, 0.9961089494163424], 'compositeID': 18, 'representationId': '1-tower-rep', 'ray': [[-717.3180998465665, 5.85103711285667, 27.765942056770616], [510.9722279695685, -4.2303597941686375, 95.28527787531458]]}
    info_hub = {'worldPosition': [-73.94327115838404, -588.2113555928286, -79.23522050445712], 'displayPosition': [407, 675, 0.9961089494163424], 'compositeID': 18, 'representationId': '1-hub-rep', 'ray': [[69.19511771010448, 705.390273059073, 424.19606797301543], [-130.5321693217238, -1093.6463635678754, -275.1499800733443]]}
    info_nacelle = {'worldPosition': [408.14524393293925, 86.63564418893745, 365.7396308885164], 'displayPosition': [1243, 634, 0.9961089494163424], 'compositeID': 18, 'representationId': '2-nacelle-rep', 'ray': [[-379.2887699625321, -96.11346614345888, -26.92764160309813], [715.9761467503329, 157.62497130727294, 519.0779537902234]]}
    
    output_tower = update_local_table_content(info_tower, geom_3d_names, wt_options_by_names)
    output_hub = update_local_table_content(info_hub, geom_3d_names, wt_options_by_names)
    output_nacelle = update_local_table_content(info_nacelle, geom_3d_names, wt_options_by_names)

    answer_tower = dcc.Link(dbc.Button("Tower Properties"), href='/windio_tower')

    # Parse and remove irrelevant formatting (like extra spaces..)
    html_output_tower = "".join([BeautifulSoup(str(html), 'html.parser').prettify() for html in output_tower])
    html_answer_tower = "".join([BeautifulSoup(str(html), 'html.parser').prettify() for html in answer_tower])
    assert html_output_tower == html_answer_tower

    # Convert single-index dbc.Table to pd.DataFrame to query
    hub_headers = [th.children for th in output_hub.children[0].children.children]
    hub_data = [[td.children.children for td in tr.children] for tr in output_hub.children[1].children]
    hub_df = pd.DataFrame(hub_data, columns=hub_headers)
    assert set(hub_df.loc[:, 'Label']) == set(geom_3d_names)
    assert hub_df.loc[geom_3d_names.index('15MW'), 'diameter'] == wt_options_by_names['15MW']['components']['hub']['diameter']
    assert hub_df.loc[geom_3d_names.index('22MW'), 'pitch_system_scaling_factor'] == wt_options_by_names['22MW']['components']['hub']['pitch_system_scaling_factor']

    # Convert multi-index dbc.Table to pd.DataFrame to query
    html_nacelle_headers = [th.children for th in output_nacelle.children[0].children]
    nacelle_field = [tuple(th.children for th in row for _ in range(th.colSpan)) for row in html_nacelle_headers]
    nacelle_headers = list(zip(nacelle_field[0], nacelle_field[1]))
    index = pd.MultiIndex.from_tuples(nacelle_headers)
    nacelle_data = [[td.children if isinstance(td.children, str) else td.children.children for td in tr.children] for tr in output_nacelle.children[1].children]
    nacelle_df = pd.DataFrame(nacelle_data, columns=index)
    assert set(nacelle_df.loc[:, 'Label']) == set(geom_3d_names)
    assert nacelle_df.loc[geom_3d_names.index('15MW'), ('drivetrain', 'uptilt')] == wt_options_by_names['15MW']['components']['nacelle']['drivetrain']['uptilt']
    assert nacelle_df.loc[geom_3d_names.index('3.4MW'), ('generator', 'rated_rpm')] == wt_options_by_names['3.4MW']['components']['nacelle']['generator']['rated_rpm']


def test_update_local_scene_content():
    info_tower = {'worldPosition': [165.9507745273005, -1.0578314024123374, 76.31877503078084], 'displayPosition': [815, 473, 0.9961089494163424], 'compositeID': 18, 'representationId': '1-tower-rep', 'ray': [[-717.3180998465665, 5.85103711285667, 27.765942056770616], [510.9722279695685, -4.2303597941686375, 95.28527787531458]]}
    geom_3d_names = ['15MW', '3.4MW', '22MW']
    tower_view = update_local_scene_content(info_tower, geom_3d_names, wt_options_by_names)

    assert set([geom.id for geom in tower_view.children[:-2]]) == set(geom_3d_names)        # Last two geometries: Axes and tooltip


def test_click_local_view():
    info_tower = {'worldPosition': [165.9507745273005, -1.0578314024123374, 76.31877503078084], 'displayPosition': [815, 473, 0.9961089494163424], 'compositeID': 18, 'representationId': '1-tower-rep', 'ray': [[-717.3180998465665, 5.85103711285667, 27.765942056770616], [510.9722279695685, -4.2303597941686375, 95.28527787531458]]}
    tooltip_children, _ = click_local_view(info_tower)
    
    assert tooltip_children == [f"{info_tower['representationId']}: {info_tower['worldPosition']}"]

