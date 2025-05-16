# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_windio_tower import load_tower_comps, draw_tower
from weis.visualization.utils import load_geometry_data
import os

this_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( this_dir ) ) ) ) )

file_table = {'File Path': [os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-15-240-RWT.yaml'),
                            os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-3p4-130-RWT.yaml')],
              'Label': ['15MW', '3.4MW'], 'Type': ['geometry', 'geometry']}

_, geom_comps, _ = load_geometry_data(file_table)
geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}
tower_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'tower' in k}     # where now k is 'filelabelname' and v is dict

def test_load_tower_comps_callbacks():
    output, _ = load_tower_comps(geom_comps_by_names)

    assert set(output) == set(['15MW', '3.4MW'])        # Let it pass even though the order is different.. (To solve the error under python=3.9)

def test_draw_tower_callbacks():
    tower_names = ['15MW', '3.4MW']       # Select 3.4MW, 15MW
    fig = draw_tower(tower_names, tower_by_names)

    assert len(fig['data']) == len(tower_names) * 4     # We draw 4 traces for each tower
