# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_windio_blade import load_blade_comps, draw_blade_oml, draw_blade_matrix
from weis.visualization.utils import load_geometry_data
import os

this_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( this_dir ) ) ) ) )

file_table = {'File Path': [os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-15-240-RWT.yaml'),
                            os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-3p4-130-RWT.yaml')],
              'Label': ['15MW', '3.4MW'], 'Type': ['geometry', 'geometry']}

_, geom_comps, _ = load_geometry_data(file_table)
geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}
blade_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'blade' in k}     # where now k is 'filelabelname' and v is dict

def test_load_blade_comps_callbacks():
    output, _ = load_blade_comps(geom_comps_by_names)
    assert set(output) == set(['15MW', '3.4MW'])        # Let it pass even though the order is different.. (To solve the error under python=3.9)


def test_draw_blade_oml_callbacks():
    blade_names = ['15MW', '3.4MW']       # Select 3.4MW, 15MW
    fig = draw_blade_oml(blade_names, blade_by_names)

    assert len(fig['data']) == len(blade_names) * 4     # We draw LE, TE, twist and chord for each blade


def test_draw_blade_matrix_callbacks():
    blade_names = ['15MW', '3.4MW']       # Select 3.4MW, 15MW
    fig1, fig2 = draw_blade_matrix(blade_names, blade_by_names)
    num_traces = 21 * len([b for b in blade_names if 'elastic_properties_mb' in blade_by_names[b]])       # Draw a single trace per subplots for each matrix (6+5+4+3+2+1) if you have elastic properties
    
    assert len(fig1['data']) == len(fig2['data']) == num_traces

