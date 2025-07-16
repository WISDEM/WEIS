from dash import html
from bs4 import BeautifulSoup
import os

# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_windio_airfoils import list_airfoils, draw_airfoil_shape, draw_airfoil_polar
from weis.visualization.utils import load_geometry_data

this_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( this_dir ) ) ) ) )

file_table = {'File Path': [os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-15-240-RWT.yaml'),
                            os.path.join(weis_dir, 'examples/00_setup/ref_turbines/IEA-3p4-130-RWT.yaml')],
              'Label': ['15MW', '3.4MW'], 'Type': ['geometry', 'geometry']}

airfoils, _, _ = load_geometry_data(file_table)
airfoil_by_names = {label+': '+airfoil['name']: dict(list(airfoil.items())[1:]) for label, airfoils_per_file in airfoils.items() for airfoil in airfoils_per_file}      # {'file1: FB90': {'coordinates': {'x': [1.0, 0.9921, 0.5], 'y': [1.0, 2.0, 3.0]}}, ... }
    
def test_list_airfoils_callbacks():
    output = list_airfoils(airfoil_by_names)
    assert set(output) == set(['15MW: circular', '15MW: SNL-FFA-W3-500', '15MW: FFA-W3-211', '15MW: FFA-W3-241', '15MW: FFA-W3-270blend', '15MW: FFA-W3-301', '15MW: FFA-W3-330blend', '15MW: FFA-W3-360', '3.4MW: DU08-W-210', '3.4MW: DU91-W2-250', '3.4MW: DU97-W-300', '3.4MW: DU00-W2-350', '3.4MW: FX77-W-400', '3.4MW: FX77-W-500', '3.4MW: cylinder'])        # Let it pass even though the order is different.. (To solve the error under python=3.9)


def test_draw_airfoil_shape_callbacks():
    airfoil_names = ['3.4MW: DU08-W-210', '15MW: FFA-W3-301']       # Select 3.4MW: DU08-W-210, 15MW: FFA-W3-301
    fig, text = draw_airfoil_shape(airfoil_names, airfoil_by_names)
    answer = [html.P([html.B('3.4MW: DU08-W-210'), html.Br(), 'N/A']), html.P([html.B('15MW: FFA-W3-301'), html.Br(), f'FFA-W3-301 (Re=1.00e+07)FFA-W3 airfoil data for 10 MW sized rotor, computed using EllipSys2D v16.0, 70% free transition, 30% fully turbulent, 360 deg extrapolated using AirfoilPreppy, no 3D correction. F Zahle, DTU Wind Energy 11 May 2017'])]

    # Parse and remove irrelevant formatting (like extra spaces..)
    html_text = "".join([BeautifulSoup(str(html), 'html.parser').prettify() for html in text])
    html_answer = "".join([BeautifulSoup(str(html), 'html.parser').prettify() for html in answer])

    assert html_text == html_answer and len(fig['data']) == len(airfoil_names)


def test_draw_airfoil_polor_callbacks():
    airfoil_names = ['3.4MW: DU08-W-210', '15MW: FFA-W3-301']       # Select 3.4MW: DU08-W-210, 15MW: FFA-W3-301
    switches_value = [1, 3]
    fig = draw_airfoil_polar(airfoil_names, airfoil_by_names, switches_value)

    assert len(fig['data']) == len(airfoil_names) * len(switches_value) and ('$C_{D}$' not in [fig['layout']['yaxis']['title']['text'], fig['layout']['yaxis2']['title']['text']])
