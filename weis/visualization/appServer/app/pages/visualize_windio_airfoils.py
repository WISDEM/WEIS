'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import weis.inputs as sch
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio_airfoils'
)

# @callback(Output('var-windio', 'data'),
#           Input('input-dict', 'data'))
# def read_variables(input_dict):
    
#     if input_dict is None or input_dict == {}:
#         raise PreventUpdate
    
#     # windio_options = parse_yaml(input_dict['wtInputPath'])
#     # windio_options = input_dict['wtInputPath']
#     windio_options = sch.load_geometry_yaml(input_dict['wtInputPath'])
#     print('windio_options\n', windio_options)

#     return windio_options


# We are using card container where we define sublayout with rows and cols.
def layout():

    airfoils = load_airfoils()      # Data structure: [{'name': 'FB90', 'coordinates': {'x': [1.0, 0.9921, ...]}}]
    airfoil_by_names = {airfoil['name']: dict(list(airfoil.items())[1:]) for airfoil in airfoils}       # {'FB90': {'coordinates': {'x': [1.0, 0.9921, ...]}, ...}, ...}
    # airfoil_names = [airfoil['name'] for airfoil in airfoils]

    coords_layout = dbc.Card([
                                dbc.CardHeader("Airfoil Name: ", className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Dropdown(id='airfoil-name', options=list(airfoil_by_names.keys()), value=None, multi=False),
                                    dcc.Loading(dcc.Graph(id='airfoil-coords', figure=empty_figure())),
                                ])
                             ], className='card')
        
    polars_layout = dbc.Card([
                                dbc.CardHeader("Airfoil Name: ", className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Dropdown(id='airfoil-name', options=list(airfoil_by_names.keys()), value=None, multi=False),
                                    dcc.Loading(dcc.Graph(id='airfoil-polars', figure=empty_figure())),
                                ])
                             ], className='card')

    for airfoil in airfoils:
        airfoil_name = airfoil['name']
        airfoil_coords_x = airfoil['coordinates']['x']
        airfoil_coords_y = airfoil['coordinates']['y']



    layout = dbc.Row([
                dbc.Col(coords_layout, width=5),
                dbc.Col(polars_layout, width=6)
            ], className='wrapper')
    
    return layout


@callback(Output('airfoil-coords', 'figure'),
          Input('airfoil-name', 'value'))
def draw_blade_shape(blade_options):
    if blade_options is None:
        raise PreventUpdate
    
    x = blade_options['x']
    ys = blade_options['ys']

    fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True)

    for y in ys:
        if y == 'rotorse.theta_deg':
            fig.append_trace(go.Scatter(
                x = refturb[x],
                y = refturb[y],
                mode = 'lines+markers',
                name = y),
                row = 2,
                col = 1)
        else:
            fig.append_trace(go.Scatter(
                x = refturb[x],
                y = refturb[y],
                mode = 'lines+markers',
                name = y),
                row = 1,
                col = 1)
        
    
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'rotorse.rc.s', row=2, col=1)
    
    return fig