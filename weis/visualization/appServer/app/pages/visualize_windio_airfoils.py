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

    global airfoil_by_names     # Need to exclude this later (for simplicity for now)
    airfoils = load_airfoils()      # Data structure: [{'name': 'FB90', 'coordinates': {'x': [1.0, 0.9921, ...]}}]
    airfoil_by_names = {airfoil['name']: dict(list(airfoil.items())[1:]) for airfoil in airfoils}       # {'FB90': {'coordinates': {'x': [1.0, 0.9921, ...]}, ...}, ...}
    # airfoil_names = [airfoil['name'] for airfoil in airfoils]

    coords_layout = dbc.Card([
                                dbc.CardHeader("Airfoil Name: ", className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Dropdown(id='airfoil-name', options=list(airfoil_by_names.keys()), value=None, multi=False),
                                    dcc.Loading(html.P(id='airfoil-description')),
                                    dcc.Loading(dcc.Graph(id='airfoil-coords', figure=empty_figure())),
                                ])
                             ], className='card')
        
    polars_layout = dbc.Card([
                                dbc.CardHeader(html.P(id='airfoil-re'), className='cardHeader'),
                                dbc.CardBody([
                                    # dcc.Dropdown(id='airfoil-name', options=list(airfoil_by_names.keys()), value=None, multi=False),
                                    dcc.Loading(dcc.Graph(id='airfoil-polars', figure=empty_figure())),
                                ])
                             ], className='card')

    for airfoil in airfoils:
        airfoil_name = airfoil['name']
        airfoil_coords_x = airfoil['coordinates']['x']
        airfoil_coords_y = airfoil['coordinates']['y']



    layout = dbc.Row([
                dbc.Col(coords_layout, width=6),
                dbc.Col(polars_layout, width=6)
            ], className='wrapper')
    
    return layout


@callback(Output('airfoil-coords', 'figure'),
          Output('airfoil-description', 'children'),
          Input('airfoil-name', 'value'))
def draw_airfoil_shape(airfoil_name):
    if airfoil_name is None:
        raise PreventUpdate
    
    text = html.P(str(airfoil_by_names[airfoil_name]['description']))
    
    x = airfoil_by_names[airfoil_name]['coordinates']['x']
    y = airfoil_by_names[airfoil_name]['coordinates']['y']

    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)

    fig.append_trace(go.Scatter(
                x = x,
                y = y,
                mode = 'lines+markers'),
                row = 1,
                col = 1)
    
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'coords', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)

    print(airfoil_by_names[airfoil_name]['polars'])
    
    return fig, text


@callback(Output('airfoil-re', 'children'),
          Output('airfoil-polars', 'figure'),
          Input('airfoil-name', 'value'))
def draw_airfoil_polar(airfoil_name):
    if airfoil_name is None:
        raise PreventUpdate
    
    polars_dict = airfoil_by_names[airfoil_name]['polars'][0]
    airfoil_re = html.P('re: ' + str(polars_dict['re']))

    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)

    fig.append_trace(go.Scatter(
                x = np.rad2deg(polars_dict['c_l']['grid']),
                y = polars_dict['c_l']['values'],
                mode = 'lines',
                name = 'c_l'),
                row = 1,
                col = 1)

    fig.append_trace(go.Scatter(
                x = np.rad2deg(polars_dict['c_d']['grid']),
                y = polars_dict['c_d']['values'],
                mode = 'lines',
                name = 'c_d'),
                row = 1,
                col = 1)
    
    fig.append_trace(go.Scatter(
                x = np.rad2deg(polars_dict['c_m']['grid']),
                y = polars_dict['c_m']['values'],
                mode = 'lines',
                name = 'c_m'),
                row = 1,
                col = 1)

    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'grid', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)

    
    return airfoil_re, fig