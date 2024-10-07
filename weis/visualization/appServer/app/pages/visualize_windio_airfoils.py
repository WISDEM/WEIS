'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import pandas as pd
import numpy as np
from pathlib import Path
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

def categorize_airfoils():
    global airfoil_by_names         # Need to exclude this later (for simplicity for now)
    airfoils = load_airfoils()      # Data structure: {file1: [{'name': 'FB90', 'coordinates': {'x': [1.0, 0.9921, ...]}}], file2: ~~~}
    airfoil_by_names = {Path(filepath).stem+': '+airfoil['name']: dict(list(airfoil.items())[1:]) for filepath, airfoils_by_file in airfoils.items() for airfoil in airfoils_by_file}      # {'file1: FB90': {'coordinates': {'x': [1.0, 0.9921, 0.5], 'y': [1.0, 2.0, 3.0]}}, ... }
    # airfoil_by_names = {airfoil['name']: dict(list(airfoil.items())[1:]) for airfoil in airfoils}       # {'FB90': {'coordinates': {'x': [1.0, 0.9921, ...]}, ...}, ...}
    # airfoil_names = [airfoil['name'] for airfoil in airfoils]




# We are using card container where we define sublayout with rows and cols.
def layout():

    categorize_airfoils()

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
    
    # Define description text (which is not must-have)
    text = html.P(str(airfoil_by_names[airfoil_name]['description'])) if 'description' in airfoil_by_names[airfoil_name] else html.P('Description: None')
    
    x = airfoil_by_names[airfoil_name]['coordinates']['x']
    y = airfoil_by_names[airfoil_name]['coordinates']['y']

    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)

    fig.append_trace(go.Scatter(
                x = x,
                y = y,
                mode = 'lines'),
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
                mode = 'lines+markers',
                marker=dict(symbol='diamond'),
                name = 'c_l'),
                row = 1,
                col = 1)

    fig.append_trace(go.Scatter(
                x = np.rad2deg(polars_dict['c_d']['grid']),
                y = polars_dict['c_d']['values'],
                mode = 'lines+markers',
                marker=dict(symbol='arrow', angleref='previous'),
                name = 'c_d'),
                row = 1,
                col = 1)
    
    fig.append_trace(go.Scatter(
                x = np.rad2deg(polars_dict['c_m']['grid']),
                y = polars_dict['c_m']['values'],
                mode = 'lines+markers',
                marker=dict(symbol='diamond-open'),
                name = 'c_m'),
                row = 1,
                col = 1)

    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'grid', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)

    
    return airfoil_re, fig