'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import pandas as pd
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots
import plotly
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

def set_colors():
    global cols
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS


# We are using card container where we define sublayout with rows and cols.
def layout():

    # Load/process airfoils data
    categorize_airfoils()

    # Set color panel
    set_colors()

    # Define layout for airfoil coords
    airfoil_items = dcc.Dropdown(id='airfoil-names', options=list(airfoil_by_names.keys()), value=None, multi=True)

    coords_inputs = html.Div([
                        dbc.Label('Please select airfoils:'),
                        dbc.Form(airfoil_items)
                    ])


    coords_layout = dbc.Card([
                                dbc.CardHeader("Airfoil Coordinates", className='cardHeader'),
                                dbc.CardBody([
                                    coords_inputs,
                                    html.Br(),
                                    dcc.Loading(html.P(id='airfoil-description')),
                                    dcc.Loading(dcc.Graph(id='airfoil-coords', figure=empty_figure())),
                                ])
                             ], className='card')
        

    # Define layout for airfoil polars
    polars_switches = dbc.Checklist(
                        options=[
                            {'label': dcc.Markdown('Lift $C_{L}$', mathjax=True), 'value': 1},
                            {'label': dcc.Markdown('Drag $C_{D}$', mathjax=True), 'value': 2},
                            {'label': dcc.Markdown('Moment $C_{M}$', mathjax=True), 'value': 3}
                        ],
                        value=[1],      # Defualt with only c_l turned on
                        id='switch-polar',
                        inline=True,
                        switch=True,
                    )
    
    polars_inputs = html.Div([
                        dbc.Label('Please select coefficients to visualize:'),
                        dbc.Form(polars_switches)
                    ])

    polars_layout = dbc.Card([
                                dbc.CardHeader("Airfoil Polars", className='cardHeader'),
                                dbc.CardBody([
                                    # Toggle switches
                                    polars_inputs,
                                    html.P(id='airfoil-re'),
                                    # Graph layout for polars
                                    dcc.Loading(dcc.Graph(id='airfoil-polars', figure=empty_figure(), mathjax=True)),
                                ])
                             ], className='card')

    layout = dbc.Row([
                dbc.Col(coords_layout, width=6),
                dbc.Col(polars_layout, width=6)
            ], className='wrapper')
    
    return layout


@callback(Output('airfoil-coords', 'figure'),
          Output('airfoil-description', 'children'),
          Input('airfoil-names', 'value'))
def draw_airfoil_shape(airfoil_names):
    if airfoil_names is None:
        raise PreventUpdate
    
    text = []
    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)

    for idx, airfoil_name in enumerate(airfoil_names):
        # Define description text (which is not must-have)
        text.append(html.P([html.B(airfoil_name), html.Br(), airfoil_by_names[airfoil_name]['description']]) if 'description' in airfoil_by_names[airfoil_name] else html.P([html.B(airfoil_name), html.Br(), 'N/A']))
        # Define graph -- add a trace per airfoil
        x = airfoil_by_names[airfoil_name]['coordinates']['x']
        y = airfoil_by_names[airfoil_name]['coordinates']['y']

        fig.append_trace(go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    line=dict(color=cols[idx]),
                    name = airfoil_name),
                    row = 1,
                    col = 1)
    
    # fig.update_layout(plot_bgcolor='white', legend={'x':0.0, 'y':-0.75})      # Relocate legend position
    # fig.update_layout(plot_bgcolor='white', legend=dict(yanchor='bottom', y=-0.75, xanchor='left', x=0.01))
    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.3))
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'coords', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)
    
    return fig, text


@callback(Output('airfoil-re', 'children'),
          Output('airfoil-polars', 'figure'),
          Input('airfoil-names', 'value'),
          Input('switch-polar', 'value'))
def draw_airfoil_polar(airfoil_names, switches_value):
    if airfoil_names is None:
        raise PreventUpdate
    
    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)
    for idx, airfoil_name in enumerate(airfoil_names):
        polars_dict = airfoil_by_names[airfoil_name]['polars'][0]
        airfoil_re = html.P('re: ' + str(polars_dict['re']))
        # re_value = str(polars_dict['re'])

        if 1 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_l']['grid']),
                        y = polars_dict['c_l']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='diamond', color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{L}$", mathjax=True)}'),
                        # name = '$C_{L}, Re=$' + str(polars_dict['re'])),            # Doesn't work
                        # name = dcc.Markdown('$C_{L}$ f"({airfoil_name})"', mathjax=True)),      # Doesn't work
                        # name = f'{dcc.Markdown("$C_{L}$", mathjax=True)} ({airfoil_name})'),        # No error but airfoil name doesn't come up
                        row = 1,
                        col = 1)

        if 2 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_d']['grid']),
                        y = polars_dict['c_d']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='arrow', angleref='previous', size=10, color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{D}$", mathjax=True)}'),
                        row = 1,
                        col = 1)
        
        if 3 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_m']['grid']),
                        y = polars_dict['c_m']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='cross', color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{M}$", mathjax=True)}'),
                        row = 1,
                        col = 1)

    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.3))
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text='$\\alpha [^\\circ]$', row=1, col=1)
    
    return airfoil_re, fig