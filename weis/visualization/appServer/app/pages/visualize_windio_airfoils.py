'''This is the page for visualize the WEIS inputs specialized in Airfoil properties.'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, State, dcc
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio_airfoils'
)

@callback(Output('airfoil-names', 'options'),
          Input('airfoil-by-names', 'data'))
def list_airfoils(airfoil_by_names):
    return list(airfoil_by_names.keys())


# We are using card container where we define sublayout with rows and cols.
def layout():

    # Define layout for airfoil coords
    airfoil_items = dcc.Dropdown(id='airfoil-names', options=[], value=None, multi=True)

    coords_inputs = html.Div([
                        dbc.Label('Please select airfoils:'),
                        dbc.Form(airfoil_items)
                    ])


    coords_layout = dbc.Card([
                                dbc.CardHeader('Airfoil Coordinates'),
                                dbc.CardBody([
                                    coords_inputs,
                                    html.Br(),
                                    html.P(id='airfoil-description'),
                                    html.Br(),
                                    dcc.Graph(id='airfoil-coords', figure=empty_figure())
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
                                dbc.CardHeader('Airfoil Polars'),
                                dbc.CardBody([
                                    # Toggle switches
                                    polars_inputs,
                                    html.Br(),
                                    # Graph layout for polars
                                    dcc.Graph(id='airfoil-polars', figure=empty_figure(), mathjax=True)
                                ])
                             ], className='card')

    layout = dcc.Loading(html.Div([
                dbc.Row([
                    dbc.Col(coords_layout, width=6),
                    dbc.Col(polars_layout, width=6)
                ], className='g-0')         # No gutters where horizontal spacing is added between the columns by default
            ]))
    
    return layout


@callback(Output('airfoil-coords', 'figure'),
          Output('airfoil-description', 'children'),
          Input('airfoil-names', 'value'),
          State('airfoil-by-names', 'data'))
def draw_airfoil_shape(airfoil_names, airfoil_by_names):
    if airfoil_names is None:
        raise PreventUpdate
    
    cols = set_colors()            # Set color panel
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
    
    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.3), margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', scaleanchor='x')      # Make it 1:1 xy-ratio
    fig.update_xaxes(title_text=f'coords', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)

    return fig, text


@callback(Output('airfoil-polars', 'figure'),
          Input('airfoil-names', 'value'),
          State('airfoil-by-names', 'data'),
          Input('switch-polar', 'value'))
def draw_airfoil_polar(airfoil_names, airfoil_by_names, switches_value):
    if airfoil_names is None:
        raise PreventUpdate
    
    if len(switches_value) == 0:
        return empty_figure()
    
    cols = set_colors()            # Set color panel

    # Update dynamic subplots graph with selected airfoils options
    fig = make_subplots(rows = len(switches_value), cols = 1, shared_xaxes=True)
    for row_idx, value in enumerate(switches_value):
        for idx, airfoil_name in enumerate(airfoil_names):
            polars_dict = airfoil_by_names[airfoil_name]['polars'][0]

            if value == 1:
                fig.append_trace(go.Scatter(
                            x = np.rad2deg(polars_dict['c_l']['grid']),
                            y = polars_dict['c_l']['values'],
                            mode = 'lines+markers',
                            marker=dict(symbol='diamond', color=cols[idx]),
                            line=dict(color=cols[idx]),
                            name = f'{dcc.Markdown("$C_{}, Re={}$".format("L", polars_dict["re"]), mathjax=True)}'),
                            row = row_idx+1,
                            col = 1)
                fig.update_yaxes(title_text='$C_{L}$', row=row_idx+1, col=1)

            elif value == 2:
                fig.append_trace(go.Scatter(
                            x = np.rad2deg(polars_dict['c_d']['grid']),
                            y = polars_dict['c_d']['values'],
                            mode = 'lines+markers',
                            marker=dict(symbol='arrow', angleref='previous', size=10, color=cols[idx]),
                            line=dict(color=cols[idx]),
                            name = f'{dcc.Markdown("$C_{}, Re={}$".format("D", polars_dict["re"]), mathjax=True)}'),
                            row = row_idx+1,
                            col = 1)
                fig.update_yaxes(title_text='$C_{D}$', row=row_idx+1, col=1)
            
            elif value == 3:
                fig.append_trace(go.Scatter(
                            x = np.rad2deg(polars_dict['c_m']['grid']),
                            y = polars_dict['c_m']['values'],
                            mode = 'lines+markers',
                            marker=dict(symbol='cross', color=cols[idx]),
                            line=dict(color=cols[idx]),
                            name = f'{dcc.Markdown("$C_{}, Re={}$".format("M", polars_dict["re"]), mathjax=True)}'),
                            row = row_idx+1,
                            col = 1)
                fig.update_yaxes(title_text='$C_{M}$', row=row_idx+1, col=1)

    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.2), margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=300 * len(switches_value))
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text='$\\alpha [^\\circ]$', row=len(switches_value), col=1)

    
    return fig