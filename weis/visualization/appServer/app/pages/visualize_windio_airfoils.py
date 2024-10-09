'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc, dash_table
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
                                    html.Br(),
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
                                    # Table for Re
                                    # html.Div(id='re-table'),      # For callback: Output('re-table', 'children'),
                                    html.Br(),
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
    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.3), margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', scaleanchor='x')      # Make it 1:1 xy-ratio
    fig.update_xaxes(title_text=f'coords', row=1, col=1)
    fig.update_yaxes(title_text=f'value', row=1, col=1)
    
    return fig, text


@callback(Output('airfoil-polars', 'figure'),
          Input('airfoil-names', 'value'),
          Input('switch-polar', 'value'))
def draw_airfoil_polar(airfoil_names, switches_value):
    if airfoil_names is None:
        raise PreventUpdate
    
    if len(switches_value) == 0:
        return empty_figure()
    
    # Update dynamic subplots graph with selected airfoils options
    fig = make_subplots(rows = len(switches_value), cols = 1, shared_xaxes=True)
    for row_idx, value in enumerate(switches_value):
        for idx, airfoil_name in enumerate(airfoil_names):
            polars_dict = airfoil_by_names[airfoil_name]['polars'][0]
            # re_data.append(html.Tr([html.Td(airfoil_name), html.Td(polars_dict['re'])]))            # For 1)
            # re_data.append({'airfoil': airfoil_name, 'Re': polars_dict['re']})                      # For 2)

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


    
    # Update graph with selected airfoils options
    '''
    # re_data = []
    fig = make_subplots(rows = 1, cols = 1, shared_xaxes=True)
    for idx, airfoil_name in enumerate(airfoil_names):
        polars_dict = airfoil_by_names[airfoil_name]['polars'][0]
        # re_data.append(html.Tr([html.Td(airfoil_name), html.Td(polars_dict['re'])]))            # For 1)
        # re_data.append({'airfoil': airfoil_name, 'Re': polars_dict['re']})                      # For 2)

        if 1 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_l']['grid']),
                        y = polars_dict['c_l']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='diamond', color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{}, Re={}$".format("L", polars_dict["re"]), mathjax=True)}'),
                        row = 1,
                        col = 1)

        if 2 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_d']['grid']),
                        y = polars_dict['c_d']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='arrow', angleref='previous', size=10, color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{}, Re={}$".format("D", polars_dict["re"]), mathjax=True)}'),
                        row = 1,
                        col = 1)
        
        if 3 in switches_value:
            fig.append_trace(go.Scatter(
                        x = np.rad2deg(polars_dict['c_m']['grid']),
                        y = polars_dict['c_m']['values'],
                        mode = 'lines+markers',
                        marker=dict(symbol='cross', color=cols[idx]),
                        line=dict(color=cols[idx]),
                        name = f'{dcc.Markdown("$C_{}, Re={}$".format("M", polars_dict["re"]), mathjax=True)}'),
                        row = 1,
                        col = 1)

    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', xanchor='center', x=0.5, y=-0.3), margin={"l": 0, "r": 0, "t": 0, "b": 0})
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text='$\\alpha [^\\circ]$', row=1, col=1)
    '''


    # Update Re data table with selected airfoils
    # 1) Nice layout but no interaction/responsive
    # table_header = [html.Thead(html.Tr([html.Th('Airfoil Name'), html.Th('Re')]))]
    # table_body = [html.Tbody(re_data)]
    # airfoil_re = html.Div([
    #                 dbc.Table(
    #                     table_header + table_body,
    #                     color='primary',
    #                     bordered=True
    #                 )
    #             ])

    # 2) Allow interaction but poor design
    # airfoil_re = html.Div([dash_table.DataTable(
    #     columns=[
    #         {"name": i, "id": i} for i in ['airfoil', 'Re']
    #     ],
    #     data=re_data,
    #     sort_action="native",
    #     sort_mode="multi",
    #     # style_data_conditional=[
    #     #     {
    #     #         "if": {
    #     #             "filter_query": '{airfoil} = {}'.format(airfoil_name),
    #     #         },
    #     #         "color": cols[idx],
    #     #         "fontWeight": "bold"
    #     #     } for idx, airfoil_name in enumerate(airfoil_names)
    #     # ],
    #     style_data={
    #         "color": "black",
    #         "backgroundColor": "white",
    #         "textAlign": "center"
    #     },
    #     style_header={
    #         "backgroundColor": "rgba(210, 210, 210, 0.65)",
    #         "color": "black",
    #         "fontWeight": "bold",
    #         "textAlign": "center"
    #     }

    # )])

    # 2) Allow interaction but poor design
    # airfoil_re = html.Div([dash_table.DataTable(
    #     columns=[
    #         {"name": i, "id": i, "deletable": True, "selectable": True} for i in ['airfoil name', 'Re']
    #     ],
    #     data=re_data,
    #     editable=True,
    #     filter_action="native",
    #     sort_action="native",
    #     sort_mode="multi",
    #     column_selectable="single",
    #     row_selectable="multi",
    #     row_deletable=True,
    #     selected_columns=[],
    #     selected_rows=[],
    #     page_action="native",
    #     page_current= 0,
    #     page_size= 10,
    # )])
    
    
    return fig