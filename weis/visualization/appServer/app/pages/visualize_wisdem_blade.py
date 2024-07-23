'''This is the page for visualize the WISDEM outputs specialized in blade properties'''

# TODO: Merge ys_struct_log into ys_struct and distinguish them by value magnitude?
# TODO: Do we need dropout list here to let user change variables? (Show default settings..)

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from weis.visualization.utils import empty_figure

register_page(
    __name__,
    name='WISDEM',
    top_nav=True,
    path='/wisdem_blade'
)

@callback(Output('var-wisdem-blade', 'data'),
          Input('input-dict', 'data'))
def read_variables(input_dict):
    # TODO: Redirect to the home page when missing input yaml file
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    # Read WISDEM output data
    global refturb, refturb_variables

    wisdem_output_path = input_dict['userPreferences']['wisdem']['output_path']
    npz_filepath = '/'.join([wisdem_output_path, f'{input_dict['userOptions']['output_fileName']}.npz'])
    csv_filepath = '/'.join([wisdem_output_path, f'{input_dict['userOptions']['output_fileName']}.csv'])
    refturb = np.load(npz_filepath)
    refturb_variables = pd.read_csv(csv_filepath).set_index('variables').to_dict('index')

    blade_options = {}
    blade_options['x'] = input_dict['userPreferences']['wisdem']['blade']['xaxis']
    blade_options['ys'] = input_dict['userPreferences']['wisdem']['blade']['shape_yaxis']
    blade_options['ys_struct_log'] = input_dict['userPreferences']['wisdem']['blade']['struct_yaxis_log']
    blade_options['ys_struct'] = input_dict['userPreferences']['wisdem']['blade']['struct_yaxis']

    print("Parse variables from wisdem blade..\n", blade_options)

    return blade_options


def layout():

    description_layout = dbc.Card(
                            [
                                dbc.CardHeader("Blade channels description", className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Loading(html.P(id='description'))
                                ])
                            ], className='card')
    
    plots1_layout = dbc.Card(
                        [
                            dbc.CardHeader('Blade Shape Properties', className='cardHeader'),
                            dbc.CardBody([
                                dcc.Loading(dcc.Graph(id='blade-shape', figure=empty_figure())),
                            ])
                        ], className='card')
    plots2_layout = dbc.Card(
                        [
                            dbc.CardHeader('Blade Structure Properties', className='cardHeader'),
                            dbc.CardBody([
                                dcc.Loading(dcc.Graph(id='blade-structure', figure=empty_figure())),
                            ])
                        ], className='card')

    layout = dbc.Row([
                # dcc.Location(id='url', refresh=False),
                dcc.Store(id='var-wisdem-blade', data={}),
                dbc.Col(description_layout, width=3),
                dbc.Col([
                    dbc.Row(plots1_layout),
                    dbc.Row(plots2_layout)
                ], width=8)
            ], className='wrapper')
    

    return layout


@callback(Output('description', 'children'),
          Input('var-wisdem-blade', 'data'))
def get_description(blade_options):
    if blade_options is None:
        raise PreventUpdate
    
    des_list = []
    channel_list = [blade_options['x']] + blade_options['ys'] + blade_options['ys_struct_log'] + blade_options['ys_struct']
    print("channel_list\n", channel_list)
    # Need to specify where channel names are saved differently..
    npz_to_csv = {'rotorse.rc.chord_m': 'rotorse.rc.chord', 'rotorse.theta_deg': 'rotorse.theta', 'rotorse.EA_N': 'rotorse.EA', 'rotorse.EIxx_N*m**2': 'rotorse.EIxx', 'rotorse.EIyy_N*m**2': 'rotorse.EIyy', 'rotorse.GJ_N*m**2': 'rotorse.GJ', 'rotorse.rhoA_kg/m': 'rotorse.rhoA'}
    for chan in channel_list:
        if chan in npz_to_csv.keys():
            value = refturb_variables[npz_to_csv[chan]]
            des = npz_to_csv[chan]
        else:
            value = refturb_variables[chan]
            des = chan
        
        if not pd.isna(value['units']):
            des += ' ('+value['units']+'): '+value['description']
        else:
            des += ' : '+value['description']

        des_list.append(html.P(des))

    return des_list


@callback(Output('blade-shape', 'figure'),
          Input('var-wisdem-blade', 'data'))
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


@callback(Output('blade-structure', 'figure'),
          Input('var-wisdem-blade', 'data'))
def draw_blade_structure(blade_options):
    if blade_options is None:
        raise PreventUpdate

    x = blade_options['x']
    ys_struct = blade_options['ys_struct']
    ys_struct_log = blade_options['ys_struct_log']

    fig = make_subplots(specs=[[{"secondary_y": True}], [{"secondary_y": False}]], rows=2, cols=1, shared_xaxes=True)
    for y in ys_struct:
        fig.add_trace(go.Scatter(
            x = refturb[x],
            y = refturb[y],
            mode = 'lines+markers',
            name = y),
            row = 2,
            col = 1)
    for y in ys_struct_log:
        fig.add_trace(go.Scatter(
            x = refturb[x],
            y = refturb[y],
            mode = 'lines+markers',
            name = y),
            secondary_y=True,
            row = 1,
            col = 1)

    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(type="log", secondary_y=True)
    fig.update_yaxes(title_text="<b>primary</b> yaxis", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis with log", secondary_y=True)
    fig.update_xaxes(title_text=f'rotorse.rc.s', row=2, col=1)


    return fig

