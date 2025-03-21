'''This is the page for visualize the WEIS inputs specialized in Tower properties.'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio_tower'
)

@callback(Output('tower-names', 'options'),
          Output('tower-by-names', 'data'),
          Input('geometry-components', 'data'))
def load_tower_comps(geom_comps_by_names):
    '''
    This function is for loading tower related components
    '''
    tower_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'tower' in k}     # where now k is 'filelabelname' and v is dict
    
    return list(tower_by_names.keys()), tower_by_names


# We are using card container where we define sublayout with rows and cols.
def layout():

    # Define layout for tower structures
    tower_items = dcc.Dropdown(id='tower-names', options=[], value=None, multi=True)

    tower_inputs = html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Label('Please select towers:'),
                                dbc.Form(tower_items)
                            ])
                        ])
                    ])

    struct_layout = dbc.Card([
                        dbc.CardHeader('Structural Properties'),
                        dbc.CardBody([
                            dcc.Graph(id='tower-graph', figure=empty_figure())
                        ])
                  ], className='card')
    
    layout = dcc.Loading(html.Div([
                dcc.Store(id='tower-by-names', data={}),
                tower_inputs,
                struct_layout
            ]))
    
    return layout



@callback(Output('tower-graph', 'figure'),
          Input('tower-names', 'value'),
          Input('tower-by-names', 'data'))
def draw_tower(tower_names, tower_by_names):
    if tower_names is None:
        raise PreventUpdate
    
    cols = set_colors()            # Set color panel
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, specs=[[{'rowspan': 2}, {}], [None, {}]])
    
    for idx, tower_name in enumerate(tower_names):
        grid = tower_by_names[tower_name]['outer_shape_bem']['reference_axis']['z']['grid']
        z = tower_by_names[tower_name]['outer_shape_bem']['reference_axis']['z']['values']
        outer_diam = np.array(tower_by_names[tower_name]['outer_shape_bem']['outer_diameter']['values'])
        tower_layers = tower_by_names[tower_name]['internal_structure_2d_fem']['layers']    # TODO: values are list -- need to confirm if layer is fixed with one or it could be more than one?
        thickness = np.array([l['thickness']['values'] for l in tower_layers][0])        # original thickness = [[0.8 1.0 2.0 ...]]

        # Realistic view combined with tower height, outer diameter and thickness
        fig.append_trace(go.Scatter(
                    x = outer_diam,
                    y = z,
                    mode = 'lines+markers',
                    line=dict(color=cols[idx]),
                    name = '-'.join([tower_name, 'outer diameter']),
                    showlegend=False),
                    row = 1,
                    col = 1)
        
        fig.append_trace(go.Scatter(
                    x = outer_diam-thickness,
                    y = z,
                    mode = 'lines+markers',
                    line=dict(color=cols[idx]),
                    name = '-'.join([tower_name, 'inter diameter']),
                    showlegend=False),
                    row = 1,
                    col = 1)
        
        # Outer Diameter over grid
        fig.append_trace(go.Scatter(
                    x = grid,
                    y = outer_diam,
                    mode = 'lines+markers',
                    line=dict(color=cols[idx]),
                    name = tower_name,
                    showlegend=False),
                    row = 1,
                    col = 2)

        # Thickness over grid
        fig.append_trace(go.Scatter(
                    x = grid,
                    y = thickness,
                    mode = 'lines+markers',
                    line=dict(color=cols[idx]),
                    name = tower_name,
                    showlegend=True),
                    row = 2,
                    col = 2)
    
    fig.update_layout(plot_bgcolor='white', height=1000)
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'Dim', row=1, col=1)
    fig.update_yaxes(title_text=f'Height', row=1, col=1)
    fig.update_yaxes(title_text=f'Outer Diameter', row=1, col=2)
    fig.update_yaxes(title_text=f'Thickness', row=2, col=2)
    fig.update_xaxes(title_text=f'Grid', row=2, col=2)

    return fig