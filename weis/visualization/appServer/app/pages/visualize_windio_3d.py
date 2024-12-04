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
    path='/windio_3d'
)


@callback(Output('geom-3d-names', 'options'),
          Input('geometry-components', 'data'))
def list_labels(geom_comps_by_names):
    return list(set([k.split(':')[0] for k in list(geom_comps_by_names.keys())]))


@callback(Output('meshes', 'data'),
          Input('geom-3d-names', 'value'),
          Input('geometry-components', 'data'))
def visualize(geom_3d_names, geom_comps_by_names):
    '''
    This function is for visualize per geometry component types
    '''
    if geom_3d_names is None:
        raise PreventUpdate
    
    tower_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'tower' in k}     # where now k is 'filelabelname' and v is dict

    meshes = []
    for name in geom_3d_names:
        meshes += render_cylinderTower(tower_by_names[name]['outer_shape_bem']['outer_diameter']['grid'], 
                                        tower_by_names[name]['outer_shape_bem']['outer_diameter']['values'])
    
    return meshes


@callback(Output('mesh-layout', 'children'),
          Input('meshes', 'data'))
def load_meshes(meshes):
    layout = [dbc.Row([
                        dbc.Row([
                            # First column
                            dbc.Col(html.Div(
                                style={'width':'100%', 'height':'600px'},
                                children=[meshes[idx]])),
                            # Second column
                            dbc.Col(html.Div(
                                style={'width':'100%', 'height':'600px'},
                                children=[meshes[idx+1]]))
                        ], className='wrapper') for idx in range(0, len(meshes)-1, 2)], className='wrapper')]
    

    return layout


# We are using card container where we define sublayout with rows and cols.
def layout():

    # Define layout for tower structures
    geom_items = dcc.Dropdown(id='geom-3d-names', options=[], value=None, multi=True)

    geom_inputs = html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Label('Please select Geometry:'),
                                dbc.Form(geom_items)
                            ])
                        ])
                    ])
    
    mesh_layout = html.Div(id='mesh-layout')

    # TODO: Should be able to upload odd number of meshes as well
    layout = dcc.Loading(html.Div([
                dcc.Store(id='meshes', data=[]),
                geom_inputs,
                mesh_layout
            ], className='wrapper'))

    return layout