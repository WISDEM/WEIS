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

@callback(Output('dummy-div-1', 'children'),
          Input('geometry-components', 'data'))
def visualize(geom_comps_by_names):
    '''
    This function is for visualize per geometry component types
    '''
    for k, v in geom_comps_by_names.items():        # where k is 'filelabelname:componenttype'
        print(k, v)
        # TODO: Add something here
    
    return html.P('')


# We are using card container where we define sublayout with rows and cols.
def layout():
    # Render mutliple meshes from raw data
    meshes = render_meshes()

    # TODO: Should be able to upload odd number of meshes as well
    layout = dbc.Row(
                [html.Div(id='dummy-div-1')]+
                [
                dbc.Row([
                    # First column
                    dbc.Col(html.Div(
                        style={'width':'100%', 'height':'600px'},
                        children=[meshes[idx]])),
                    # Second column
                    dbc.Col(html.Div(
                        style={'width':'100%', 'height':'600px'},
                        children=[meshes[idx+1]]))
                ], className='wrapper') for idx in range(0, len(meshes)-1, 2)], className='wrapper')
    

    return layout