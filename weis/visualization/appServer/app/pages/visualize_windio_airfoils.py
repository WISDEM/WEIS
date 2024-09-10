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

    airfoils = load_airfoils()

    return layout