'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio'
)


# We are using card container where we define sublayout with rows and cols.
def layout():

    # 1) Render mesh by loading 3d model from stl or vtk files
    # Sample files to test
    # file_path = '/Users/sryu/Desktop/FY24/windio2cad/nrel5mw-semi_oc4.stl'
    # file_path = '/Users/sryu/Desktop/FY24/ACDC/project1/case01/vtk/01_NREL_5MW-ED.Mode1.LinTime1.ED_BladeLn2Mesh_motion1.010.vtp'
    # content = load_mesh(file_path)

    # 2) Create mesh from scratch
    content = render_mesh_with_grid()

    layout = html.Div(
        style={'width':'100%', 'height':'600px'},
        children=[content]
    )

    return layout