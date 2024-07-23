from dash import html, register_page
from dash import dcc, Input, State, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from weis.visualization.utils import parse_yaml, dict_to_html

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():

    layout = dbc.Row([
                dbc.InputGroup([
                    html.H3('vizInputFile'),
                    dbc.Button('Reload VizInputFile', id='reload', n_clicks=0)
                ], style={'width':'35vw', 'marginLeft': 50, 'marginTop': 50, 'display':'flex', 'justify-content':'space-between'}),
                dbc.Col(dcc.Loading(html.Div(id='input-cfg-div')))
    ])

    return layout


@callback(Output('input-cfg-div', 'children'),
          Input('input-dict', 'data'))
def check_input_file(contents):
    '''
    Store data in mainApp.py so that it's accessible over pages.
    Show if input file data has been loaded and parsed successfully
    '''
    if contents is None:
        raise PreventUpdate
    
    if contents == {}:
        return html.Div([html.H5("Empty content..")])
    
    file_tree_list = dict_to_html(contents, [], level=1)
    
    return html.Div([*file_tree_list], style={'width':'80vw', 'marginLeft': 100, 'border-left-style':'dotted'})


@callback(Output('input-dict', 'data'),
          State('input-dict', 'data'),
          Input('reload', 'n_clicks'))
def reload_input_file(contents, btn):
    updated_contents = parse_yaml(contents['yamlPath'])

    return updated_contents

