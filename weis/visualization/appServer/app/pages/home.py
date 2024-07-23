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
                    dbc.Input(id='vizInput_path', placeholder='Enter input visualization file path..', type='text'),
                    dbc.Button('Reload', id='reload', n_clicks=0)
                    ], style={'width':'50vw', 'marginLeft': 50, 'marginTop': 50, 'display':'flex', 'justify-content':'space-between'}),
                html.Div([
                    html.H3('vizInputFile'),
                    dbc.Col(dcc.Loading(html.Div(id='input-cfg-div')))
                    ], style={'width':'50vw', 'marginLeft': 50, 'marginTop': 50})
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
          Input('vizInput_path', 'value'),
          Input('reload', 'n_clicks'))
def reload_input_file(contents, vizInput_path, btn):

    # Default
    if vizInput_path is None:
        updated_contents = parse_yaml(contents['yamlPath'])

    # Update yaml file
    if vizInput_path is not None and btn > 0:
        contents['yamlPath'] = vizInput_path
        updated_contents = parse_yaml(contents['yamlPath'])

    return updated_contents