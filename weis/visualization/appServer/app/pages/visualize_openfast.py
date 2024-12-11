'''This is the page for visualizing table and plots of OpenFAST output'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from either Input or State. Update the output.
'''

# Import Packages
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, register_page, ctx
from dash.exceptions import PreventUpdate
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from weis.visualization.utils import store_dataframes, get_file_info, update_yaml

register_page(
    __name__,
    name='OpenFAST',
    top_nav=True,
    path='/open_fast'
)

file_indices = ['file1', 'file2', 'file3', 'file4', 'file5']       # Need to define as the way defined in .yaml file - max 5

###############################################
#   Read openfast related variables from yaml file
###############################################

@callback(Output('var-openfast', 'data'),
          Output('var-openfast-graph', 'data'),
          [[Output(f'df-{idx}', 'data') for idx in file_indices]],
          Input('input-dict', 'data'))
def read_default_variables(input_dict):
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    of_options = {}
    var_openfast = input_dict['userPreferences']['openfast']
    var_files = var_openfast['file_path']
    dfs = store_dataframes(var_files)       # [{file1: df1, file2: df2, ... }]

    of_options['graph_x'] = var_openfast['graph']['xaxis']
    of_options['graph_y'] = var_openfast['graph']['yaxis']

    print("Parse variables from open fast..\n", of_options)

    return var_openfast, of_options, dfs


###############################################
#   Basic layout definition
###############################################
# We are using card container where we define sublayout with rows and cols.
def layout():

    #######################################
    # Layout for OpenFAST Viz Options
    #######################################
    signaly_input = dbc.Row([
                        dbc.Label('Signal-Y', width=2),
                        dbc.Col(
                            dcc.Dropdown(id='signaly', options=[], value=None, multi=True),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                            width=7
                        ),
                        dbc.Col(
                            dbc.Button('Save', id='save-of', n_clicks=0, color='primary'),
                            width='auto'
                        )
                ], className="mb-3")
    
    signalx_input = dbc.Row([
                        dbc.Label('Signal-X', width=2),
                        dbc.Col(
                            dcc.Dropdown(id='signalx', options=[], value=None),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                            width=7
                        )
                    ], className="mb-3")

    plotoption_input = dbc.Row([
                            dbc.Label('Plot Option', width=2),
                            dbc.Col(
                                dbc.RadioItems(
                                    id='plotOption',
                                    options=[
                                        {'label': 'Single', 'value': 'single plot'},
                                        {'label': 'Multiple', 'value': 'multiple plot'},
                                    ],
                                    value = 'multiple plot'
                                ),
                                width='auto'
                            )
                        ], className="mb-3")

    form_layout = dbc.Card([
                            dbc.CardHeader('Channels'),
                            dbc.CardBody([
                                dbc.Form([signaly_input, signalx_input, plotoption_input]),
                            ])
                  ], className='card')
    
    
    layout = dcc.Loading(html.Div([
                # Confirm Dialog to check updated
                dcc.ConfirmDialog(
                    id='confirm-update-of',
                    message='Updated'
                ),
                dbc.Row([
                    dbc.Col(form_layout, width=4),              # Channels
                    dbc.Col(html.Div(id='output'), width=8)     # Append cards per file
                ], className='g-0')         # No gutters where horizontal spacing is added between the columns by default
            ]))
    
    return layout


###############################################
#   Update graph configuration layout - first row
###############################################

@callback(Output('signaly', 'options'),
          Output('signaly', 'value'),
          Output('signalx', 'options'),
          Output('signalx', 'value'),
          Input('df-file1', 'data'),
          Input('var-openfast-graph', 'data'))
def define_graph_cfg_layout(df1, of_options):

    if df1 is None or df1 == {}:
        raise PreventUpdate
    
    channels = sorted(df1['file1'][0].keys())
    # print(df_dict['file1'][0])          # First row channels

    return channels, of_options['graph_y'], channels, of_options['graph_x']


###############################################
#   Update file description layout
###############################################

def define_des_layout(file_info, df):
    file_abs_path = file_info['file_abs_path']
    file_size = file_info['file_size']
    creation_time = file_info['creation_time']
    modification_time = file_info['modification_time']
    
    return html.Div([
                # File Info
                html.P(f'File Path: {file_abs_path}'),
                html.P(f'File Size: {file_size} MB'),
                html.P(f'Creation Date: {datetime.datetime.fromtimestamp(creation_time)}'),
                html.P(f'Modification Date: {datetime.datetime.fromtimestamp(modification_time)}')
            ])


###############################################
#   Update graph layout per card
###############################################

def update_figure(signalx, signaly, plotOption, df_dict):
    df, = df_dict.values()
    return draw_graph(signalx, signaly, plotOption, pd.DataFrame(df))


for idx in file_indices:
    callback(Output(f'graph-div-{idx}', 'figure'),
                State('signalx', 'value'),
                State('signaly', 'value'),
                State('plotOption', 'value'),
                Input(f'df-{idx}', 'data'))(update_figure)


def draw_graph(signalx, signaly, plotOption, df):
    # Whenever signalx, signaly, plotOption has been entered, draw the graph.
    # Create figure with that setting and add that figure to the graph layout.
    # Note that we set default settings (see analyze() function), it will show corresponding default graph.
    # You can dynamically change signalx, signaly, plotOption, and it will automatically update the graph.

    # Put all traces in one single plot
    if plotOption == 'single plot':
        fig = make_subplots(rows = 1, cols = 1)
        for col_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                name = label),
                row = 1,
                col = 1)
        

    # Put each traces in each separated vertically aligned subplots
    elif plotOption == 'multiple plot':
        fig = make_subplots(rows = len(signaly), cols = 1, shared_xaxes=True, vertical_spacing=0.05)

        for row_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                name = label),
                row = row_idx + 1,
                col = 1)
            fig.update_yaxes(title_text=label, row=row_idx+1, col=1)
    
        fig.update_layout(height=150 * len(signaly))
        fig.update_xaxes(title_text=signalx, row=len(signaly), col=1)

    return fig


###############################################
#   Dynamic card creation
###############################################

def make_card(idx, file_path, df):
    file_info = get_file_info(file_path)
    file_abs_path = file_info['file_abs_path']

    return dbc.Card([
        dbc.CardHeader(f'File path: {file_abs_path}'),
        dbc.CardBody([
            # define_des_layout(file_info, df),
            dcc.Graph(id=f'graph-div-{idx}')
        ])
    ], className='card')


@callback(Output('output', 'children'),
          Input('var-openfast', 'data'),
          [[Input(f'df-{idx}', 'data') for idx in file_indices]])
def manage_cards(var_openfast, df_dict_list):
    # df_dict_list = [{file1: df1}, {file2: df2}, ...]

    children = []
    for i, (idx, file_path) in enumerate(var_openfast['file_path'].items()):            # idx = file1, file2, ... where {'file1': 'of-output/NREL5MW_OC3_spar_0.out', 'file2': 'of-output/IEA15_0.out'}
        if file_path == 'None':
            continue
        df_idx = [d.get(idx, None) for d in df_dict_list][i]
        children.append(make_card(idx, file_path, df_idx))      # Pass: file1, file1.out, df1
    
    return children



###############################################
#   Save configurations with button
###############################################

@callback(Output('confirm-update-of', 'displayed'),
          Output('var-openfast-graph', 'data', allow_duplicate=True),
          Input('save-of', 'n_clicks'),
          State('var-openfast-graph', 'data'),
          State('input-dict', 'data'),
          State('signalx', 'value'),
          State('signaly', 'value'),
          prevent_initial_call=True)
def save_openfast(btn, of_options, input_dict, signalx, signaly):
    
    of_options['graph_x'] = signalx
    of_options['graph_y'] = signaly

    if "save-of" == ctx.triggered_id:
        print('save button with ', signalx, signaly)
        input_dict['userPreferences']['openfast']['graph']['xaxis'] = signalx
        input_dict['userPreferences']['openfast']['graph']['yaxis'] = signaly

        update_yaml(input_dict, input_dict['yamlPath'])
        
        return True, of_options

    return False, of_options
