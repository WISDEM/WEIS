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
    layout = dcc.Loading(html.Div([
                # Confirm Dialog to check updated
                dcc.ConfirmDialog(
                    id='confirm-update-of',
                    message='Updated'
                ),
                dbc.Card([
                    dbc.CardBody([
                        dbc.InputGroup(
                            [
                                # Layout for showing graph configuration setting
                                html.Div(id='graph-cfg-div', className='text-center'),
                                dbc.Button('Save', id='save-of', n_clicks=0, style={'float': 'right'})
                            ]
                        )
                    ])
                ]),
                # Append cards per file
                dbc.Row([], id='output')
            ]))
    
    return layout


###############################################
#   Update graph configuration layout - first row
###############################################

@callback(Output('graph-cfg-div', 'children'),
          Input('df-file1', 'data'),
          Input('var-openfast-graph', 'data'))
def define_graph_cfg_layout(df1, of_options):

    if df1 is None or df1 == {}:
        raise PreventUpdate
    
    channels = sorted(df1['file1'][0].keys())
    # print(df_dict['file1'][0])          # First row channels

    return html.Div([
                html.Div([
                    html.Label(['Signal-y:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.Dropdown(id='signaly', options=channels, value=of_options['graph_y'], multi=True),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                ], style = {'float':'left', 'padding-left': '1.0rem'}),
                html.Div([
                    html.Label(['Signal-x:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.Dropdown(id='signalx', options=channels, value=of_options['graph_x']),          # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                ], style = {'float':'left', 'width': '200px', 'padding-left': '1.0rem'}),
                html.Div([
                    html.Label(['Plot options:'], style={'font-weight':'bold', 'text-align':'center'}),
                    dcc.RadioItems(id='plotOption', options=['single plot', 'multiple plot'], value='single plot', inline=True),
                ], style = {'float':'left', 'padding-left': '1.0rem', 'padding-right': '1.0rem'})
            ])


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
                    html.H5(f'File Path: {file_abs_path}'),
                    html.H5(f'File Size: {file_size} MB'),
                    html.H5(f'Creation Date: {datetime.datetime.fromtimestamp(creation_time)}'),
                    html.H5(f'Modification Date: {datetime.datetime.fromtimestamp(modification_time)}'),
                    html.Br(),

                    # Data Table
                    # dash_table.DataTable(
                    #     data=df,
                    #     columns=[{'name': i, 'id': i} for i in pd.DataFrame(df).columns],
                    #     fixed_columns = {'headers': True, 'data': 1},
                    #     page_size=10,
                    #     style_table={'height': '300px', 'overflowX': 'auto', 'overflowY': 'auto'})
            ])


###############################################
#   Update graph layout per card
###############################################

def update_figure(signalx, signaly, plotOption, df_dict):
    df, = df_dict.values()
    return draw_graph(signalx, signaly, plotOption, pd.DataFrame(df))


for idx in file_indices:
    callback(Output(f'graph-div-{idx}', 'figure'),
                Input('signalx', 'value'),
                Input('signaly', 'value'),
                Input('plotOption', 'value'),
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
        

    # Put each traces in each separated horizontally aligned subplots
    elif plotOption == 'multiple plot':
        fig = make_subplots(rows = 1, cols = len(signaly))

        for col_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                name = label),
                row = 1,
                col = col_idx + 1)
            fig.update_yaxes(title_text=label, row=1, col=col_idx+1)
    
    # Define the graph layout where it includes the rendered figure
    fig.update_xaxes(title_text=signalx)

    return fig


###############################################
#   Dynamic card creation
###############################################

def make_card(idx, file_path, df):
    file_info = get_file_info(file_path)
    file_name = file_info['file_name']

    return dbc.Card([
        dbc.CardHeader(f'File name: {file_name}', className='cardHeader'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Loading(define_des_layout(file_info, df)), width=3),
                dbc.Col(dcc.Loading(dcc.Graph(id=f'graph-div-{idx}')), width=9)
            ])
        ])
    ])


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
          State('var-openfast-graph', 'data'),
          Input('save-of', 'n_clicks'),
          Input('input-dict', 'data'),
          Input('signalx', 'value'),
          Input('signaly', 'value'),
          prevent_initial_call=True)
def save_openfast(of_options, btn, input_dict, signalx, signaly):
    
    of_options['graph_x'] = signalx
    of_options['graph_y'] = signaly

    if "save-of" == ctx.triggered_id:
        print('save button with ', signalx, signaly)
        input_dict['userPreferences']['openfast']['graph']['xaxis'] = signalx
        input_dict['userPreferences']['openfast']['graph']['yaxis'] = signaly

        update_yaml(input_dict, input_dict['yamlPath'])
        
        return True, of_options

    return False, of_options
