'''This is the page for visualizing table and plots of OpenFAST output'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from Input and this updates the Output. State doesn't trigger the function.
'''

# Import Packages
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
import datetime
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from weis.visualization.utils import store_dataframes, get_file_info, update_yaml, remove_duplicated_legends

register_page(
    __name__,
    name='OpenFAST',
    top_nav=True,
    path='/open_fast'
)

##################################################
#   Read openfast related variables from yaml file
###############################################

@callback(Output('var-openfast', 'data'),
          Output('var-openfast-graph', 'data'),
          Output('var-openfast-dfs', 'data'),
          Input('input-dict', 'data'))
def read_default_variables(input_dict):
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    of_options = {}
    var_openfast = input_dict['userPreferences']['openfast']
    var_files = var_openfast['file_path']
    dfs = store_dataframes(var_files)               # {file_path1: df1, file_path2: df2, ...} where df is list

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
                            dbc.Button('Plot', id='save-of', n_clicks=0, color='primary'),
                            width='auto'
                        )
                ], className="mb-3")
    
    signalx_input = dbc.Row([
                        dbc.Label('Signal-X', width=2),
                        dbc.Col(
                            dcc.Dropdown(id='signalx', options=[], value=None),                     # options look like ['Azimuth', 'B1N1Alpha', ...]. select ['Wind1VelX', 'Wind1VelY', 'Wind1VelZ'] as default value
                            width=7
                        )
                    ], className="mb-3")

    plotoption_input = dbc.Row([
                            dbc.Label('Plot Option', width=2),
                            dbc.Col(
                                dbc.RadioItems(
                                    id='plotOption',
                                    options=[
                                        {'label': 'Full', 'value': 'full'},
                                        {'label': 'Individual', 'value': 'individual'},
                                    ],
                                    value = 'individual'
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
                # OpenFAST related Data fetched from input-dict
                dcc.Store(id='var-openfast', data={}),
                dcc.Store(id='var-openfast-graph', data={}),
                # Dataframe to share over functions - openfast .out file
                dcc.Store(id='var-openfast-dfs', data={}),
                dbc.Row([
                    dbc.Col(form_layout, width=4),              # Channels
                    dbc.Col(html.Div(id='output'), width=8)     # Append cards per file
                ], className='g-0')         # No gutters where horizontal spacing is added between the columns by default
            ]))
    
    return layout


#################################################
#   Update graph configuration layout - left col
#################################################

@callback(Output('signaly', 'options'),
          Output('signaly', 'value'),
          Output('signalx', 'options'),
          Output('signalx', 'value'),
          State('var-openfast-dfs', 'data'),
          Input('var-openfast-graph', 'data'))
def define_graph_cfg_layout(dfs, of_options):

    if dfs == {}:
        raise PreventUpdate
    
    channels = sorted(list(dfs.values())[0][0].keys())           # df_dict['file1'][0]: First row channels

    return channels, of_options['graph_y'], channels, of_options['graph_x']



############################################################################
#   Plot graphs based on configuration setting and update it on yaml file
############################################################################

@callback(Output('var-openfast-graph', 'data', allow_duplicate=True),       # Dump into file
          Input('save-of', 'n_clicks'),
          State('var-openfast-graph', 'data'),
          State('input-dict', 'data'),
          State('signalx', 'value'),
          State('signaly', 'value'),
          prevent_initial_call=True)
def save_openfast(btn, of_options, input_dict, signalx, signaly):

    if btn==0:
        raise PreventUpdate
    
    print('Plot graph with ', signalx, signaly)

    # Update signalx, signaly graph config settings into yaml file
    of_options['graph_x'] = signalx
    of_options['graph_y'] = signaly

    input_dict['userPreferences']['openfast']['graph']['xaxis'] = signalx
    input_dict['userPreferences']['openfast']['graph']['yaxis'] = signaly

    update_yaml(input_dict, input_dict['yamlPath'])
    
    return of_options


@callback(Output('output', 'children'),
          Input('save-of', 'n_clicks'),
          State('var-openfast-dfs', 'data'),
          State('signalx', 'value'),
          State('signaly', 'value'),
          State('plotOption', 'value'))
def update_graph_layout(btn, dfs, signalx, signaly, plotOption):

    if btn==0:
        raise PreventUpdate
    
    return manage_cards(dfs, signalx, signaly, plotOption)



###############################################
#   Update file description layout
###############################################

def define_des_layout(file_info):
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

def draw_graph(file_path_list, df_dict_list, signalx, signaly, plotOption):     # Standards for params are from 'full' plotOption
    # Whenever signalx, signaly, plotOption has been entered, draw the graph.
    # Create figure with that setting and add that figure to the graph layout.
    # Note that we set default settings (see analyze() function), it will show corresponding default graph.
    # You can dynamically change signalx, signaly, plotOption, and it will automatically update the graph.

    # Put all traces in one single plot
    if plotOption == 'full':

        cols = plotly.colors.DEFAULT_PLOTLY_COLORS
        fig = make_subplots(rows = len(signaly), cols = 1, shared_xaxes=True, vertical_spacing=0.05)
        
        for idx, df_dict in enumerate(df_dict_list):
            df = pd.DataFrame(df_dict)
            for row_idx, label in enumerate(signaly):
                fig.append_trace(go.Scatter(
                    x = df[signalx],
                    y = df[label],
                    mode = 'lines',
                    line=dict(color=cols[idx]),
                    name = file_path_list[idx]),
                    row = row_idx + 1,
                    col = 1)
                fig.update_yaxes(title_text=label, row=row_idx+1, col=1)

        # Remove duplicated legends
        remove_duplicated_legends(fig)

        fig.update_layout(height=150 * len(signaly), legend=dict(orientation='h', yanchor='bottom', xanchor='right', x=1, y=1.02, itemsizing='constant'))
        fig.update_xaxes(title_text=signalx, row=len(signaly), col=1)
        

    # Put each traces in each separated vertically aligned subplots
    elif plotOption == 'individual':
        df = pd.DataFrame(df_dict_list)
        fig = make_subplots(rows = len(signaly), cols = 1, shared_xaxes=True, vertical_spacing=0.05)

        for row_idx, label in enumerate(signaly):
            fig.append_trace(go.Scatter(
                x = df[signalx],
                y = df[label],
                mode = 'lines',
                showlegend=False),
                row = row_idx + 1,
                col = 1)
            fig.update_yaxes(title_text=label, row=row_idx+1, col=1)
    
        fig.update_layout(height=150 * len(signaly))
        fig.update_xaxes(title_text=signalx, row=len(signaly), col=1)

    return fig


###############################################
#   Dynamic card creation
###############################################

def make_card(file_path, df, signalx, signaly, plotOption):

    return dbc.Card([
        dbc.CardHeader(f'Processing {len(file_path)} Files:' if isinstance(file_path, list) else f'File: {file_path}'),
        dbc.CardBody([
            dcc.Graph(figure=draw_graph(file_path, df, signalx, signaly, plotOption))
        ])
    ], className='card')


def manage_cards(dfs, signalx, signaly, plotOption):

    children = []

    if plotOption == 'full':
        # For full view, add all of the tracks in a single plot
        children.append(make_card(list(dfs.keys()), list(dfs.values()), signalx, signaly, plotOption))

    elif plotOption == 'individual':
        # For individual view, make multiple subplots and add the track individually
        for file_path, df in dfs.items():
            if file_path == 'None':
                continue

            children.append(make_card(file_path, df, signalx, signaly, plotOption))      # Pass: file1, file1.out, df1
    
    return children
