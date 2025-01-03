from dash import html, register_page
from dash import dcc, Input, State, Output, callback, dash_table, no_update, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from pathlib import Path
import pandas as pd
from weis.visualization.utils import parse_yaml, dict_to_html, load_geometry_data, find_rows

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():
    #######################################
    # Layout for form - WEIS VIZ Config
    #######################################
    file_input_cfg = dbc.Row([
                    dbc.Label('File', width=2),
                    dbc.Col(
                        dbc.Input(
                            id='vizInput_path', type='text', placeholder='Enter the absolute path for file to import'
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dbc.Button('Reload', id='reload', n_clicks=0),
                        width='auto'
                    )
                ], className="mb-3")

    cfg_form_layout = dbc.Card([
                            dbc.CardHeader("Import WEIS Visualization Configuration File"),
                            dbc.CardBody([
                                dbc.Form([file_input_cfg])
                            ])
                  ], className='card')
    

    #######################################
    # Layout for form - WEIS Input
    #######################################
    file_input = dbc.Row([
                    dbc.Label('File', width=2),
                    dbc.Col(
                        dbc.Input(
                            id='file-path', type='text', placeholder='Enter the absolute path for file to import'
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dbc.Button('Add', id='import-btn', n_clicks=0, color='primary'),
                        width='auto'
                    )
                ], className="mb-3")
    
    label_input = dbc.Row([
                        dbc.Label('Label', width=2),
                        dbc.Col(
                            dbc.Input(
                                id='file-label', type='text', placeholder='Enter label'
                            ),
                            width=6
                        )
                    ], className="mb-3")

    type_input = dbc.Row([
                            dbc.Label('Type', width=2),
                            dbc.Col(
                                dbc.RadioItems(
                                    id='file-type',
                                    options=[
                                        {'label': 'Model', 'value': 'model'},
                                        {'label': 'Analysis', 'value': 'analysis'},
                                        {'label': 'Geometry', 'value': 'geometry'}
                                    ]
                                ),
                                width='auto'
                            )
                        ], className="mb-3")
    
    warning_msg = dbc.Toast(
                        html.P(id='form-warning-text'),
                        id="form-warning",
                        header="Warning",
                        is_open=False,
                        dismissable=True,
                        icon="danger",
                        style={"position": "fixed", "top": 66, "right": 10, "width": 350}
                    )

    form_layout = dbc.Card([
                            dbc.CardHeader("Import WEIS Input Files"),
                            dbc.CardBody([
                                dbc.Form([file_input, label_input, type_input]),
                                warning_msg
                            ])
                  ], className='card')


    #######################################
    # Layout for Results
    #######################################
    file_table = dash_table.DataTable(
                    id = 'file-table-interactivity',
                    columns = [],
                    data = [],
                    style_header={
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'font-family': 'sans-serif'
                    },
                    editable = True,
                    sort_action = "native",
                    sort_mode = "multi",
                    # row_selectable = "multi",
                    row_deletable = True,
                    # selected_rows = [],
                    persistence=True,
                    persistence_type='session',
                )
    
    text_output = dbc.Row([
                    dbc.Label('WEIS VIZ Config File', width=2),
                    dbc.Col(
                        html.Div(id='input-cfg-div'),
                        width='auto'
                    ),
                ], className="mb-3")
    
    result_layout = dbc.Card([
                        dbc.CardHeader('Result'),
                        dbc.CardBody([
                            text_output,
                            file_table
                        ])
                  ], className='card')
    

    #######################################
    # Main Layout - Overview
    #######################################
    layout = dcc.Loading(html.Div([
                cfg_form_layout,
                form_layout,
                result_layout
            ]))

    return layout



#######################################
# Functions related to WEIS Viz Config
#######################################
@callback(Output('input-cfg-div', 'children'),
          Input('input-dict', 'data'))
def check_input_file(contents):
    '''
    Store data in mainApp.py so that it's accessible over pages.
    Show if input file data has been loaded and parsed successfully
    '''

    # For File Tree (previous version)
    if contents is None:
        raise PreventUpdate

    # Showing File Tree (previous version)
    # file_tree_list = dict_to_html(contents, [], level=1)
    # return html.Div([*file_tree_list], style={'width':'80vw', 'marginLeft': 100, 'border-left-style':'dotted'})

    return html.Div([contents['yamlPath']])


@callback(Output('input-dict', 'data'),
          State('input-dict', 'data'),
          State('vizInput_path', 'value'),
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



#######################################
# Functions related to WEIS Input
#######################################
def handle_file_error(filepath, filelabel, filetype):
    if (filepath in [None, '']) or (filelabel in [None, '']) or (filetype in [None, '']):
        return True

    if not filepath.endswith('.yaml'):
        return True
    
    if not Path(filepath).is_file():
        return True
    
    return False


def add_row(filepath, filelabel, filetype, df_dict):
    # Error handling
    if handle_file_error(filepath, filelabel, filetype):
        return True, "Please check file again", no_update
    
    # Check if row already exists
    df = pd.DataFrame(df_dict)
    if ((df['File Path'] == filepath) | (df['Label'] == filelabel)).any():
        return True, "Already entered", df_dict
    
    # Add row to dataframes
    df_dict['File Path'] += [filepath]
    df_dict['Label'] += [filelabel]
    df_dict['Type'] += [filetype]

    return False, "", df_dict


def del_row(curr_data, prev_data):

    return True, "File has been deleted", curr_data


@callback(Output('form-warning', 'is_open'),
          Output('form-warning-text', 'children'),
          Output('file-df', 'data'),
          # Either triggered by form or table
          Input('import-btn', 'n_clicks'),
          Input('file-table-interactivity', 'data_timestamp'),
          # Form
          State('file-path', 'value'),
          State('file-label', 'value'),
          State('file-type', 'value'),
          State('file-df', 'data'),
          # Table
          State('file-table-interactivity', 'data'),
          State('file-table-interactivity', 'data_previous'))
def edit_table(nclickForm, ts, filepath, filelabel, filetype, df_dict, curr_data, prev_data):
    '''
    Update data frame to be shown in table once the form input button has been clicked
    form => table (add new row) => editable table where you can delete => update dataframe based on table
    '''
    if nclickForm not in [None, 0] and ctx.triggered_id == 'import-btn':
        warning_flag, warning_msg, updated_df_dict = add_row(filepath, filelabel, filetype, df_dict)
        return warning_flag, warning_msg, updated_df_dict
    
    elif ctx.triggered_id == 'file-table-interactivity':
        del_flag, del_msg, updated_df_dict = del_row(curr_data, prev_data)
        return del_flag, del_msg, updated_df_dict

    else:
        return False, "", df_dict
    

@callback(Output('file-table-interactivity', 'columns'),
          Output('file-table-interactivity', 'data'),
          Input('file-df', 'data'))
def update_table(df_dict):
    '''
    Create/update the (editable) table based on data frame
    '''

    df = pd.DataFrame(df_dict)
    print('\nUpdate Table Data\n', df)

    return [{'name': i, 'id': i, 'deletable': True, 'selectable': True} for i in df.columns], df.to_dict('records')



@callback(Output('airfoil-by-names', 'data'),
          Output('geometry-components', 'data'),
          Output('wt-options', 'data'),
          Input('file-df', 'data'))
def categorize_data(df_dict):
    '''
    This function is for loading and processing airfoils data
    Note that even if you don't navigate to this page, variables are all defined from main app page so this callback function will be auto-initiated.
    '''
    # airfoils, geom_comps = load_geometry_data(df['geometry'])
    try:
        airfoils, geom_comps, wt_options_by_files = load_geometry_data(find_rows(df_dict, 'geometry'))      # Data structure: {file1: [{'name': 'FB90', 'coordinates': {'x': [1.0, 0.9921, ...]}}], file2: ~~~}
        airfoil_by_names = {label+': '+airfoil['name']: dict(list(airfoil.items())[1:]) for label, airfoils_per_file in airfoils.items() for airfoil in airfoils_per_file}      # {'file1: FB90': {'coordinates': {'x': [1.0, 0.9921, 0.5], 'y': [1.0, 2.0, 3.0]}}, ... }
        geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}
    except Exception as e:
        print(e)

    return airfoil_by_names, geom_comps_by_names, wt_options_by_files