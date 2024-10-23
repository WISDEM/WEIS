import pandas as pd
from dash import html, register_page
from dash import dcc, Input, State, Output, callback, ALL, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from weis.visualization.utils import parse_yaml, dict_to_html, load_geometry_data

register_page(
    __name__,
    name='Import',
    top_nav=True,
    path='/import'
)

def layout():

    # Layout for form
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

    # Layout for result table
    table_layout = dbc.Card([
                            dbc.CardBody([
                                html.Div(id='file-table')
                            ])
                  ], className='card')

    layout = html.Div([
                form_layout,
                table_layout
            ])
    

    return layout


@callback(Output('form-warning', 'is_open'),
          Output('form-warning-text', 'children'),
          Output('file-df', 'data'),
          Input('import-btn', 'n_clicks'),
          State('file-path', 'value'),
          State('file-label', 'value'),
          State('file-type', 'value'),
          State('file-df', 'data'))
def add_row(nclick, filepath, filelabel, filetype, df_dict):
    '''
    Update data frame to be shown in table once the form input button has been clicked
    '''
    if nclick is None or nclick==0:
        raise PreventUpdate
    
    elif (filepath in [None, '']) or (filelabel in [None, '']) or (filetype in [None, '']):
        return True, "Please enter all of the forms", df_dict

    if not filepath.endswith('.yaml'):
        return True, "Please enter yaml file", df_dict
    
    # Check if row already exists
    df = pd.DataFrame(df_dict)
    if ((df['File Path'] == filepath) & (df['Label'] == filelabel) & (df['Type'] == filetype)).any():
        return True, "Already entered", df_dict
    
    df_dict['File Path'] += [filepath]
    df_dict['Label'] += [filelabel]
    df_dict['Type'] += [filetype]

    return False, "", df_dict


@callback(Output('file-table', 'children'),
          Input('file-df', 'data'))
def update_table(df_dict):
    '''
    Create/update the (editable) table based on data frame
    '''

    df = pd.DataFrame(df_dict)
    print('\nUpdate Table Data\n', df)

    return dash_table.DataTable(
        id = 'file-table-interactivity',
        columns = [{'name': i, 'id': i, 'deletable': True, 'selectable': True} for i in df.columns],
        data = df.to_dict('records'),
        editable = True,
        sort_action = "native",
        sort_mode = "multi",
        row_selectable = "multi",
        row_deletable = True,
        selected_rows = [],
        # persistence=True,
        # persistence_type='session',
    )

# @callback(Output('file-table-interactivity', 'data'),
#           Output('file-table-interactivity', 'columns'),
#           Input('file-df', 'data'))
# def update_table_data(df_dict):
#     df = pd.DataFrame(df_dict)
#     print('\nUpdate Table Data\n', df)

#     data = df.to_dict('records')
#     cols = [{'name': i, 'id': i, 'deletable': True, 'selectable': True} for i in df.columns]

#     return data, cols


@callback(Output('file-table-interactivity', 'style_data_conditional'),
          Input('file-table-interactivity', 'derived_virtual_selected_rows'))
def update_styles(selected_rows_indices):
    '''
    Change the row color once clicked
    '''

    if selected_rows_indices is None:
        selected_rows_indices = []
    
    return [{
        'if': { 'row_index': i },
        'background_color': '#D2F3FF'
    } for i in selected_rows_indices]


@callback(Output('selected-file-df', 'data'),
          State('file-table-interactivity', 'derived_virtual_data'),
          Input('file-table-interactivity', 'derived_virtual_selected_rows'))
def save_selections(rows, selected_rows_indices):
    '''
    Save selections into another small version of dataframe, where this will be ultimately used over other pages
    '''
    if selected_rows_indices is None:
        selected_rows_indices = []
    
    selected_rows = [rows[i] for i in selected_rows_indices]

    df = {'model': [], 'analysis': [], 'geometry': []}
    for row in selected_rows:
        row_type = row['Type']
        row.popitem()               # Remove duplicated 'Type' field
        df[row_type].append(row)

    print('selected df\n', df)
    
    return df


@callback(Output('airfoil-by-names', 'data'),
          Output('geometry-components', 'data'),
          Input('selected-file-df', 'data'))
def categorize_airfoils(df):
    '''
    This function is for loading and processing airfoils data
    Note that even if you don't navigate to this page, variables are all defined from main app page so this callback function will be auto-initiated.
    '''
    airfoils, geom_comps = load_geometry_data(df['geometry'])      # Data structure: {file1: [{'name': 'FB90', 'coordinates': {'x': [1.0, 0.9921, ...]}}], file2: ~~~}
    airfoil_by_names = {label+': '+airfoil['name']: dict(list(airfoil.items())[1:]) for label, airfoils_per_file in airfoils.items() for airfoil in airfoils_per_file}      # {'file1: FB90': {'coordinates': {'x': [1.0, 0.9921, 0.5], 'y': [1.0, 2.0, 3.0]}}, ... }
    geom_comps_by_names = {label+': '+comp_type: comp_info for label, geom_comps_per_file in geom_comps.items() for comp_type, comp_info in geom_comps_per_file.items()}

    return airfoil_by_names, geom_comps_by_names


'''
# Use dbc.DataTable -- issue occurs on duplicated outputs, which doesn't allow initial callback. Need more than twice clicks to be actioned.
@callback(Output('file-table', 'children'),
          Output('form-warning', 'is_open'),
          Output('form-warning-text', 'children'),
          Output('file-df', 'data'),
          Input('import-btn', 'n_clicks'),
          State('file-path', 'value'),
          State('Label', 'value'),
          State('file-type', 'value'),
          State('file-df', 'data'))
def add_row(nclick, filepath, label, filetype, df):

    if nclick is None or nclick==0:
        raise PreventUpdate
    
    elif (filepath in [None, '']) or (label in [None, '']) or (filetype in [None, '']):
        return dbc.Table.from_dataframe(pd.DataFrame(df)), True, "Please enter all of the forms", df

    if not filepath.endswith('.yaml'):
        return dbc.Table.from_dataframe(pd.DataFrame(df)), True, "Please enter yaml file", df

    # Check if row already exists
    if ((pd.DataFrame(df)['File Path'] == filepath) & (pd.DataFrame(df)['Label'] == label) & (pd.DataFrame(df)['Type'] == filetype)).any():
        return dbc.Table.from_dataframe(pd.DataFrame(df)), True, "Already entered", df
    

    print('Adding new row:\n', filepath, label, filetype)
    df['File Path'] += [filepath]
    df['Label'] += [label]
    df['Type'] += [filetype]
    print('df\n', df)

    # table_header = [
    #     html.Thead(html.Tr([html.Th('File Path'), html.Th('Label'), html.Th('Type')]))
    # ]

    # table_body = [
    #     html.Tbody([html.Td(0), html.Td(0), html.Td(0), dbc.Button('Select', color='success', id={'type': 'add'}, n_clicks=0)])
    # ]

    # table = dbc.Table(table_header + table_body)

    # df = {
    #         "Service": ['1','2','3','4'],
    #         "Status": ['1','2','3','4'],
    #         "Select": [dbc.Button('Select', color='success', id={'type': 'add'}, n_clicks=0), dbc.Button('Select', color='success', id={'type': 'add'}, n_clicks=0), dbc.Button('Select', color='success', id={'type': 'add'}, n_clicks=0), dbc.Button('Select', color='success', id={'type': 'add'}, n_clicks=0)]
    #     }
    
    # Update file table - adding new row with 'select/delete' button
    table = dbc.Table.from_dataframe(pd.DataFrame(df))
    tbody = table.children[1].children
    # tbody = [html.Tr(row.children + [dbc.Button('Select', id=f'select-btn-{idx}', n_clicks=0, color='success', className="me-2"), dbc.Button('Delete', id=f'delete-btn-{idx}', n_clicks=0, color='danger', outline=False)]) for idx, row in enumerate(tbody)]
    tbody = [html.Tr(row.children + [dbc.Button('Select', id={'type': 'add', 'index': idx}, n_clicks=0, color='success', className="me-2"), dbc.Button('Delete', id={'type': 'remove', 'index': idx}, n_clicks=0, color='danger', outline=False)]) for idx, row in enumerate(tbody)]
    table.children[1].children = tbody

    return table, False, "", df


@callback(Output('file-table', 'children'),
          Input({'type': 'add', 'index': ALL}, 'n_clicks'))
def select_row(nclick):
    if nclick is None or nclick==0:
        raise PreventUpdate
    
    print('nclick', nclick)
'''
