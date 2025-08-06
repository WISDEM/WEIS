from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import base64
import yaml
import io
import pandas as pd
import plotly.express as px

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

###################################
## Define the layout of the app
###################################
csv_file_input = dbc.Row([
                    dbc.Label('CSV File', width=1),
                    dbc.Col(
                        dbc.Input(
                            id='csv-file-path', type='text', placeholder='Enter the absolute path for file to import'
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Upload(
                            id='csv-upload-data',
                            children=dbc.Button('Browse', n_clicks=0, color='secondary', className='me-1'),
                            multiple=False
                        ),
                        width='auto'
                    ),
                    dbc.Col(
                        dbc.Button('Load', id='csv-load-btn', n_clicks=0, color='primary'),
                        width='auto'
                    ),
                    dbc.Tooltip(
                        'Load the CSV file from the given path and enter this button',
                        target='csv-load-btn',
                    ),
                ], className="mb-3")

yaml_file_input = dbc.Row([
                    dbc.Label('YAML File', width=1),
                    dbc.Col(
                        dbc.Input(
                            id='yaml-file-path', type='text', placeholder='Enter the absolute path for file to import'
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Upload(
                            id='yaml-upload-data',
                            children=dbc.Button('Browse', n_clicks=0, color='secondary', className='me-1'),
                            multiple=False
                        ),
                        width='auto'
                    ),
                    dbc.Col(
                        dbc.Button('Load', id='yaml-load-btn', n_clicks=0, color='primary'),
                        width='auto'
                    ),
                    dbc.Tooltip(
                        'Load the YAML file from the given path and enter this button',
                        target='yaml-load-btn',
                    ),
                ], className="mb-3")

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(csv_file_input, width=12),
            dbc.Col(yaml_file_input, width=12)
        ], className="mt-4"),  # Added margin at the top
        dbc.Row([
            dbc.Col(dcc.Graph(id='splom'), width=12)
        ], className="mt-3")
    ]),
    dcc.Store(id='csv-df'),
    dcc.Store(id='yaml-df')
])


#######################################################
## Load Data - CSV File and Yaml File
## We give two options to load data
## 1. Enter the absolute path of the file
## 2. Browse the file from the local directory
#######################################################

# Callback to update CSV file path when file is browsed
@callback(Output('csv-file-path', 'value'),
          Input('csv-upload-data', 'filename'),
          prevent_initial_call=True)
def update_csv_file_path(filename):
    if filename:
        return filename
    return ""

# Callback to update YAML file path when file is browsed
@callback(Output('yaml-file-path', 'value'),
          Input('yaml-upload-data', 'filename'),
          prevent_initial_call=True)
def update_yaml_file_path(filename):
    if filename:
        return filename
    return ""

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return df.to_json(date_format='iso', orient='split')    # Need to convert DataFrame to JSON for dcc.Store
        
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df.to_json(date_format='iso', orient='split')    # Need to convert DataFrame to JSON for dcc.Store
        
        elif 'yaml' in filename or 'yml' in filename:
            # Assume that the user uploaded a YAML file
            return yaml.safe_load(io.StringIO(decoded.decode('utf-8')))

    except Exception as e:
        print(e)
        return None


def extract_variable_names(var_list):
    """Extract variable names from nested YAML structure."""
    names = []
    for item in var_list:
        if isinstance(item, list):
            if isinstance(item[0], list):
                names.append(item[0][0])  # Extract the first element of the first list
            else:
                names.append(item[0])
        else:
            names.append(item)
    return names

@callback(Output('csv-df', 'data'),
          [Input('csv-upload-data', 'contents'),
           Input('csv-load-btn', 'n_clicks')],
          [State('csv-upload-data', 'filename'),
           State('csv-upload-data', 'last_modified'),
           State('csv-file-path', 'value')],
          prevent_initial_call=True)
def load_csv_data(contents, load_clicks, filename, date, file_path):
    
    # Determine which input triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'csv-upload-data' and contents is not None:
        # File was uploaded via browse button
        return parse_contents(contents, filename, date)
    elif trigger_id == 'csv-load-btn' and load_clicks and file_path:
        # File path was entered and load button clicked
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                print("Unsupported file format")
                return None
            
            print(df.head())
            return df.to_json(date_format='iso', orient='split')
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    return None


@callback(Output('yaml-df', 'data'),
          [Input('yaml-upload-data', 'contents'),
           Input('yaml-load-btn', 'n_clicks')],
          [State('yaml-upload-data', 'filename'),
           State('yaml-upload-data', 'last_modified'),
           State('yaml-file-path', 'value')],
          prevent_initial_call=True)
def load_yaml_data(contents, load_clicks, filename, date, file_path):
    # Determine which input triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'yaml-upload-data' and contents is not None:
        # File was uploaded via browse button
        config = parse_contents(contents, filename, date)
    
    elif trigger_id == 'yaml-load-btn' and load_clicks and file_path:
        # File path was entered and load button clicked
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            config = yaml.safe_load(file_path)
    
    objectives = extract_variable_names(config.get('objectives', []))
    constraints = extract_variable_names(config.get('constraints', []))
    design_vars = extract_variable_names(config.get('design_vars', []))
    print(f"Objectives: {objectives}")
    print(f"Constraints: {constraints}")
    print(f"Design Variables: {design_vars}")
    
    return {
        'objectives': objectives,
        'constraints': constraints,
        'design_vars': design_vars
    }


# @callback(Output('output-data-upload', 'children'),
#           Input('csv-df', 'data'))
# def update_output(stored_data):
#     if stored_data is not None:
#         df = pd.read_json(io.StringIO(stored_data), orient='split')
#         return html.Div([
#             dash_table.DataTable(
#                 df.to_dict('records'),
#                 [{'name': i, 'id': i} for i in df.columns]
#             ),
#         ])

@callback(Output('splom', 'figure'),
          Input('csv-df', 'data'),
          Input('yaml-df', 'data'))
def update_splom(csv_data, yaml_data):
    if csv_data is None or yaml_data is None:
        # Return empty figure with a message
        return {
            'data': [],
            'layout': {
                'title': 'Load CSV data to view scatter plot matrix',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'No data available',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            }
        }

    # Convert JSON back to DataFrame
    df = pd.read_json(io.StringIO(csv_data), orient='split')
    
    print(yaml_data)
    # Create scatter plot matrix
    splom_fig = px.scatter_matrix(df)
    splom_fig.update_layout(title='Scatter Plot Matrix')
    
    return splom_fig


if __name__ == '__main__':
    # Host, port is required for successful ssh connection
    app.run(debug=True, host='0.0.0.0', port=8050)