from dash import Dash, dcc, html, Input, Output, State, callback, callback_context, ALL
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

cfg_graph_input = dbc.Row([
                    dbc.Col([
                        dbc.Label("Channels to display:", className="fw-bold mb-2"),
                        html.Div(id='channels', children=[
                            dbc.Alert("Load YAML file to see variable options", color="info", className="text-center")
                        ]),
                    ]),
                ], className="mb-3")

app.layout = html.Div([
    dbc.Container([
        # File Loaders
        dbc.Row([
            dbc.Col(csv_file_input, width=12),
            dbc.Col(yaml_file_input, width=12)
        ], className="mt-4"),  # Added margin at the top

        # Plot Renderers
        dbc.Row([
            dbc.Col(cfg_graph_input, width=4),
            dbc.Col(dcc.Graph(id='splom'), width=8)
        ], className="mt-4"),  # Added margin at the top
    ]),
    # Store data to use across callbacks
    dcc.Store(id='csv-df'),
    dcc.Store(id='yaml-df'),
    dcc.Store(id='selected-channels', data=[]),  # Store list of selected channels
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
        try:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                print("Unsupported file format")
                return None
        except Exception as e:
            print(f"Error reading YAML file: {e}")
            return None
    else:
        return None
    
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

#######################################################
## Configuration Panel for Dynamic SPLOM Generation
## We give users which channels to display in the SPLOM
#######################################################

# Callback to create dynamic button groups for channels selection
@callback(Output('channels', 'children'),  # Remove the list brackets
          [Input('yaml-df', 'data'),
           Input('selected-channels', 'data')])
def update_channel_buttons(yaml_data, selected_channels):
    if yaml_data is None:
        no_data_msg = dbc.Alert("Load YAML file to see variable options", color="info", className="text-center")
        return [no_data_msg]  # Return as a list since we want multiple children
    
    # Ensure selected variables are lists
    selected_channels = selected_channels or []

    # Extract all variables
    objectives = yaml_data.get('objectives', [])
    constraints = yaml_data.get('constraints', [])
    design_vars = yaml_data.get('design_vars', [])
    
    all_variables = objectives + constraints + design_vars
    
    if not all_variables:
        no_vars_msg = dbc.Alert("No variables found in YAML file", color="warning", className="text-center")
        return [no_vars_msg]  # Return as a list
    
    # Create button groups for each category
    def create_button_group(variables, category_name, color, selected_vars):
        if not variables:
            return []
        
        buttons = []
        for var in variables:
            # Determine if this button should be active (filled) or outline
            is_selected = (var in selected_vars)
            buttons.append(
                dbc.Button(
                    var,
                    id={'type': 'channel-btn', 'index': var},  # Changed to 'channel-btn'
                    color=color,
                    outline=not is_selected,  # Filled if selected, outline if not
                    size='sm',
                    className='me-1 mb-1'
                )
            )
        
        return [
            html.Div([
                html.Small(category_name, className="text-muted fw-bold"),
                html.Div(buttons, className="d-flex flex-wrap")
            ], className="mb-2")
        ]

    # Create channel buttons
    buttons = []
    if objectives:
        buttons.extend(create_button_group(objectives, "Objectives", "primary", selected_channels))
    if constraints:
        buttons.extend(create_button_group(constraints, "Constraints", "warning", selected_channels))
    if design_vars:
        buttons.extend(create_button_group(design_vars, "Design Variables", "success", selected_channels))

    return buttons  # Return the list directly

# Callback to handle channel button clicks (multi-select)
@callback(Output('selected-channels', 'data'),
          [Input({'type': 'channel-btn', 'index': ALL}, 'n_clicks')],  # Changed to match new button type
          State('selected-channels', 'data'),
          prevent_initial_call=True)
def handle_channels_selection(n_clicks_list, current_selected):
    if not n_clicks_list or not any(n_clicks_list):
        return current_selected or []
    
    # Find which button was clicked
    ctx = callback_context
    if ctx.triggered:
        try:
            trigger_info = ctx.triggered[0]['prop_id']
            button_id_str = trigger_info.split('.n_clicks')[0]
            
            import ast
            button_info = ast.literal_eval(button_id_str)
            clicked_var = button_info['index']
            
            # Toggle selection: add if not present, remove if present
            current_selected = current_selected or []
            if clicked_var in current_selected:
                current_selected.remove(clicked_var)
            else:
                current_selected.append(clicked_var)
            
            return current_selected
        except Exception as e:
            print(f"Error parsing button ID: {e}")
            # Fallback method
            if '"index":"' in trigger_info:
                start = trigger_info.find('"index":"') + 9
                end = trigger_info.find('"', start)
                if end > start:
                    clicked_var = trigger_info[start:end]
                    current_selected = current_selected or []
                    if clicked_var in current_selected:
                        current_selected.remove(clicked_var)
                    else:
                        current_selected.append(clicked_var)
                    return current_selected
    
    return current_selected or []


@callback(Output('splom', 'figure'),
          [Input('csv-df', 'data'),
           Input('selected-channels', 'data')])
def update_splom(csv_data, selected_channels):
    if csv_data is None:
        # Return empty figure with a message
        return {
            'data': [],
            'layout': {
                'title': 'Load CSV data to view plots',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'No CSV data available',
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
    
    # Ensure selected variables are lists
    selected_channels = selected_channels or []

    # Combine all selected variables
    all_selected_vars = list(set(selected_channels))

    # Filter variables that exist in the DataFrame
    available_vars = [var for var in all_selected_vars if var in df.columns]
    
    if len(available_vars) == 0:
        return {
            'data': [],
            'layout': {
                'title': 'Select variables to display SPLOM',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'Click variable buttons to select channels for SPLOM',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'blue'}
                }]
            }
        }
    
    # Create a mapping of simplified names for display
    simplified_names = {}
    simplified_df = df[available_vars].copy()
    
    # Add row index for linking across subplots with color
    simplified_df['sample_id'] = range(len(simplified_df))
    
    for var in available_vars:
        # Split by '.' and take the last element
        simplified_name = var.split('.')[-1]
        simplified_names[var] = simplified_name
        # Rename column in dataframe
        simplified_df = simplified_df.rename(columns={var: simplified_name})
    
    # Get the simplified column names (excluding sample_id)
    simplified_vars = [simplified_names[var] for var in available_vars]
    
    # Create scatter plot matrix with color-coded samples for linking
    splom_fig = px.scatter_matrix(
        simplified_df,
        dimensions=simplified_vars,
        color='sample_id',
        hover_data=['sample_id'],
        color_continuous_scale='viridis',
        title=f'Scatter Plot Matrix ({len(available_vars)} variables)'
    )
    
    splom_fig.update_layout(
        width=800, 
        height=800,
        title={
            'text': f'Scatter Plot Matrix ({len(available_vars)} variables)',
            'x': 0.5,
            'xanchor': 'center'
        },
        # Improve hover behavior for linked data
        hovermode='closest',
        # Add colorbar title
        coloraxis_colorbar=dict(
            title="Sample ID",
            title_side="right"
        )
    )
    splom_fig.update_traces(
        diagonal_visible=True, 
        showlowerhalf=False, 
        showupperhalf=True,
        # Customize hover template to show sample ID
        hovertemplate='<b>Sample %{customdata[0]}</b><br>' +
                      '%{xaxis.title.text}: %{x}<br>' +
                      '%{yaxis.title.text}: %{y}<br>' +
                      '<extra></extra>',
        # Increase marker size for better visibility
        marker=dict(size=4, line=dict(width=0.5, color='white'))
    )
    
    return splom_fig


if __name__ == '__main__':
    # Host, port is required for successful ssh connection
    app.run(debug=True, host='0.0.0.0', port=8050)