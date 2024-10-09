from dash import html, register_page
from dash import dcc, Input, State, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from weis.visualization.utils import parse_yaml, dict_to_html

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
                            type='text', placeholder='Enter file path to import'
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dbc.Button('Submit', id='import-btn', color='primary'),
                        width='auto'
                    )
                ], className="mb-3")
    
    nickname_input = dbc.Row([
                        dbc.Label('Nickname', width=2),
                        dbc.Col(
                            dbc.Input(
                                type='text', placeholder='Enter nickname'
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
                                        {'label': 'Model', 'value': 1},
                                        {'label': 'Analysis', 'value': 2},
                                        {'label': 'Geometry', 'value':3}
                                    ]
                                ),
                                width='auto'
                            )
                        ], className="mb-3")

    form_layout = dbc.Card([
                            dbc.CardHeader("Import WEIS Input Files"),
                            dbc.CardBody([
                                dbc.Form([file_input, nickname_input, type_input])
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


@callback(Output('file-table', 'data'),
          Input('import-btn', 'click'))
def update_table(btn):
    pass