'''This is the page for visualize the WISDEM outputs specialized in calculating costs'''

import dash_bootstrap_components as dbc
from dash import register_page, callback, Input, Output, dcc
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import plotly.figure_factory as ff
from weis.visualization.utils import empty_figure, read_cost_variables

register_page(
    __name__,
    name='WISDEM',
    top_nav=True,
    path='/wisdem_cost'
)

@callback(Output('var-wisdem-cost', 'data'),
          Input('input-dict', 'data'))
def read_variables(input_dict):
    # TODO: Redirect to the home page when missing input yaml file
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    # Read numpy file
    wisdem_output_path = input_dict['userPreferences']['wisdem']['output_path']
    csv_filepath = '/'.join([wisdem_output_path, f'{input_dict['userOptions']['output_fileName']}.csv'])
    refturb_variables = pd.read_csv(csv_filepath).set_index('variables').to_dict('index')

    cost_options = {}
    main_labels = ['turbine', 'rotor', 'nacelle', 'tower']
    rotor_labels = ['blade', 'pitch_system', 'hub', 'spinner']
    nacelle_labels = ['lss', 'main_bearing', 'gearbox', 'hss', 'generator', 'bedplate', 'yaw_system', 'hvac', 'cover', 'elec', 'controls', 'transformer', 'converter']

    cost_options['turbine'] = read_cost_variables(main_labels, refturb_variables)
    cost_options['rotor'] = read_cost_variables(rotor_labels, refturb_variables)
    cost_options['nacelle'] = read_cost_variables(nacelle_labels, refturb_variables)

    print("Parse variables from wisdem cost..\n", cost_options)

    return cost_options


def layout():

    description_layout = dbc.Card(
                            [
                                dbc.CardHeader('Cost Description', className='cardHeader'),
                                dbc.CardBody([
                                    dcc.Loading(dcc.Graph(id='description-cost', figure=empty_figure()))
                                ])
                            ], className='card')
    
    chart_layout = dbc.Card(
                        [
                            dbc.CardHeader('Cost Breakdown', className='cardHeader'),
                            dbc.CardBody([
                                dcc.Loading(dcc.Graph(id='cost-chart', figure=empty_figure())),
                            ])
                        ], className='card')

    layout = dbc.Row([
                # dcc.Location(id='url', refresh=False),
                dcc.Store(id='var-wisdem-cost', data={}),
                dbc.Col(description_layout, width=5),
                dbc.Col([
                    dbc.Row(chart_layout, justify='center')
                ], width=6)
            ], className='wrapper')
    

    return layout


@callback(Output('description-cost', 'figure'),
          Input('var-wisdem-cost', 'data'))
def draw_cost_table(cost_options):
    if cost_options is None:
        raise PreventUpdate

    # 1
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=tuple(cost_options.keys()),
                        vertical_spacing=0.00,
                        specs=[[{'type': 'table'}] for _ in range(3)])

    for i, (_, cost_matrix) in enumerate(cost_options.items()):
        fig.add_trace(
            go.Table(
                cells={'values': [[row[0] for row in cost_matrix[1:]], [f'{round(row[1]/1000, 3)} k USD' for row in cost_matrix[1:]]], 'height': 20}, header={'values': cost_matrix[0]}

            ), row=i+1, col=1)
        # fig = ff.create_table(cost_matrix)
    

    fig.update_layout(autosize=True, height=900, font_size=12, margin=dict(l=0, r=0, b=0, t=0))

    '''
    #2
    fig1 = ff.create_table(cost_options['turbine'])
    fig2 = ff.create_table(cost_options['rotor'])

    for i in range(len(fig1.data)):
        fig1.data[i].xaxis='x1'
        fig1.data[i].yaxis='y1'

    fig1.layout.xaxis1.update({'anchor': 'y1'})
    fig1.layout.yaxis1.update({'anchor': 'x1', 'domain': [.55, 1]})

    for i in range(len(fig2.data)):
        fig2.data[i].xaxis='x2'
        fig2.data[i].yaxis='y2'

    # initialize xaxis2 and yaxis2
    fig2['layout']['xaxis2'] = {}
    fig2['layout']['yaxis2'] = {}

    fig2.layout.xaxis2.update({'anchor': 'y2'})
    fig2.layout.yaxis2.update({'anchor': 'x2', 'domain': [0, .45]})


    fig = go.Figure()
    fig.add_traces([fig1.data[0], fig2.data[0]])
    '''


    return fig



@callback(Output('cost-chart', 'figure'),
          Input('var-wisdem-cost', 'data'))
def draw_cost_chart(cost_options):
    if cost_options is None:
        raise PreventUpdate
    
    labels, parents, values = [], [""], []

    for type, cost_matrix in cost_options.items():
        for item, cost in cost_matrix[1:]:
            labels.append(item)
            values.append(cost)

            if item == 'turbine':
                continue

            parents.append(type)


    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total'
    ))
    fig.update_traces(textinfo='label+percent parent')
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    return fig
