'''This is the page for visualize the WEIS inputs specialized in Blade OML, Axis, Elastic & Mass properties.'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio_blade'
)


@callback(Output('blade-names', 'options'),
          Output('blade-by-names', 'data'),
          Input('geometry-components', 'data'))
def load_blade_comps(geom_comps_by_names):
    '''
    This function is for loading blade related components
    '''
    blade_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'blade' in k}     # where now k is 'filelabelname' and v is dict
    
    return list(blade_by_names.keys()), blade_by_names


# We are using card container where we define sublayout with rows and cols.
def layout():

    # Define layout for blade oml properties
    blade_items = dcc.Dropdown(id='blade-names', options=[], value=None, multi=True)

    blade_inputs = html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Label('Please select blades:'),
                                dbc.Form(blade_items)
                            ])
                        ])
                   ])

    oml_layout = dbc.Card([
                        dbc.CardHeader('OML & Axis Properties'),
                        dbc.CardBody([
                            dcc.Graph(id='blade-oml', figure=empty_figure())
                        ])
                  ], className='card')

    # Define layout for elastic and mass properties
    matrix_layout = dbc.Card([
                        dbc.CardHeader('Elastic & Mass Properties'),
                        dbc.CardBody([
                            dcc.Graph(id='blade-elastic', figure=empty_figure()),
                            dcc.Graph(id='blade-mass', figure=empty_figure())
                        ])
                  ], className='card')
    
    layout = dcc.Loading(html.Div([
                dcc.Store(id='blade-by-names', data={}),
                blade_inputs,
                oml_layout,
                matrix_layout

            ]))
    
    return layout



@callback(Output('blade-oml', 'figure'),
          Input('blade-names', 'value'),
          Input('blade-by-names', 'data'))
def draw_blade_oml(blade_names, blade_by_names):
    if blade_names is None:
        raise PreventUpdate
    
    channels = ['chord', 'twist', 'pitch_axis']
    cols = set_colors()            # Set color panel
    fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True)  # 3 subplots where chord, twist, LE/TE are each plotted

    for idx, blade_name in enumerate(blade_names):
        # Add a trace per blade over subplots
        for row_idx, channel in enumerate(channels):
            trace = blade_by_names[blade_name]['outer_shape_bem'][channel]
            # LE/TE Equation
            if channel == 'pitch_axis':
                chord_values = np.array(blade_by_names[blade_name]['outer_shape_bem']['chord']['values'])
                pitchAxis = np.interp(blade_by_names[blade_name]['outer_shape_bem']['chord']['grid'], blade_by_names[blade_name]['outer_shape_bem']['pitch_axis']['grid'], blade_by_names[blade_name]['outer_shape_bem']['pitch_axis']['values'])
                leading_edge = pitchAxis * chord_values
                tailing_edge = leading_edge - chord_values

                fig.append_trace(go.Scatter(
                                    x = trace['grid'],
                                    y = leading_edge,
                                    mode = 'lines+markers',
                                    line = dict(color=cols[idx]),
                                    name = '-'.join([blade_name, 'LE']),
                                    showlegend=False),
                                    row = row_idx+1,
                                    col = 1)
                
                fig.append_trace(go.Scatter(
                                    x = trace['grid'],
                                    y = tailing_edge,
                                    mode = 'lines+markers',
                                    line = dict(color=cols[idx]),
                                    name = '-'.join([blade_name, 'TE']),
                                    showlegend=False),
                                    row = row_idx+1,
                                    col = 1)
                fig.update_yaxes(title_text='LE/TE', row=row_idx+1, col=1)
            
            # Chord, Twist
            else:
                fig.append_trace(go.Scatter(
                                    x = trace['grid'],
                                    y = trace['values'],
                                    mode = 'lines+markers',
                                    line = dict(color=cols[idx]),
                                    name = blade_name,
                                    showlegend=True if channel=='chord' else False),
                                    row = row_idx+1,
                                    col = 1)
            
                fig.update_yaxes(title_text=channel, row=row_idx+1, col=1)
    
    fig.update_layout(plot_bgcolor='white', legend=dict(orientation='h', yanchor='bottom', xanchor='right', x=1, y=1.02), height=600)
    fig.update_xaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror = True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_xaxes(title_text=f'grid', row=len(channels), col=1)

    return fig


@callback(Output('blade-elastic', 'figure'),
          Output('blade-mass', 'figure'),
          Input('blade-names', 'value'),
          Input('blade-by-names', 'data'))
def draw_blade_matrix(blade_names, blade_by_names):
    if blade_names is None:
        raise PreventUpdate

    # Initialize 6x6 matrices per blade
    subplot_titles = tuple(' ' for _ in range(6) for _ in range(6))
    cols = set_colors()            # Set color panel
    fig_elastic = make_subplots(rows=6, cols=6, subplot_titles=subplot_titles)
    fig_mass = make_subplots(rows=6, cols=6, subplot_titles=subplot_titles)

    for idx, blade_name in enumerate(blade_names):
        # There are some files which doesn't contain elastic properties..
        if 'elastic_properties_mb' in blade_by_names[blade_name].keys():
            stiff_matrix = blade_by_names[blade_name]['elastic_properties_mb']['six_x_six']['stiff_matrix']
            inertia_matrix = blade_by_names[blade_name]['elastic_properties_mb']['six_x_six']['inertia_matrix']

            stiff_grid = stiff_matrix['grid']
            stiff_values = np.array(stiff_matrix['values'])         # n rows x 21 cols (where 21 = 6+5+4+3+2+1)

            inertia_grid = inertia_matrix['grid']
            inertia_values = np.array(inertia_matrix['values'])     # n rows x 21 cols (where 21 = 6+5+4+3+2+1)

            counter = 0
            for pltRow in range(6):
                for pltCol in range(pltRow, 6):
                    # Define Stiff Matrix
                    fig_elastic.append_trace(go.Scatter(
                                            x = stiff_grid,
                                            y = stiff_values[:,counter],
                                            mode = 'lines',
                                            line = dict(color=cols[idx]),
                                            name = blade_name,
                                            showlegend = True if pltRow==0 and pltCol==0 else False),
                                            row = pltRow+1,
                                            col = pltCol+1)
                    
                    # Add xaxis label / subplot title
                    if pltRow == pltCol:
                        fig_elastic.update_xaxes(title_text='grid', showticklabels=True, row=pltRow+1, col=pltCol+1)
                    else:
                        fig_elastic.update_xaxes(showticklabels=False, row=pltRow+1, col=pltCol+1)
                    
                    fig_elastic.layout.annotations[6*pltRow + pltCol].text = f'K{pltRow+1}{pltCol+1}'
                    
                    fig_elastic.update_layout(title='Stiffness Matrix', yaxis=dict(tickformat='.1e'), margin=dict(t=100, b=50), legend=dict(orientation='h', yanchor='bottom', xanchor='right', x=1, y=1.02), height=1000)

                    # Define Mass Matrix
                    fig_mass.append_trace(go.Scatter(
                                            x = inertia_grid,
                                            y = inertia_values[:,counter],
                                            mode = 'lines',
                                            line = dict(color=cols[idx]),
                                            name = blade_name,
                                            showlegend = True if pltRow==0 and pltCol==0 else False),
                                            row = pltRow+1,
                                            col = pltCol+1)
                    
                    # Add xaxis label / subplot title
                    if pltRow == pltCol:
                        fig_mass.update_xaxes(title_text='grid', showticklabels=True, row=pltRow+1, col=pltCol+1)
                    else:
                        fig_mass.update_xaxes(showticklabels=False, row=pltRow+1, col=pltCol+1)
                    
                    fig_mass.layout.annotations[6*pltRow + pltCol].text = f'K{pltRow+1}{pltCol+1}'
                    
                    fig_mass.update_layout(title='Inertia Matrix', yaxis=dict(tickformat='.1e'), margin=dict(t=100, b=50), legend=dict(orientation='h', yanchor='bottom', xanchor='right', x=1, y=1.02), height=1000)

                    counter += 1


    return fig_elastic, fig_mass