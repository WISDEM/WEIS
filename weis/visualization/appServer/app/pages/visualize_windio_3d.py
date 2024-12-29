'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *

register_page(
    __name__,
    name='WindIO',
    top_nav=True,
    path='/windio_3d'
)

def set_components():
    global component_types
    component_types = ['blade', 'hub', 'nacelle', 'tower', 'monopile', 'floating_platform', 'mooring']


@callback(Output('geom-3d-names', 'options'),
          Input('geometry-components', 'data'))
def list_labels(geom_comps_by_names):
    return list(set([k.split(':')[0] for k in list(geom_comps_by_names.keys())]))



@callback(Output('vtk-view', 'children'),
          Input('geom-3d-names', 'value'),
          Input('geom-types', 'value'),
          State('geometry-components', 'data'),
          State('airfoil-by-names', 'data'))
def visualize(geom_3d_names, geom_types, geom_comps_by_names,airfoils_by_names):
    '''
    This function is for visualizing per geometry component types from selected file data
    '''
    if geom_3d_names is None or geom_types == []:
        raise PreventUpdate

    tower_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'tower' in k}     # where now k is 'filelabelname' and v is dict
    blade_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'blade' in k}     # where now k is 'filelabelname' and v is dict
    hub_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'hub' in k}     
    nacelle_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'nacelle' in k}

    

    geometries = []

    print(airfoils_by_names.keys())

    for idx, gname in enumerate(geom_3d_names):
        meshes = []
        for gtype in geom_types:
            if gtype == 'tower':


                airfoil_used = blade_by_names[gname]['outer_shape_bem']['airfoil_position']['labels']
                selectAirfoils = {}
                for a in airfoil_used:
                    selectAirfoils[a] = airfoils_by_names[f'{gname}: {a}']

                meshes += [dash_vtk.Mesh(

                                        # state=render_cylinderTower(tower_by_names[gname]['outer_shape_bem']['outer_diameter']['grid'], 
                                        #     tower_by_names[gname]['outer_shape_bem']['outer_diameter']['values'])

                                        state=render_blade(blade_by_names[gname], 
                                            selectAirfoils)
                                            
                                            )]
            

        # Add by geom data (same color over components from the turbine)
        geometries += [dash_vtk.GeometryRepresentation(
            children=meshes,
            showCubeAxes=True,      # Show origins
            property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]

    # Add rendered meshes to the final content at the end (at once!)    
    content = dash_vtk.View(geometries)
    
    return content



def make_card(comp_type):
    return dbc.Card([
                dbc.CardHeader(f'{comp_type.title()}', className='cardHeader'),
                dbc.CardBody([
                    html.Div(id=f'{comp_type}-meshes')
                ])
           ])



def define_mesh_layout():
    # layout = [dbc.Row([
    #                     dbc.Row([
    #                         # First column
    #                         dbc.Col(html.Div(
    #                             style={'width':'100%', 'height':'600px'},
    #                             children=[meshes[idx]])),
    #                         # Second column
    #                         dbc.Col(html.Div(
    #                             style={'width':'100%', 'height':'600px'},
    #                             children=[meshes[idx+1]]))
    #                     ], className='wrapper') for idx in range(0, len(meshes)-1, 2)], className='wrapper')]

    cards = [make_card(comp) for comp in component_types]

    blade_layout = dbc.Card([
                        dbc.CardHeader('Blade', className='cardHeader'),
                        dbc.CardBody([
                            html.Div(id='blade-meshes')
                        ])
                    ], style={'height': '600px'})
    
    hub_layout = dbc.Card([
                        dbc.CardHeader('Hub', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '600px'})
    
    nacelle_layout = dbc.Card([
                        dbc.CardHeader('Nacelle', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '600px'})
    

    tower_layout = dbc.Card([
                        dbc.CardHeader('Tower', className='cardHeader'),
                        dbc.CardBody([
                            html.Div(dash_vtk.View([
                                        dash_vtk.GeometryRepresentation(
                                            id='tower-meshes',
                                            children=[],
                                            showCubeAxes=True,      # Show origins
                                            # colorMapPreset='jet',   # Doesn't work
                                            # colorDataRange=[0.2, 0.9]   # Doesn't work
                                            # opcacity=0.5            # Set the transparency
                                            property={"edgeVisibility": False, 'color': [(0,0,0), (0.5, 0.5, 0.5)]},
                                            # mapper={
                                            #     "colorByArrayName":"data",
                                            #     "scalarMode": 1,
                                            #     "interpolateScalarsBeforeMapping": True,
                                            #     "useLookupTableScalarRange":True,
                                            #     "colorMode":1,
                                            #     "GetArray":1,
                                            #     "scalarVisibility":True,
                                            # },
                                            # property={
                                            #     "edgeVisibility": False,
                                            #     'representation': 2,
                                            # },
                                            # actor={
                                            #     "visibility":1,
                                            # },
                                            # colorMapPreset="rainbow",
                                        )
                                    ]),       # Need to define width, height for div itself to show up
                                    style={'width':'100%', 'height':'100%'})
                        ])
                    ], style={'height': '600px'})
    
    monopile_layout = dbc.Card([
                        dbc.CardHeader('Monopile', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '600px'})
    

    floating_layout = dbc.Card([
                        dbc.CardHeader('Floating', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '280px'})
    
    mooring_layout = dbc.Card([
                        dbc.CardHeader('Mooring', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '280px'})
    
    full_layout = dbc.Card([
                        dbc.CardHeader('Full', className='cardHeader'),
                        dbc.CardBody([
                        ])
                    ], style={'height': '600px'})

    layout = html.Div([
                dbc.Row([
                    dbc.Col(blade_layout, width=6),
                    dbc.Col(hub_layout, width=3),
                    dbc.Col(nacelle_layout, width=3),
                ]),
                dbc.Row([
                    dbc.Col(tower_layout, width=3),
                    dbc.Col(monopile_layout, width=3),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(floating_layout),
                        ]),
                        dbc.Row([
                            dbc.Col(mooring_layout),
                        ])
                    ], width=3),
                    dbc.Col(full_layout, width=3),
                ])
            ])
    

    return layout


# We are using card container where we define sublayout with rows and cols.
def layout():
    # Define which components to be visualized from the geometry files
    set_components()

    # Define layout for tower structures
    geom_items = dcc.Dropdown(id='geom-3d-names', options=[], value=None, multi=True)

    geom_inputs = html.Div([
                        dbc.Card([
                            dbc.CardHeader('Geometry Data'),
                            dbc.CardBody([
                                dbc.Form(geom_items)
                            ])
                        ])
                    ])
    
    type_items = dcc.Checklist(id='geom-types', options=component_types, value=[])

    type_inputs = html.Div([
                    dbc.Card([
                        dbc.CardHeader('Components to Visualize'),
                        dbc.CardBody([
                            dbc.Form(type_items)
                        ])
                    ])
                ])
    
    controls = [geom_inputs, type_inputs]

    vtk_view = html.Div(
                    id='vtk-view',
                    style={"width": "100%", "height": "400px"},
                )

    vtk_view_own = html.Div(
                        style={"width": "100%", "height": "400px"},
                        children=[dash_vtk.View([
                                dash_vtk.GeometryRepresentation(
                                    mapper={'orientationArray': 'Normals'},
                                    children=[dash_vtk.Mesh(state=render_cylinderTower())],
                                    showCubeAxes=True,      # Show origins
                                )
                            ])]
                    )   

    layout = dcc.Loading(html.Div([
                dcc.Store(id='meshes', data=[]),
                dbc.Row([
                    dbc.Col(controls, width=4),
                    dbc.Col(vtk_view, width=8)
                ]),
                dbc.Row(vtk_view_own)
            ], className='wrapper'))

    return layout