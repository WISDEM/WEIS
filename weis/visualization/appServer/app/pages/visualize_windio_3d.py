'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
from weis.visualization.utils import *
from weis.visualization.meshRender import *

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
          Input('3d-viz-btn', 'n_clicks'),
          State('geom-3d-names', 'value'),
          State('geom-types', 'value'),
          State('geometry-components', 'data'),
          State('airfoil-by-names', 'data'))
def visualize(nClicks, geom_3d_names, geom_types, geom_comps_by_names,airfoils_by_names):
    '''
    This function is for visualizing per geometry component types from selected file data
    '''
    # if geom_3d_names is None or geom_types == []:
    #     raise PreventUpdate
    if nClicks==0:
        raise PreventUpdate

    tower_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'tower' in k}     # where now k is 'filelabelname' and v is dict
    blade_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'blade' in k}     # where now k is 'filelabelname' and v is dict
    hub_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'hub' in k}     
    nacelle_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'nacelle' in k}

    # need logic for different sub structure types that are to be supported
    monopile_by_names = {k.split(':')[0]: v for k, v in geom_comps_by_names.items() if 'monopile' in k}

    geometries = []

    for idx, gname in enumerate(geom_3d_names):         # gname: selected geometry file
        meshes = []

        # Set 'Coords' flag depending on the size of components
        if len(geom_types) > 1:
            print('Multiple components selected', geom_types)
            coords = 'global'
        
        elif len(geom_types) == 1:
            print('Single component selected', geom_types)
            coords = 'local'
        
        else:
            print('nothing selected..')
        
        # Render, add meshes
        if 'tower' in geom_types:

            meshes += [dash_vtk.Mesh(
                                    state=render_cylinderTower(tower_by_names[gname], coords)
                                        )]
                
        if 'blade' in geom_types:

            airfoil_used = blade_by_names[gname]['outer_shape_bem']['airfoil_position']['labels']
            selectAirfoils = {}
            for a in airfoil_used:
                selectAirfoils[a] = airfoils_by_names[f'{gname}: {a}']

            meshes += [dash_vtk.Mesh(
                                    state=render_blade_only(blade_by_names[gname], 
                                        selectAirfoils, coords)
                                        )]
            
        if 'hub' in geom_types:
            meshes += [dash_vtk.Mesh(
                                    state=render_hub_only(hub_by_names[gname], coords)
                                    )]
            
        if 'monopile' in geom_types:
            meshes += [dash_vtk.Mesh(
                                    state=render_monopile_only(monopile_by_names[gname], coords)
                                    )]
            
        if 'nacelle' in geom_types:
            meshes += [dash_vtk.Mesh(
                                    state=render_nacelle_only(nacelle_by_names[gname], hub_by_names[gname], coords)
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


# We are using card container where we define sublayout with rows and cols.
def layout():
    # Define which components to be visualized from the geometry files
    set_components()

    # Define layout for tower structures
    geom_items = dcc.Dropdown(id='geom-3d-names', options=[], value=None, multi=True)

    geom_inputs = dbc.Card([
                            dbc.CardHeader('Geometry Data'),
                            dbc.CardBody([
                                dbc.Form(geom_items)
                            ])
                        ], className='card')
    
    # type_items = dcc.Checklist(id='geom-types', options=component_types, value=[], inline=True)
    type_items = dbc.Row([
                    dbc.Col(dbc.Form(dcc.Checklist(id='geom-types', options=component_types, value=[], inline=True))),
                    dbc.Col(dbc.Button('Visualize', id='3d-viz-btn', n_clicks=0, color='primary'), width='auto')
                ])

    type_inputs = dbc.Card([
                        dbc.CardHeader('Components to Visualize'),
                        dbc.CardBody([
                            type_items
                        ])
                    ], className='card')

    vtk_view = html.Div(
                    id='vtk-view',
                    style={"width": "100%", "height": "600px"},
                )

    layout = dcc.Loading(html.Div([
                dcc.Store(id='meshes', data=[]),
                dbc.Row([
                    dbc.Col(geom_inputs, width=4),
                    dbc.Col(type_inputs, width=8)
                ], className='g-0'),         # No gutters where horizontal spacing is added between the columns by default
                dbc.Row(vtk_view)
            ]))

    return layout