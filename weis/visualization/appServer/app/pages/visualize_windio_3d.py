'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly
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
          State('airfoil-by-names', 'data'),
          State('wt-options', 'data'))
def visualize(nClicks, geom_3d_names, geom_types, geom_comps_by_names, airfoils_by_names, wt_options_by_names):
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
    x, y, z = np.array([]), np.array([]), np.array([])

    # colors_scale = px.colors.sequential.Plasma
    # colors_scale = px.colors.qualitative.Plotly
    default_rgbs = plotly.colors.DEFAULT_PLOTLY_COLORS
    colors_scale = [list(map(lambda x: int(x)/255, c[4:-1].split(', '))) for c in default_rgbs]


    for idx, gname in enumerate(geom_3d_names):         # gname: selected geometry file (selected filelabelname)

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
        print(f"{gname} with wt_options\n {wt_options_by_names[gname]}")

        # Note: Geometry Representation include only one mesh. Cannot include multiple meshes..
        if 'tower' in geom_types:
            
            tower_meshes, tower_points = render_cylinderTower(tower_by_names[gname], coords)
            tower_pointsX, tower_pointsY, tower_pointsZ = tower_points
            x = np.concatenate((x, tower_pointsX))
            y = np.concatenate((y, tower_pointsY))
            z = np.concatenate((z, tower_pointsZ))
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=tower_meshes)],
                property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
         
        if 'blade' in geom_types:

            airfoil_used = blade_by_names[gname]['outer_shape_bem']['airfoil_position']['labels']
            selectAirfoils = {}
            for a in airfoil_used:
                selectAirfoils[a] = airfoils_by_names[f'{gname}: {a}']

            blade_meshes, blade_points = render_blade_only(blade_by_names[gname], selectAirfoils, coords)
            blade_pointsX, blade_pointsY, blade_pointsZ = blade_points
            x = np.concatenate((x, blade_pointsX))
            y = np.concatenate((y, blade_pointsY))
            z = np.concatenate((z, blade_pointsZ))
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=blade_meshes)],
                property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]

        if 'hub' in geom_types:
            hub_meshes, hub_points = render_hub_only(hub_by_names[gname], coords)
            
            hub_pointsX, hub_pointsY, hub_pointsZ = hub_points
            x = np.concatenate((x, hub_pointsX))
            y = np.concatenate((y, hub_pointsY))
            z = np.concatenate((z, hub_pointsZ))

            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=hub_meshes)],
                property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
        if 'monopile' in geom_types:
            monopile_meshes, monopile_points = render_monopile_only(monopile_by_names[gname], coords)

            monopile_pointsX, monopile_pointsY, monopile_pointsZ = monopile_points
            x = np.concatenate((x, monopile_pointsX))
            y = np.concatenate((y, monopile_pointsY))
            z = np.concatenate((z, monopile_pointsZ))
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=monopile_meshes)],
                property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
        if 'nacelle' in geom_types:
            nacelle_meshes, nacelle_points = render_nacelle_only(nacelle_by_names[gname], hub_by_names[gname], coords)
            nacelle_pointsX, nacelle_pointsY, nacelle_pointsZ = nacelle_points
            x = np.concatenate((x, nacelle_pointsX))
            y = np.concatenate((y, nacelle_pointsY))
            z = np.concatenate((z, nacelle_pointsZ))
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=nacelle_meshes)],
                property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
    # Add grid coordinates using max-sized box - a single global coordinate over geometries
    bounds=[x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
    
    geometries += [dash_vtk.GeometryRepresentation(
        children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
        showCubeAxes=True,      # Show origins
        property={'color': [255, 255, 255], 'opacity': 0}     # Make the object totally transparent.. We just need axes
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
    
    box = pv.Box(
        bounds=[-1.0,1.0,-1.0,1.0,-1.0,1.0]
    )
    mesh_state = to_mesh_state(box)


    vtk_view_own = html.Div(
                        style={"width": "100%", "height": "400px"},
                        children=[dash_vtk.View([
                                dash_vtk.GeometryRepresentation(
                                    mapper={'orientationArray': 'Normals'},
                                    children=[dash_vtk.Mesh(state=render_cylinderTower()[0])],
                                    property={'color': [1, 0, 0], 'opacity': 0.5}
                                ),
                                dash_vtk.GeometryRepresentation(
                                    mapper={'orientationArray': 'Normals'},
                                    children=[dash_vtk.Mesh(state=mesh_state)],
                                    property={'color': [0, 1, 0], 'opacity': 0.5}
                                ),
                                dash_vtk.GeometryRepresentation(
                                    children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=[-2.0,2.0,-2.0,2.0,-2.0,2.0])))],
                                    property={'color': [0, 0, 1], 'opacity': 0},
                                    showCubeAxes=True
                                )
                            ])]
                    )  

    layout = dcc.Loading(html.Div([
                dcc.Store(id='meshes', data=[]),
                dbc.Row([
                    dbc.Col(geom_inputs, width=4),
                    dbc.Col(type_inputs, width=8)
                ], className='g-0'),         # No gutters where horizontal spacing is added between the columns by default
                dbc.Row(vtk_view, className='card'),
                dbc.Row(vtk_view_own, className='card')
            ]))

    return layout