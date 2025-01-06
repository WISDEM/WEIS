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
    component_types = ['blade', 'hub', 'nacelle', 'tower', 'substructure']


@callback(Output('geom-3d-names', 'options'),
          Input('geometry-components', 'data'))
def list_labels(geom_comps_by_names):
    return list(set([k.split(':')[0] for k in list(geom_comps_by_names.keys())]))



@callback(Output('vtk-view', 'children'),
          Input('3d-viz-btn', 'n_clicks'),
          State('geom-3d-names', 'value'),
          State('geom-types', 'value'),
          State('wt-options', 'data'))
def visualize(nClicks, geom_3d_names, geom_types, wt_options_by_names):
    '''
    This function is for visualizing per geometry component types from selected file data
    '''
    if nClicks==0:
        raise PreventUpdate

    geometries = []
    xMax, xMin, yMax, yMin, zMax, zMin = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # colors_scale = px.colors.sequential.Plasma
    # colors_scale = px.colors.qualitative.Plotly
    default_rgbs = plotly.colors.DEFAULT_PLOTLY_COLORS
    colors_scale = [list(map(lambda x: int(x)/255, c[4:-1].split(', '))) for c in default_rgbs]
    colors_scale = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]]      # temporary..


    for idx, gname in enumerate(geom_3d_names):         # gname: selected geometry file (selected filelabelname)

        # Set 'local' flag depending on the size of components
        if len(geom_types) > 1:
            print('Multiple components selected', geom_types)
            local = False
        
        elif len(geom_types) == 1:
            print('Single component selected', geom_types)
            local = True
        
        else:
            print('nothing selected..')
        
        # Render, add meshes
        print(f"{gname} with wt_options\n {wt_options_by_names[gname].keys()}")     # dict_keys(['name', 'assembly', 'components', 'airfoils', 'materials', 'control', 'environment', 'bos', 'costs'])

        # Note: Geometry Representation include only one mesh. Cannot include multiple meshes..
        if 'tower' in geom_types:
            
            tower_mesh_state, _ , tower_points = render_Tower(wt_options_by_names[gname], local)
            xMin = np.append(xMin, tower_points[0])
            xMax = np.append(xMax, tower_points[1])
            yMin = np.append(yMin, tower_points[2])
            yMax = np.append(yMax, tower_points[3])
            zMin = np.append(zMin, tower_points[4])
            zMax = np.append(zMax, tower_points[5])
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=tower_mesh_state)],
                property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
         
        if 'blade' in geom_types:

            blade_mesh_state, _ , blade_points = render_blade(wt_options_by_names[gname], local)
            xMin = np.append(xMin, blade_points[0])
            xMax = np.append(xMax, blade_points[1])
            yMin = np.append(yMin, blade_points[2])
            yMax = np.append(yMax, blade_points[3])
            zMin = np.append(zMin, blade_points[4])
            zMax = np.append(zMax, blade_points[5])

            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=blade_mesh_state)],
                property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]

        if 'hub' in geom_types:
            hub_mesh_state, _ , hub_points = render_hub(wt_options_by_names[gname], local)
            
            xMin = np.append(xMin, hub_points[0])
            xMax = np.append(xMax, hub_points[1])
            yMin = np.append(yMin, hub_points[2])
            yMax = np.append(yMax, hub_points[3])
            zMin = np.append(zMin, hub_points[4])
            zMax = np.append(zMax, hub_points[5])

            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=hub_mesh_state)],
                property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
        if 'substructure' in geom_types:
            if 'monopile' in list(wt_options_by_names[gname]['components'].keys()):
                sub_mesh_state, _ , sub_points = render_monopile(wt_options_by_names[gname], local)
            
            elif 'floating_platform' in list(wt_options_by_names[gname]['components'].keys()):
                sub_mesh_state, _, sub_points = render_floatingPlatform(wt_options_by_names[gname], local)
            
            else:
                continue

            xMin = np.append(xMin, sub_points[0])
            xMax = np.append(xMax, sub_points[1])
            yMin = np.append(yMin, sub_points[2])
            yMax = np.append(yMax, sub_points[3])
            zMin = np.append(zMin, sub_points[4])
            zMax = np.append(zMax, sub_points[5])
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=sub_mesh_state)],
                property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
        if 'nacelle' in geom_types:
            nacelle_mesh_state, _, nacelle_points = render_nacelle(wt_options_by_names[gname], local)
            
            xMin = np.append(xMin, nacelle_points[0])
            xMax = np.append(xMax, nacelle_points[1])
            yMin = np.append(yMin, nacelle_points[2])
            yMax = np.append(yMax, nacelle_points[3])
            zMin = np.append(zMin, nacelle_points[4])
            zMax = np.append(zMax, nacelle_points[5])
            
            # Add by geom data (same color over components from the turbine)
            geometries += [dash_vtk.GeometryRepresentation(
                children=[dash_vtk.Mesh(state=nacelle_mesh_state)],
                property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            )]
            
    # Add grid coordinates using max-sized box - a single global coordinate over geometries
    bounds=[xMin.min(), xMax.max(), yMin.min(), yMax.max(), zMin.min(), zMax.max()]
    
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
                dbc.Row(vtk_view, className='card')
            ]))

    return layout