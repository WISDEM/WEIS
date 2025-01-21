'''This is the page for visualize the WEIS inputs in 3D simulation model'''

import random
import dash
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


########
# Initialize global variables
# - Geometry Representations will be updated from callbacks
########
global component_types
component_types = ['blade', 'hub', 'nacelle', 'tower', 'substructure']

# global geometries
geometries = [
                dash_vtk.GeometryRepresentation(
                    id = f'{idx}-{gtype}-rep'
                ) for idx in range(5) for gtype in component_types         # We are expecting less than 10 geometry files..
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = f'axes-{gtype}',
                    property={'color': [0.5,0.5,0.5], 'opacity': 0}        # The axes to be shown when each component type has been clicked
                ) for gtype in component_types
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = f'axes-blade-local',
                    property={'color': [0.5,0.5,0.5], 'opacity': 0}        # The blade local axes to be shown when blade is clicked
                )
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = 'axes',
                    showCubeAxes=True,      # Always show origins
                    property={'color': [0,0,0], 'opacity': 0}
                )
            ]


@callback(Output('geom-3d-names', 'options'),
          Input('geometry-components', 'data'))
def list_labels(geom_comps_by_names):
    return list(set([k.split(':')[0] for k in list(geom_comps_by_names.keys())]))


@callback(Output('vtk-view-container', 'children'),
          Output('geometry-color', 'children'),
          Output('local-meshes', 'data'),
          Input('3d-viz-btn', 'n_clicks'),
          State('geom-3d-names', 'value'),
          State('wt-options', 'data'))
def initial_loading(nClicks, geom_3d_names, wt_options_by_names):
    '''
    This function is for visualizing per geometry component types from selected file data
    '''
    if nClicks==0:
        raise PreventUpdate

    global geometries
    xMax, xMin, yMax, yMin, zMax, zMin, local_meshes = {}, {}, {}, {}, {}, {}, {}
    xMaxBlade, xMinBlade, yMaxBlade, yMinBlade, zMaxBlade, zMinBlade = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for gtype in component_types:
        xMax[gtype] = np.array([])
        xMin[gtype] = np.array([])
        yMax[gtype] = np.array([])
        yMin[gtype] = np.array([])
        zMax[gtype] = np.array([])
        zMin[gtype] = np.array([])

    colors_scale = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]]      # temporary..

    geometries = [
                dash_vtk.GeometryRepresentation(
                    id = f'{idx}-{gtype}-rep'
                ) for idx in range(5) for gtype in component_types         # We are expecting less than 10 geometry files..
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = f'axes-{gtype}',
                    property={'color': [0.5,0.5,0.5], 'opacity': 0}     # Make the object totally transparent.. We just need axes
                ) for gtype in component_types
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = f'axes-blade-local',
                    property={'color': [0.5,0.5,0.5], 'opacity': 0}        # The axes to be shown when each component type has been clicked
                )
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = 'axes',
                    showCubeAxes=True,      # Show origins
                    property={'color': [0,0,0], 'opacity': 0}     # Make the object totally transparent.. We just need axes
                )
            ]

    ########
    # Add geometry meshes
    ########
    vtk_idx = 0
    local = False
    color_description = []
    while vtk_idx < len(geom_3d_names) * len(component_types):
        for idx, gname in enumerate(geom_3d_names):
            # color_description += [html.P(f'{gname}', style={'color': 'rgb(255, 0, 0)'})]  # Works
            color_description += [html.P(f'{gname}', style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})]
            for gtype in component_types:
                if gtype == 'blade':
                    mesh_state, _ , points = render_blade(wt_options_by_names[gname], local)
                    _, _, local_blade_points = render_blade(wt_options_by_names[gname], local=True)     # Note that points for blade are from local coordinates, not global.. (To show one blade, not three blades!)

                    xMinBlade = np.append(xMinBlade, local_blade_points[0])
                    xMaxBlade = np.append(xMaxBlade, local_blade_points[1])
                    yMinBlade = np.append(yMinBlade, local_blade_points[2])
                    yMaxBlade = np.append(yMaxBlade, local_blade_points[3])
                    zMinBlade = np.append(zMinBlade, local_blade_points[4])
                    zMaxBlade = np.append(zMaxBlade, local_blade_points[5])

                elif gtype == 'hub':
                    mesh_state, _ , points = render_hub(wt_options_by_names[gname], local)
                elif gtype == 'nacelle':
                    mesh_state, _, points = render_nacelle(wt_options_by_names[gname], local)
                elif gtype == 'tower':
                    mesh_state, _ , points = render_Tower(wt_options_by_names[gname], local)
                elif gtype == 'substructure':
                    if 'monopile' in list(wt_options_by_names[gname]['components'].keys()):
                        mesh_state, _ , points = render_monopile(wt_options_by_names[gname], local)
                    
                    elif 'floating_platform' in list(wt_options_by_names[gname]['components'].keys()):
                        mesh_state, _, points = render_floatingPlatform(wt_options_by_names[gname], local)
                    
                    else:
                        vtk_idx += 1
                        continue
                
                xMin[gtype] = np.append(xMin[gtype], points[0])
                xMax[gtype] = np.append(xMax[gtype], points[1])
                yMin[gtype] = np.append(yMin[gtype], points[2])
                yMax[gtype] = np.append(yMax[gtype], points[3])
                zMin[gtype] = np.append(zMin[gtype], points[4])
                zMax[gtype] = np.append(zMax[gtype], points[5])

                geometries[vtk_idx].children = [dash_vtk.Mesh(state=mesh_state)]
                geometries[vtk_idx].property = {'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
                vtk_idx += 1
    
    #####
    # Add axes - Add grid coordinates using max-sized box - a single global coordinate over geometries
    #####
    bounds = [min([v for u in list(xMin.values()) for v in u]), max([v for u in list(xMax.values()) for v in u]), min([v for u in list(yMin.values()) for v in u]), max([v for u in list(yMax.values()) for v in u]), min([v for u in list(zMin.values()) for v in u]), max([v for u in list(zMax.values()) for v in u])]
    
    for geom_vtk in geometries:
        if geom_vtk.id == 'axes':
            geom_vtk.children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))]
        
        elif geom_vtk.id == 'axes-blade-local':
            geom_vtk.children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=[xMinBlade.min(), xMaxBlade.max(), yMinBlade.min(), yMaxBlade.max(), zMinBlade.min(), zMaxBlade.max()])))]
        
        for gtype in component_types:
            if geom_vtk.id == f'axes-{gtype}':
                try:
                    geom_vtk.children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=[xMin[gtype].min(), xMax[gtype].max(), yMin[gtype].min(), yMax[gtype].max(), zMin[gtype].min(), zMax[gtype].max()])))]
                except:
                    continue
            



    # Front face - where xy-plane is on the ground and z direction upwards
    setCameraPosition = [-1, 0, 0]
    setCameraViewUp = [0, 0, 1]
    # Add rendered meshes to the final content at the end (at once!)    
    return dash_vtk.View(id='vtk-view', 
                         cameraPosition=setCameraPosition,
                         cameraViewUp=setCameraViewUp,
                         children=geometries,
                         pickingModes=['click']), color_description, local_meshes
    

@callback([Output(geom_vtk.id, 'showCubeAxes') for geom_vtk in geometries],
          Input('vtk-view', 'clickInfo'))
def update_full_scene(info):
    
    if info:
        if "representationId" not in info:
            return [True if geom_vtk.id=='axes' else False for geom_vtk in geometries]
        
        gnameIdx, gtype, _ = info["representationId"].split('-')
        print('Clicked: ', gtype)
        geo_viz = []
        if gtype == 'blade':
            for vtk_idx, geom_vtk in enumerate(geometries):
                if geom_vtk.id == f'axes-blade-local' or geom_vtk.id == 'axes':
                    # geo_viz.append({"visibility": 1})
                    geo_viz.append(True)
                else:
                    # geo_viz.append({"visibility": 0})
                    geo_viz.append(False)
        else:
            for vtk_idx, geom_vtk in enumerate(geometries):
                if geom_vtk.id == f'axes-{gtype}' or geom_vtk.id == 'axes':
                    # geo_viz.append({"visibility": 1})
                    geo_viz.append(True)
                else:
                    # geo_viz.append({"visibility": 0})
                    geo_viz.append(False)

        return geo_viz
    
    return [True if geom_vtk.id=='axes' else False for geom_vtk in geometries]



# @callback(Output('vtk-view-local-container', 'children'),
#           Input('vtk-view', 'clickInfo'),
#           State('local-meshes', 'data'))
# def update_local_scene(info, local_meshes):
    
#     if not info:
#         raise PreventUpdate

#     if "representationId" not in info:
#         raise PreventUpdate
    
#     gnameIdx, gtype, _ = info["representationId"].split('-')
#     print('Clicked: ', gtype)
    
#     print('geometries\n', geometries)
#     geometries_local = []
#     for vtk_idx, geom_vtk in enumerate(geometries):
#         if gtype in geom_vtk.id or geom_vtk.id == 'axes':
#             if geom_vtk.children:
#                 print('adding: ', geom_vtk.id, geom_vtk.children)
#             geometries_local += [dash_vtk.GeometryRepresentation(children=geom_vtk.children)]

#     return dash_vtk.View(children=geometries_local)



    # for idx, gname in enumerate(geom_3d_names):         # gname: selected geometry file (selected filelabelname)

    #     # Set 'local' flag depending on the size of components
    #     local = False
        
    #     # Render, add meshes
    #     # Note: Geometry Representation include only one mesh. Cannot include multiple meshes..

    #     ################################################################################
    #     # Add blade geometry representation
    #     ################################################################################
    #     blade_mesh_state, _ , blade_points = render_blade(wt_options_by_names[gname], local)
    #     xMin = np.append(xMin, blade_points[0])
    #     xMax = np.append(xMax, blade_points[1])
    #     yMin = np.append(yMin, blade_points[2])
    #     yMax = np.append(yMax, blade_points[3])
    #     zMin = np.append(zMin, blade_points[4])
    #     zMax = np.append(zMax, blade_points[5])

        
    #     # Add by geom data (same color over components from the turbine)
    #     geometries += [dash_vtk.GeometryRepresentation(
    #         id=f"{gname}-blade-rep",
    #         children=[dash_vtk.Mesh(state=blade_mesh_state)],
    #         actor={"visibility": 1} if 'blade' in geom_types else {"visibility": 0},
    #         property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #         # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #     )]

    #     ################################################################################
    #     # Add hub geometry representation
    #     ################################################################################
    #     hub_mesh_state, _ , hub_points = render_hub(wt_options_by_names[gname], local)
        
    #     xMin = np.append(xMin, hub_points[0])
    #     xMax = np.append(xMax, hub_points[1])
    #     yMin = np.append(yMin, hub_points[2])
    #     yMax = np.append(yMax, hub_points[3])
    #     zMin = np.append(zMin, hub_points[4])
    #     zMax = np.append(zMax, hub_points[5])

    #     # Add by geom data (same color over components from the turbine)
    #     geometries += [dash_vtk.GeometryRepresentation(
    #         id=f"{gname}-hub-rep",
    #         children=[dash_vtk.Mesh(state=hub_mesh_state)],
    #         actor={"visibility": 1} if 'hub' in geom_types else {"visibility": 0},
    #         property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #         # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #     )]

    #     ################################################################################
    #     # Add nacelle geometry representation
    #     ################################################################################
    #     nacelle_mesh_state, _, nacelle_points = render_nacelle(wt_options_by_names[gname], local)
        
    #     xMin = np.append(xMin, nacelle_points[0])
    #     xMax = np.append(xMax, nacelle_points[1])
    #     yMin = np.append(yMin, nacelle_points[2])
    #     yMax = np.append(yMax, nacelle_points[3])
    #     zMin = np.append(zMin, nacelle_points[4])
    #     zMax = np.append(zMax, nacelle_points[5])
        
    #     # Add by geom data (same color over components from the turbine)
    #     geometries += [dash_vtk.GeometryRepresentation(
    #         id=f"{gname}-nacelle-rep",
    #         children=[dash_vtk.Mesh(state=nacelle_mesh_state)],
    #         actor={"visibility": 1} if 'nacelle' in geom_types else {"visibility": 0},
    #         property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #         # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #     )]

    #     ################################################################################
    #     # Add tower geometry representation
    #     ################################################################################
    #     tower_mesh_state, _ , tower_points = render_Tower(wt_options_by_names[gname], local)
    #     xMin = np.append(xMin, tower_points[0])
    #     xMax = np.append(xMax, tower_points[1])
    #     yMin = np.append(yMin, tower_points[2])
    #     yMax = np.append(yMax, tower_points[3])
    #     zMin = np.append(zMin, tower_points[4])
    #     zMax = np.append(zMax, tower_points[5])
        
    #     # Add by geom data (same color over components from the turbine)
    #     geometries += [dash_vtk.GeometryRepresentation(
    #         id=f"{gname}-tower-rep",
    #         children=[dash_vtk.Mesh(state=tower_mesh_state)],
    #         actor={"visibility": 1} if 'tower' in geom_types else {"visibility": 0},
    #         property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #         # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #     )]

    #     ################################################################################
    #     # Add substructure geometry representation
    #     ################################################################################
    #     if 'monopile' in list(wt_options_by_names[gname]['components'].keys()):
    #         sub_mesh_state, _ , sub_points = render_monopile(wt_options_by_names[gname], local)
        
    #     elif 'floating_platform' in list(wt_options_by_names[gname]['components'].keys()):
    #         sub_mesh_state, _, sub_points = render_floatingPlatform(wt_options_by_names[gname], local)
        
    #     else:
    #         continue

    #     xMin = np.append(xMin, sub_points[0])
    #     xMax = np.append(xMax, sub_points[1])
    #     yMin = np.append(yMin, sub_points[2])
    #     yMax = np.append(yMax, sub_points[3])
    #     zMin = np.append(zMin, sub_points[4])
    #     zMax = np.append(zMax, sub_points[5])
        
    #     # Add by geom data (same color over components from the turbine)
    #     geometries += [dash_vtk.GeometryRepresentation(
    #         id=f"{gname}-substructure-rep",
    #         children=[dash_vtk.Mesh(state=sub_mesh_state)],
    #         actor={"visibility": 1} if 'substructure' in geom_types else {"visibility": 0},
    #         property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #         # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
    #     )]
            
    # # Add grid coordinates using max-sized box - a single global coordinate over geometries
    # bounds=[xMin.min(), xMax.max(), yMin.min(), yMax.max(), zMin.min(), zMax.max()]
    
    # geometries += [dash_vtk.GeometryRepresentation(
    #     id="axes",
    #     children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
    #     showCubeAxes=True,      # Show origins
    #     property={'color': [255, 255, 255], 'opacity': 0}     # Make the object totally transparent.. We just need axes
    # )]

    # ################################################################################
    # # Pointer to indicate where it's clicked
    # ################################################################################
    # cone_pointer = dash_vtk.GeometryRepresentation(
    #     property = {"color": [0, 0, 0]},
    #     children = [dash_vtk.Algorithm(id="pointer", vtkClass="vtkConeSource")]
    # )

    # tooltip = html.Pre(
    #     id = "tooltip",
    #     style = {
    #         "position": "absolute",
    #         "bottom": "25px",
    #         "left": "25px",
    #         "zIndex": 1,
    #         "color": "white"
    #     }
    # )

    # # Add rendered meshes to the final content at the end (at once!)    
    # return dash_vtk.View(id='vtk-view', 
    #                         children=geometries + [cone_pointer, tooltip],
    #                         pickingModes=['hover'])

"""
@callback(Output('vtk-view-container', 'children'),
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
        local = False
        
        # Render, add meshes
        # Note: Geometry Representation include only one mesh. Cannot include multiple meshes..

        ################################################################################
        # Add blade geometry representation
        ################################################################################
        blade_mesh_state, _ , blade_points = render_blade(wt_options_by_names[gname], local)
        xMin = np.append(xMin, blade_points[0])
        xMax = np.append(xMax, blade_points[1])
        yMin = np.append(yMin, blade_points[2])
        yMax = np.append(yMax, blade_points[3])
        zMin = np.append(zMin, blade_points[4])
        zMax = np.append(zMax, blade_points[5])

        
        # Add by geom data (same color over components from the turbine)
        geometries += [dash_vtk.GeometryRepresentation(
            id=f"{gname}-blade-rep",
            children=[dash_vtk.Mesh(state=blade_mesh_state)],
            actor={"visibility": 1} if 'blade' in geom_types else {"visibility": 0},
            property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]

        ################################################################################
        # Add hub geometry representation
        ################################################################################
        hub_mesh_state, _ , hub_points = render_hub(wt_options_by_names[gname], local)
        
        xMin = np.append(xMin, hub_points[0])
        xMax = np.append(xMax, hub_points[1])
        yMin = np.append(yMin, hub_points[2])
        yMax = np.append(yMax, hub_points[3])
        zMin = np.append(zMin, hub_points[4])
        zMax = np.append(zMax, hub_points[5])

        # Add by geom data (same color over components from the turbine)
        geometries += [dash_vtk.GeometryRepresentation(
            id=f"{gname}-hub-rep",
            children=[dash_vtk.Mesh(state=hub_mesh_state)],
            actor={"visibility": 1} if 'hub' in geom_types else {"visibility": 0},
            property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]

        ################################################################################
        # Add nacelle geometry representation
        ################################################################################
        nacelle_mesh_state, _, nacelle_points = render_nacelle(wt_options_by_names[gname], local)
        
        xMin = np.append(xMin, nacelle_points[0])
        xMax = np.append(xMax, nacelle_points[1])
        yMin = np.append(yMin, nacelle_points[2])
        yMax = np.append(yMax, nacelle_points[3])
        zMin = np.append(zMin, nacelle_points[4])
        zMax = np.append(zMax, nacelle_points[5])
        
        # Add by geom data (same color over components from the turbine)
        geometries += [dash_vtk.GeometryRepresentation(
            id=f"{gname}-nacelle-rep",
            children=[dash_vtk.Mesh(state=nacelle_mesh_state)],
            actor={"visibility": 1} if 'nacelle' in geom_types else {"visibility": 0},
            property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]

        ################################################################################
        # Add tower geometry representation
        ################################################################################
        tower_mesh_state, _ , tower_points = render_Tower(wt_options_by_names[gname], local)
        xMin = np.append(xMin, tower_points[0])
        xMax = np.append(xMax, tower_points[1])
        yMin = np.append(yMin, tower_points[2])
        yMax = np.append(yMax, tower_points[3])
        zMin = np.append(zMin, tower_points[4])
        zMax = np.append(zMax, tower_points[5])
        
        # Add by geom data (same color over components from the turbine)
        geometries += [dash_vtk.GeometryRepresentation(
            id=f"{gname}-tower-rep",
            children=[dash_vtk.Mesh(state=tower_mesh_state)],
            actor={"visibility": 1} if 'tower' in geom_types else {"visibility": 0},
            property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]

        ################################################################################
        # Add substructure geometry representation
        ################################################################################
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
            id=f"{gname}-substructure-rep",
            children=[dash_vtk.Mesh(state=sub_mesh_state)],
            actor={"visibility": 1} if 'substructure' in geom_types else {"visibility": 0},
            property={'color': colors_scale[idx], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
            # property={'color': [idx+1/len(geom_3d_names), idx/len(geom_3d_names), 0], 'opacity': 0.5}     # Color: (r,g,b) with [0,1] range
        )]
            
    # Add grid coordinates using max-sized box - a single global coordinate over geometries
    bounds=[xMin.min(), xMax.max(), yMin.min(), yMax.max(), zMin.min(), zMax.max()]
    
    geometries += [dash_vtk.GeometryRepresentation(
        id="axes",
        children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
        showCubeAxes=True,      # Show origins
        property={'color': [255, 255, 255], 'opacity': 0}     # Make the object totally transparent.. We just need axes
    )]

    ################################################################################
    # Pointer to indicate where it's clicked
    ################################################################################
    cone_pointer = dash_vtk.GeometryRepresentation(
        property = {"color": [0, 0, 0]},
        children = [dash_vtk.Algorithm(id="pointer", vtkClass="vtkConeSource")]
    )

    tooltip = html.Pre(
        id = "tooltip",
        style = {
            "position": "absolute",
            "bottom": "25px",
            "left": "25px",
            "zIndex": 1,
            "color": "white"
        }
    )

    # Add rendered meshes to the final content at the end (at once!)    
    return dash_vtk.View(id='vtk-view', 
                            children=geometries + [cone_pointer, tooltip],
                            pickingModes=['hover'])


@callback(Output('tooltip', 'children'),
          Output('tooltip', 'style'),
          Output('abc-tower-rep', 'showCubeAxes'),
          Output('abc-tower-rep', 'property'),
          Output('vtk-view', 'triggerRender'),
          Input('vtk-view', 'hoverInfo'),
          State('vtk-view', 'children'))
def probe_data(hoverInfo, geometries_with_pointers):
    '''
    Callback structure
    _____________________
    displayPosition: The x,y,z coordinate with on the user's screen.
    ray: A line between two points in 3D space (xyz1, xyz2) that represent the mouse position. It covers the full space under the 2D mouse position.
    representationId: The ID assigned to the dash_vtk.GeometryRepresentation containing your object.
    worldPosition: The x, y, z coordinates in the 3D environment that you are rendering where the ray hit the object. It corresponds to the 3D coordinate on the surface of the object under your mouse.
    '''
    cone_state = {"resolution": 12}

    print('probe data..')

    if hoverInfo:
        print('hoverInfo\n', hoverInfo)
        if "representationId" not in hoverInfo:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        gname, gtype, _ = hoverInfo["representationId"].split('-')

        for geometry in geometries_with_pointers[:-2]:
            # print('geometry\n', geometry.keys())        # dict_keys(['props', 'type', 'namespace'])

            # print(geometry['props'].keys())             # dict_keys(['children', 'id', 'actor', 'property'])
            # print(geometry['type'])                     # GeometryRepresentation
            # print(geometry['namespace'])                # dash_vtk
            # print(geometry['props']['id'])
            if geometry['props']['id'] == hoverInfo["representationId"]:
                print('clicked: ', geometry['props']['id'])
                geometry['props']['showCubeAxes']=True
        
        style = {
            "position": "absolute",
            "bottom": hoverInfo['displayPosition'][1],
            "left": hoverInfo['displayPosition'][0],
            "zIndex": hoverInfo['displayPosition'][2],
            "color": "white"
        }

        return [f"{hoverInfo['representationId']}: {hoverInfo['worldPosition']}"], style, True, {'color': [0,0,1], 'opacity': 0.5}, random.random()


    # elif clickInfo:
    #     print('clickInfo\n', clickInfo)
    #     if "representationId" not in clickInfo:
    #         return dash.no_update, dash.no_update, dash.no_update
        
    #     gname, gtype, _ = clickInfo["representationId"].split('-')

    #     for geometry in geometries_with_pointers[:-2]:
    #         # print('geometry\n', geometry.keys())        # dict_keys(['props', 'type', 'namespace'])

    #         # print(geometry['props'].keys())             # dict_keys(['children', 'id', 'actor', 'property'])
    #         # print(geometry['type'])                     # GeometryRepresentation
    #         # print(geometry['namespace'])                # dash_vtk
    #         # print(geometry['props']['id'])
    #         if geometry['props']['id'] == clickInfo["representationId"]:
    #             print('clicked: ', geometry['props']['id'])
    #             geometry['props']['property']={'color': [0,0,1], 'opacity': 0.5}

    #     # info
    #     # {'worldPosition': [253.33301878217912, -56.93506994027632, 227.71971074152225], 'displayPosition': [319, 451, 0.9961089494163424], 'compositeID': 16, 'representationId': 'abcde-nacelle-rep', 'ray': [[-804.434149795535, 159.10463863288712, -14.680539061758665], [666.5233190078518, -141.3255811016689, 322.4073083209296]]}

    #     style = {
    #         "position": "absolute",
    #         "bottom": clickInfo['displayPosition'][1],
    #         "left": clickInfo['displayPosition'][0],
    #         "zIndex": clickInfo['displayPosition'][2],
    #         "color": "white"
    #     }

    #     return [f"{clickInfo['representationId']}: {clickInfo['worldPosition']}"], style, random.random()
    
    return [""], dash.no_update, dash.no_update, dash.no_update
"""


# We are using card container where we define sublayout with rows and cols.
def layout():
    # Define which components to be visualized from the geometry files
    # set_components()

    # define_initial_geometries()     # Define inital geometery representations with expecting 10 arbitary meshes.. This is for letting it accessible from other callback functions!

    # Define layout for tower structures
    geom_items = dcc.Dropdown(id='geom-3d-names', options=[], value=None, multi=True)

    geom_inputs = dbc.Card([
                            dbc.CardHeader('Geometry Data'),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(dbc.Form(geom_items)),
                                    dbc.Col(dbc.Button('Visualize', id='3d-viz-btn', n_clicks=0, color='primary'), width='auto')
                                ])
                            ])
                        ], className='card')
    
    # type_items = dbc.Row([
    #                 # dbc.Col(dbc.Form(dcc.Checklist(id='geom-types', options=component_types, value=[], inline=True))),
    #                 # dbc.Col(dbc.Button('Visualize', id='3d-viz-btn', n_clicks=0, color='primary'), width='auto')
    #             ])

    # type_inputs = dbc.Card([
    #                     dbc.CardHeader('Components to Visualize'),
    #                     dbc.CardBody([
    #                         type_items
    #                     ])
    #                 ], className='card')

    vtk_view = html.Div([
                        html.Div(dbc.Spinner(color="primary"),
                                    style={
                                        "background-color": "#ffffff",
                                        "height": "calc(100vh - 230px)",
                                        "width": "100%",
                                        "text-align": "center",
                                        "padding-top": "calc(50vh - 105px)"
                                    })
                    ],
                    id='vtk-view-container',
                    style={"height": "calc(100vh - 230px)", "width": "100%"},
                )
    
    # vtk_view_local = html.Div([
    #                     html.Div(dbc.Spinner(color="primary"),
    #                                 style={
    #                                     "background-color": "#ffffff",
    #                                     "height": "calc(100vh - 230px)",
    #                                     "width": "100%",
    #                                     "text-align": "center",
    #                                     "padding-top": "calc(50vh - 105px)"
    #                                 })
    #                 ],
    #                 id='vtk-view-local-container',
    #                 style={"height": "calc(100vh - 230px)", "width": "100%"},
    #             )

    layout = html.Div([
                dcc.Store(id='local-meshes', data={}),
                dcc.Loading(dbc.Row([
                    dbc.Col(geom_inputs),
                    # dbc.Col(type_inputs, width=8)
                ], className='g-0')),         # No gutters where horizontal spacing is added between the columns by default
                # html.Div(id='geometry-color', style={'text-align': 'center', 'display': 'inline-flex', 'width': '100%'}),
                html.Div(id='geometry-color', style={'text-align': 'center'}),
                dbc.Row(vtk_view, className='card'),
                # dbc.Row(vtk_view_local, className='card')
            ])

    return layout