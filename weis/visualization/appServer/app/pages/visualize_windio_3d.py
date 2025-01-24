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
            # color_description += [html.P(f'{gname}', style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})]
            color_description += [html.Span(f'  {gname}   ', style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})]
            # color_description += [f"<span style='color: rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'>{gname}</span>"]
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
        
        # elif geom_vtk.id == 'axes-blade-local':
        #     geom_vtk.children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=[xMinBlade.min(), xMaxBlade.max(), yMinBlade.min(), yMaxBlade.max(), zMinBlade.min(), zMaxBlade.max()])))]
        
        # for gtype in component_types:
        #     if geom_vtk.id == f'axes-{gtype}':
        #         try:
        #             geom_vtk.children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=[xMin[gtype].min(), xMax[gtype].max(), yMin[gtype].min(), yMax[gtype].max(), zMin[gtype].min(), zMax[gtype].max()])))]
        #         except:
        #             continue
            



    # Front face - where xy-plane is on the ground and z direction upwards
    setCameraPosition = [-1, 0, 0]
    setCameraViewUp = [0, 0, 1]

    # Add rendered meshes to the final content at the end (at once!)    
    return dash_vtk.View(id='vtk-view', 
                         cameraPosition=setCameraPosition,
                         cameraViewUp=setCameraViewUp,
                         children=geometries,
                         pickingModes=['click']), color_description, local_meshes
    

"""
@callback([Output(geom_vtk.id, 'showCubeAxes') for geom_vtk in geometries],
          Output('tooltip', 'children'),
          Output('tooltip', 'style'),
          Input('vtk-view', 'clickInfo'))
def update_full_scene(info):
    
    if info:
        if "representationId" not in info:
            return [True if geom_vtk.id=='axes' else False for geom_vtk in geometries], dash.no_update, dash.no_update
        
        gnameIdx, gtype, _ = info["representationId"].split('-')
        # print('Clicked: ', gtype)
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
        style = {
            "position": "absolute",
            "bottom": info['displayPosition'][1],
            "left": info['displayPosition'][0],
            "zIndex": info['displayPosition'][2],
            "color": "white"
        }


        return geo_viz, [f"{info['representationId']}: {info['worldPosition']}"], style
    
    return [True if geom_vtk.id=='axes' else False for geom_vtk in geometries], [""], dash.no_update
"""

########################
# Handle Local VTK View
########################

@callback(Output('vtk-view-local-div', 'is_open'),
          Input('vtk-view', 'clickInfo'),
          State('vtk-view-local-div', 'is_open'))
def toggle_local_scene(info, is_open):
    if not info:
        raise PreventUpdate

    if "representationId" not in info:
        raise PreventUpdate
    
    return toggle(info, is_open)


@callback(Output('vtk-view-local-header', 'children'),
          Input('vtk-view', 'clickInfo'))
def update_local_header(info):

    if (not info) or ("representationId" not in info):
        raise PreventUpdate
    
    _, gtype, _ = info["representationId"].split('-')

    return gtype


@callback(Output('vtk-text-description', 'children'),
          Input('vtk-view', 'clickInfo'),
          State('geom-3d-names', 'value'),
          State('wt-options', 'data'))
def update_local_table_content(info, geom_3d_names, wt_options_by_names):
    '''
    Print yaml properties of each component with table format
    '''
    if (not info) or ("representationId" not in info):
        raise PreventUpdate

    _, gtype, _ = info["representationId"].split('-')

    data = [wt_options_by_names[gname]['components'][gtype] for gname in geom_3d_names]
    columns = list(dict.fromkeys(key for dictionary in data for key, value in dictionary.items() if not isinstance(value, list)).keys())        # Get union of dictionary keys only where its value is single value type

    table_columns = [html.Th(c) for c in [""]+columns]
    table_rows = []
    for idx, dictionary in enumerate(data):
        row = [html.Td(geom_3d_names[idx])]
        row += [html.Td(dictionary[c]) if c in dictionary.keys() else html.Td("") for c in columns]
        table_rows.append(html.Tr(row))


    # table_header = [
    #     html.Thead(html.Tr([html.Th("First Name"), html.Th("Last Name")]))
    # ]

    # row1 = html.Tr([html.Td("Arthur"), html.Td("Dent")])
    # row2 = html.Tr([html.Td("Ford"), html.Td("Prefect")])
    # row3 = html.Tr([html.Td("Zaphod"), html.Td("Beeblebrox")])
    # row4 = html.Tr([html.Td("Trillian"), html.Td("Astra")])

    # table_body = [html.Tbody([row1, row2, row3, row4])]

    table_header = [html.Thead(html.Tr(table_columns))]
    table_body = [html.Tbody(table_rows)]

    table = dbc.Table(table_header + table_body, bordered=True)

    return table


@callback(Output('vtk-view-local-container', 'children'),
          Input('vtk-view', 'clickInfo'),
          State('geom-3d-names', 'value'),
          State('wt-options', 'data'))
def update_local_scene_content(info, geom_3d_names, wt_options_by_names):
    
    if (not info) or ("representationId" not in info):
        raise PreventUpdate
    
    gnameIdx, gtype, _ = info["representationId"].split('-')
    # print('Clicked: ', gtype)
    
    # print('geometries\n', geometries)
    geometries_local = []
    local = True
    xMax, xMin, yMax, yMin, zMax, zMin = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    colors_scale = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]]      # temporary..

    for idx, gname in enumerate(geom_3d_names):
        if gtype == 'blade':
            mesh_state, _ , points = render_blade(wt_options_by_names[gname], local)
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
                continue
        
        xMin = np.append(xMin, points[0])
        xMax = np.append(xMax, points[1])
        yMin = np.append(yMin, points[2])
        yMax = np.append(yMax, points[3])
        zMin = np.append(zMin, points[4])
        zMax = np.append(zMax, points[5])

        geometries_local += [dash_vtk.GeometryRepresentation(id=gname, children=[dash_vtk.Mesh(state=mesh_state)], property={'color': colors_scale[idx], 'opacity': 0.5}, showCubeAxes=False)]

    # Add grid coordinates using max-sized box - a single global coordinate over geometries
    bounds = [xMin.min(), xMax.max(), yMin.min(), yMax.max(), zMin.min(), zMax.max()]
    geometries_local += [dash_vtk.GeometryRepresentation(
        children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
        showCubeAxes=True,      # Show origins
        property={'color': [255, 255, 255], 'opacity': 0}     # Make the object totally transparent.. We just need axes
    )]

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

    # for vtk_idx, geom_vtk in enumerate(geometries):
    #     if gtype in geom_vtk.id:
    #         if geom_vtk.children:
    #             print('adding: ', geom_vtk.id)
    #         geometries_local += [dash_vtk.GeometryRepresentation(children=geom_vtk.children, showCubeAxes=True)]

    return dash_vtk.View(id='vtk-view-local', children=geometries_local+[tooltip], pickingModes=['click'])


@callback(Output('tooltip', 'children'),
          Output('tooltip', 'style'),
          Input('vtk-view-local', 'clickInfo'))
def click_local_view(info):

    if info:
        if "representationId" not in info:
            return dash.no_update, dash.no_update
        
        print('Clicked: ', info["representationId"])
        style = {
            "position": "absolute",
            "bottom": info['displayPosition'][1],
            "left": info['displayPosition'][0],
            "zIndex": info['displayPosition'][2],
            "color": "white"
        }

        return [f"{info['representationId']}: {info['worldPosition']}"], style
    
    return [""], dash.no_update




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
    
    # vtk_view = dcc.Loading(html.Div(
    #                     id='vtk-view-container',
    #                     style={"height": "calc(100vh - 230px)", "width": "100%"},
    #                 ))
    vtk_view = dbc.Card([
                         dbc.CardHeader(html.Div(id='geometry-color', style={'text-align': 'center'})),
                         dbc.CardBody([
                             html.Div(
                                id='vtk-view-container',
                                style={"height": "calc(100vh - 230px)", "width": "100%"}
                             ) 
                         ])
                    ])
    
    vtk_view_local = dcc.Loading(
                            html.Div(
                            id='vtk-view-local-container',
                            style={"height": "calc(100vh - 230px)", "width": "100%"},
                        ))

    layout = html.Div([
                dcc.Store(id='local-meshes', data={}),
                dcc.Loading(dbc.Row([
                    dbc.Col(geom_inputs),
                ], className='g-0')),         # No gutters where horizontal spacing is added between the columns by default
                # html.Div(id='geometry-color', style={'text-align': 'center'}),
                # dbc.Row(vtk_view, className='card'),
                dcc.Loading(vtk_view),

                # Modal Window layout for visualizing Outlier timeseries data
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(html.Div(id='vtk-view-local-header'))),                                 # Related function: display_outlier()
                    dbc.ModalBody([html.Div(id='vtk-text-description', style={"overflow": "scroll"}), vtk_view_local])],                                                         # Related function: display_outlier()
                    id='vtk-view-local-div',
                    size='xl',
                    is_open=False)
            ])

    return layout