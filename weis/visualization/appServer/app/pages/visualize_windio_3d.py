'''This is the page for visualize the WEIS inputs in 3D VTK model'''

import dash
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


#############################################################
# Initialize global variables
# - Geometry Representations will be updated from callbacks
#############################################################
component_types = ['blade', 'hub', 'nacelle', 'tower', 'substructure']
geometries = [
                dash_vtk.GeometryRepresentation(
                    id = f'{idx}-{gtype}-rep',
                ) for idx in range(10) for gtype in component_types         # We are expecting less than 10 geometry files..
            ] + [
                dash_vtk.GeometryRepresentation(
                    id = 'axes',
                    showCubeAxes=True,      # Always show origins
                    property={'color': [0,0,0], 'opacity': 0}
                )
            ]

colors_scale = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1]]      # TODO: Change it into something dynamic one..? Or, default plotly color panels..?


@callback(Output('geom-3d-names', 'options'),
          Input('geometry-components', 'data'))
def list_labels(geom_comps_by_names):
    return list(set([k.split(':')[0] for k in list(geom_comps_by_names.keys())]))


@callback(Output('vtk-view-container', 'children'),
          Output('geometry-color', 'children'),
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
    xMax, xMin, yMax, yMin, zMax, zMin = {}, {}, {}, {}, {}, {}

    for gtype in component_types:
        xMax[gtype] = np.array([])
        xMin[gtype] = np.array([])
        yMax[gtype] = np.array([])
        yMin[gtype] = np.array([])
        zMax[gtype] = np.array([])
        zMin[gtype] = np.array([])
    
    geometries_with_blade_size = []

    ################################
    # Add geometry meshes
    ################################
    local = False
    color_description = []
    for idx, gname in enumerate(geom_3d_names):
        color_description += [html.Span(f'  {gname}   ', style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})]
        for gtype in component_types:
            if gtype == 'blade':
                mesh_state, _ , points = render_blade(wt_options_by_names[gname], local)
                blade_size = points[1] - points[0]
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
            
            xMin[gtype] = np.append(xMin[gtype], points[0])
            xMax[gtype] = np.append(xMax[gtype], points[1])
            yMin[gtype] = np.append(yMin[gtype], points[2])
            yMax[gtype] = np.append(yMax[gtype], points[3])
            zMin[gtype] = np.append(zMin[gtype], points[4])
            zMax[gtype] = np.append(zMax[gtype], points[5])

            geometries_with_blade_size.append({'bladeSize': blade_size,
                                               'geometry': dash_vtk.GeometryRepresentation(
                                                                            id = f'{idx}-{gtype}-rep',
                                                                            children = [dash_vtk.Mesh(state=mesh_state)],
                                                                            property = {'color': colors_scale[idx], 'opacity': 0.5})
                                                })
    
    #####################################################################################################
    # Add axes - Add grid coordinates using max-sized box - a single global coordinate over geometries
    #####################################################################################################
    bounds = [min([v for u in list(xMin.values()) for v in u]), max([v for u in list(xMax.values()) for v in u]), min([v for u in list(yMin.values()) for v in u]), max([v for u in list(yMax.values()) for v in u]), min([v for u in list(zMin.values()) for v in u]), max([v for u in list(zMax.values()) for v in u])]

    geometries_with_blade_size.append({'bladeSize': 0,
                                       'geometry': dash_vtk.GeometryRepresentation(
                                                                    id = 'axes',
                                                                    children = [dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
                                                                    showCubeAxes=True,      # Always show origins
                                                                    property = {'color': [0,0,0], 'opacity': 0})
                                        })
    
    # Sort geometries by blade size -- so that we can draw small wind turbine later
    geometries = [entry['geometry'] for entry in sorted(geometries_with_blade_size, key=lambda x: x['bladeSize'], reverse=True)]

    # Front face - where xy-plane is on the ground and z direction upwards
    setCameraPosition = [-1, 0, 0]
    setCameraViewUp = [0, 0, 1]

    # Add rendered meshes to the final content at the end (at once!)
    return dash_vtk.View(id='vtk-view', 
                         cameraPosition=setCameraPosition,
                         cameraViewUp=setCameraViewUp,
                         children=geometries,
                         pickingModes=['click']), color_description
    

################################################
# Handle Local VTK View
################################################

@callback(Output('vtk-view-local-div', 'is_open'),
          Input('vtk-view', 'clickInfo'),
          State('vtk-view-local-div', 'is_open'))
def toggle_local_scene(info, is_open):
    if (not info) or ("representationId" not in info):
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

    if gtype == 'tower':
        return dcc.Link(dbc.Button("Tower Properties"), href='/windio_tower')
    
    elif gtype == 'blade':
        return dcc.Link(dbc.Button("Blade Properties"), href='/windio_blade')

    elif gtype == 'hub':
        data = [wt_options_by_names[gname]['components'][gtype] for gname in geom_3d_names]
        columns = list(dict.fromkeys(key for dictionary in data for key, value in dictionary.items() if not isinstance(value, list) and not isinstance(value, dict)).keys())        # Get union of dictionary keys only where its value is single value type

        # style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'}
        table_columns = [html.Th(c) for c in ["Label"]+columns]
        table_rows = []
        for idx, dictionary in enumerate(data):
            row = [html.Td(html.P(geom_3d_names[idx], style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'}))]
            row += [html.Td(html.Code(dictionary[c], style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})) if c in dictionary.keys() else html.Td("-") for c in columns]
            table_rows.append(html.Tr(row))
        
        table_header = [html.Thead(html.Tr(table_columns))]
        table_body = [html.Tbody(table_rows)]

        table = dbc.Table(table_header + table_body, bordered=True)
    
    elif gtype == 'nacelle':
        multiindex_df = {}
        for field in ['drivetrain', 'generator']:
            data = [wt_options_by_names[gname]['components'][gtype][field] if field in wt_options_by_names[gname]['components'][gtype] else {} for gname in geom_3d_names]
            columns = list(dict.fromkeys(key for dictionary in data for key, value in dictionary.items() if not isinstance(value, list) and not isinstance(value, dict)).keys())

            for c in columns:
                multiindex_df[(field, c)] = {geom_3d_names[idx] : (html.Code(dictionary[c], style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'}) if c in dictionary else html.Code("-", style={'color': f'rgb({colors_scale[idx][0]*255}, {colors_scale[idx][1]*255}, {colors_scale[idx][2]*255})'})) for idx, dictionary in enumerate(data)}

        
        df = pd.DataFrame(multiindex_df)
        df.index.set_names("Label", inplace=True)
        # TODO: Freeze first column with fixed_columns={'headers': True, 'data': 1} => Doesn't work..
        table = dbc.Table.from_dataframe(df, bordered=True, index=True)

    elif gtype == 'substructure':
        # TODO: How and What to visualize for substructure tables?
        table = dbc.Table(children=[])      # Return blank table for now..

    return table


@callback(Output('vtk-view-local-container', 'children'),
          Input('vtk-view', 'clickInfo'),
          State('geom-3d-names', 'value'),
          State('wt-options', 'data'))
def update_local_scene_content(info, geom_3d_names, wt_options_by_names):
    
    if (not info) or ("representationId" not in info):
        raise PreventUpdate
    
    gnameIdx, gtype, _ = info["representationId"].split('-')
    geometries_local_with_comp_size = []
    local = True
    xMax, xMin, yMax, yMin, zMax, zMin = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

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
            continue
            # if 'monopile' in list(wt_options_by_names[gname]['components'].keys()):
            #     mesh_state, _ , points = render_monopile(wt_options_by_names[gname], local)     
            # elif 'floating_platform' in list(wt_options_by_names[gname]['components'].keys()):
            #     mesh_state, _, points = render_floatingPlatform(wt_options_by_names[gname], local)
            # else:
            #     continue
        
        xMin = np.append(xMin, points[0])
        xMax = np.append(xMax, points[1])
        yMin = np.append(yMin, points[2])
        yMax = np.append(yMax, points[3])
        zMin = np.append(zMin, points[4])
        zMax = np.append(zMax, points[5])

        geometries_local_with_comp_size.append({'compSize': points[1]-points[0],
                                                'geometry': dash_vtk.GeometryRepresentation(id=gname,
                                                                                            children=[dash_vtk.Mesh(state=mesh_state)],
                                                                                            property={'color': colors_scale[idx], 'opacity': 0.5},
                                                                                            showCubeAxes=False)
                                                })

    # Add grid coordinates using max-sized box - a single global coordinate over geometries
    bounds = [xMin.min(), xMax.max(), yMin.min(), yMax.max(), zMin.min(), zMax.max()]
    geometries_local_with_comp_size.append({'compSize': 0,
                                             'geometry': dash_vtk.GeometryRepresentation(
                                                                children=[dash_vtk.Mesh(state=to_mesh_state(pv.Box(bounds=bounds)))],
                                                                showCubeAxes=True,      # Show origins
                                                                property={'color': [255, 255, 255], 'opacity': 0}     # Make the object totally transparent.. We just need axes
                                                            )
                                            })

    # Sort geometries by blade size -- so that we can draw small wind turbine later
    geometries_local = [entry['geometry'] for entry in sorted(geometries_local_with_comp_size, key=lambda x: x['compSize'], reverse=True)]

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

    return dash_vtk.View(id='vtk-view-local', children=geometries_local+[tooltip], pickingModes=['click'])


@callback(Output('tooltip', 'children'),
          Output('tooltip', 'style'),
          Input('vtk-view-local', 'clickInfo'))
def click_local_view(info):

    if (not info) or ("representationId" not in info):
        return [""], dash.no_update
        
    print('Clicked: ', info["representationId"])
    style = {
        "position": "absolute",
        "bottom": info['displayPosition'][1],
        "left": info['displayPosition'][0],
        "zIndex": info['displayPosition'][2],
        "color": "white"
    }

    return [f"{info['representationId']}: {info['worldPosition']}"], style



# We are using card container where we define sublayout with rows and cols.
def layout():
    # Define layout
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
                dbc.Row([
                    dbc.Col(geom_inputs),
                ], className='g-0'),         # No gutters where horizontal spacing is added between the columns by default
                dcc.Loading(vtk_view),       # Global View

                # Modal Window layout for visualizing Local View
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(html.Div(id='vtk-view-local-header'))),
                    dbc.ModalBody([html.Div(id='vtk-text-description', style={"overflow": "scroll"}), vtk_view_local])],
                    id='vtk-view-local-div',
                    size='xl',
                    is_open=False)
            ])

    return layout