import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dash_vtk.utils import to_mesh_state
import pyvista as pv

try:
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except:
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import yaml


def render_blade(turbineData, local=True):

    # fetch airfoil names from array of airfoils
    airfoils_by_names = {}
    for a in turbineData['airfoils']:
        airfoils_by_names[a['name']] = a

    points = bladeMesh(turbineData['components']['blade'], airfoils_by_names)

    if local:
        mesh_state, mesh = render_our_own_delaunay(points)

    else:
        ## transforming the blade to be in the global coordinate system

        # Rotate about z-axis by 90-deg so that x is from HP to LP surface
        points = rotation_transformation(points, [0, 0, np.pi/2])

        # Translate to the hub position
        points = translation_transformation(points, [0, 0, turbineData['components']['hub']['diameter'] / 2])

        # if turbineData['assembly']['rotor_orientation'] has downwind (case insensitive) then rotate set downwindScalar to -1
        downwindScalar = -1 if 'downwind' in turbineData['assembly']['rotor_orientation'].lower() else 1

        # rotate about y-axis by the cone angle
        coneAngle = turbineData['components']['hub']['cone_angle'] * -1 * downwindScalar # in rads is about x-axis
        blade_1 = rotation_transformation(points, [0,           coneAngle,  0])
        # blade_2 = rotation_transformation(blade_1, [2*np.pi/3,   0,  0])
        # blade_3 = rotation_transformation(blade_1, [4*np.pi/3,   0,  0])
        # This achieves the symmetry of the rotor.

        _, mesh_1 = render_our_own_delaunay(blade_1)
        mesh_2 = mesh_1.rotate_x( 2*np.pi/3 * 180/np.pi, point=(0,0,0), inplace=False)
        mesh_3 = mesh_1.rotate_x( 4*np.pi/3 * 180/np.pi, point=(0,0,0), inplace=False)


        # angle by the tilt of the rotor
        tiltAngle = turbineData['components']['nacelle']['drivetrain']['uptilt'] * downwindScalar # in rads is about y-axis
        # blade_1 = rotation_transformation(blade_1, [0, tiltAngle, 0])
        # blade_2 = rotation_transformation(blade_2, [0, tiltAngle, 0])
        # blade_3 = rotation_transformation(blade_3, [0, tiltAngle, 0])

        mesh_1 = mesh_1.rotate_y(tiltAngle * 180/np.pi, point=(0,0,0), inplace=False)
        mesh_2 = mesh_2.rotate_y(tiltAngle * 180/np.pi, point=(0,0,0), inplace=False)
        mesh_3 = mesh_3.rotate_y(tiltAngle * 180/np.pi, point=(0,0,0), inplace=False)

        # Tanslation along the z-axis to the hub height
        zTranslation = turbineData['assembly']['hub_height']
        # Translation in the x-axis
        xTranslation = turbineData['components']['nacelle']['drivetrain']['distance_tt_hub'] * downwindScalar * np.cos(tiltAngle) * -1

        # blade_1 = translation_transformation(blade_1, [xTranslation, 0, zTranslation])
        # blade_2 = translation_transformation(blade_2, [xTranslation, 0, zTranslation])
        # blade_3 = translation_transformation(blade_3, [xTranslation, 0, zTranslation])

        mesh_1 = mesh_1.translate((xTranslation, 0, zTranslation), inplace=False)
        mesh_2 = mesh_2.translate((xTranslation, 0, zTranslation), inplace=False)
        mesh_3 = mesh_3.translate((xTranslation, 0, zTranslation), inplace=False)

        # _, mesh_1 = render_our_own_delaunay(blade_1)
        # _, mesh_2 = render_our_own_delaunay(blade_2)
        # _, mesh_3 = render_our_own_delaunay(blade_3)

        mesh = mesh_1.merge(mesh_2).merge(mesh_3)

        mesh_state = to_mesh_state(mesh)

    # extracting all the points from the 
    rotor = mesh.points

    extremes = extractExtremes(rotor)

    return mesh_state, mesh, extremes

def render_hub(turbineData, local=True):

    points = hubMesh(turbineData['components']['hub'])

    if local:
        mesh_state, mesh = render_our_own_delaunay(points)
    
    else:

        # if turbineData['assembly']['rotor_orientation'] has downwind (case insensitive) then rotate set downwindScalar to -1
        downwindScalar = -1 if 'downwind' in turbineData['assembly']['rotor_orientation'].lower() else 1

        # Rotate the hub about the y-axis by 90 - tilt angle
        tiltAngle = turbineData['components']['nacelle']['drivetrain']['uptilt'] * downwindScalar # in rads is about y-axis

        points = rotation_transformation(points, [0, -1 * (np.pi/2 - tiltAngle), 0])
        
        # Tanslation along the z-axis to the hub height
        zTranslation = turbineData['assembly']['hub_height']
        # Translation in the x-axis
        xTranslation = turbineData['components']['nacelle']['drivetrain']['distance_tt_hub'] * downwindScalar * np.cos(tiltAngle) * -1

        points = translation_transformation(points, [xTranslation, 0, zTranslation])

        mesh_state, mesh = render_our_own_delaunay(points)

    extremes = extractExtremes(points)

    return mesh_state, mesh, extremes

def render_Tower(turbineData, local=True):

    points = towerMesh(turbineData['components']['tower'])

    if local:
        mesh_state, mesh = render_our_own_delaunay(points)
    else:
        # Tower does not have any orientation or tilt or translations
        mesh_state, mesh = render_our_own_delaunay(points)

    extremes = extractExtremes(points)

    return mesh_state, mesh, extremes

def render_monopile(turbineData, local=True):

    points = monopileMesh(turbineData['components']['monopile'])

    if local:
        mesh_state, mesh = render_our_own_delaunay(points)
    else:
        # Monopile does not have any orientation or tilt or translations
        mesh_state, mesh = render_our_own_delaunay(points)

    extremes = extractExtremes(points)

    return mesh_state, mesh, extremes

def render_nacelle(turbineData, local=True):

    points = nacelleMesh(turbineData['components']['nacelle'],turbineData['components']['hub'])

    if local:
        mesh_state, mesh = render_our_own_delaunay(points)
    else:

        # if turbineData['assembly']['rotor_orientation'] has downwind (case insensitive) then rotate set downwindScalar to -1
        downwindScalar = -1 if 'downwind' in turbineData['assembly']['rotor_orientation'].lower() else 1

        # need to translate it above the tower, the nacelle is always above the tower
        zTranslation = turbineData['assembly']['hub_height']

        # move the nacelle so that the -x face is touching the hub
        xTranslation = turbineData['components']['nacelle']['drivetrain']['overhang'] * downwindScalar * 0.5

        points = translation_transformation(np.array(points), [xTranslation, 0, zTranslation])

        mesh_state, mesh = render_our_own_delaunay(points)

    extremes = extractExtremes(points)

    return mesh_state, mesh, extremes


def rotation_transformation(coords, angles):
    '''
    Apply rotation transformation to a set of coordinates
    here, angle(1) is the rotation about x-axis, angle(2) is the rotation about y-axis, angle(3) is the rotation about z-axis
    angles are in radians
    '''
    # Create rotation matrices for each angle
    rot_x = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
    rot_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    rot_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])

    # Apply rotation
    coords = np.dot(rot_x, coords)
    coords = np.dot(rot_y, coords)
    coords = np.dot(rot_z, coords)

    return coords
    

def translation_transformation(coords, translation):
    '''
    Apply translation transformation to a set of coordinates
    '''
    coords[0, :] += translation[0]
    coords[1, :] += translation[1]
    coords[2, :] += translation[2]

    return coords


def extractExtremes(points):
    '''
    Extract the minimum and maximum values of the points
    '''
    # no matter the shape, transform to 3xN
    points = np.array(points)
    if points.shape[0] != 3:
        points = points.T

    x_min = np.min(points[0])
    x_max = np.max(points[0])
    y_min = np.min(points[1])
    y_max = np.max(points[1])
    z_min = np.min(points[2])
    z_max = np.max(points[2])

    return (x_min, x_max, y_min, y_max, z_min, z_max)

###################################################
# Mesh generation for different turbine components
###################################################

def bladeMesh(bladeData, airfoils):
    '''
    Generate mesh for blade component
    '''
   # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    
    # interpolate the chord, twist, pitch axis and reference axis to the locations of the airfoils
    chord = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'], 
                        bladeData['outer_shape_bem']['chord']['grid'],
                        bladeData['outer_shape_bem']['chord']['values'])
    
    twist = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'],
                        bladeData['outer_shape_bem']['twist']['grid'],
                        bladeData['outer_shape_bem']['twist']['values'])
    
    pitch_axis = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'],
                            bladeData['outer_shape_bem']['pitch_axis']['grid'],
                            bladeData['outer_shape_bem']['pitch_axis']['values'])
    
    ref_axis_x = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['x']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['x']['values'])
    
    ref_axis_y = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['y']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['y']['values'])
    
    ref_axis_z = np.interp(bladeData['outer_shape_bem']['airfoil_position']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['z']['grid'],
                            bladeData['outer_shape_bem']['reference_axis']['z']['values'])
    

    # extract the airfoil names at the grid locations 
    airfoil_names = [bladeData['outer_shape_bem']['airfoil_position']['labels'][i] for i in range(len(bladeData['outer_shape_bem']['airfoil_position']['grid']))]

    # Arrays to store points
    x = np.array([])
    y = np.array([])
    z = np.array([])

    # For each blade station, we transform the coordinates and store
    for i in range(len(ref_axis_z)):
        # Get the airfoil coordinates for this section
        af_coordinates = np.array([airfoils[airfoil_names[i]]['coordinates']['x'], airfoils[airfoil_names[i]]['coordinates']['y']]).T
        
        # Scale by chord
        af_coordinates[:, 0] = (af_coordinates[:, 0] - pitch_axis[i]) * chord[i]
        af_coordinates[:, 1] = af_coordinates[:, 1] * chord[i]
        
        # Create rotation matrix for twist angle
        cos_t = np.cos(np.radians(twist[i]))
        sin_t = np.sin(np.radians(twist[i]))
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Apply rotation
        af_coordinates = np.dot(af_coordinates, rot_matrix.T)
        
        # Translate to reference axis position
        af_coordinates_3d = np.zeros((af_coordinates.shape[0], 3))
        af_coordinates_3d[:, 0] = af_coordinates[:, 0] + ref_axis_y[i]
        af_coordinates_3d[:, 1] = af_coordinates[:, 1] + -ref_axis_x[i] 
        af_coordinates_3d[:, 2] = np.full(af_coordinates.shape[0], ref_axis_z[i])

        # Append points
        x = np.append(x, af_coordinates_3d[:,0])
        y = np.append(y, af_coordinates_3d[:,1]) 
        z = np.append(z, af_coordinates_3d[:,2])

    points = (x, y, z)

    return points


def towerMesh(towerData):

    # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    
    towerGrid = np.interp(towerData['outer_shape_bem']['outer_diameter']['grid'],
                            towerData['outer_shape_bem']['reference_axis']['z']['grid'],
                            towerData['outer_shape_bem']['reference_axis']['z']['values'])
    TowerOD = towerData['outer_shape_bem']['outer_diameter']['values']

    # Create points for each cylinder segment
    for i in range(len(towerGrid)-1):
        # Get parameters for this segment
        h_start = towerGrid[i]
        h_end = towerGrid[i+1] 
        r_start = TowerOD[i]/2
        r_end = TowerOD[i+1]/2
        
        # Create points along height
        h_points = np.linspace(h_start, h_end, 20)
        
        # For each height, create a circle of points
        for h in h_points:
            # Calculate interpolated radius at this height
            r = r_start + (r_end - r_start) * (h - h_start)/(h_end - h_start)
            
            # Create circle of points at this height
            theta = np.linspace(0, 2*np.pi, 36)
            x = np.append(x, r * np.cos(theta))
            y = np.append(y, r * np.sin(theta))
            z = np.append(z, np.full_like(theta, h))


    points = (x, y, z)
    return points

def hubMesh(hubData):

    # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    
    dia = hubData['diameter']
    h = 0.5 * dia
    '''
    let the height of the hub be 0.5 of the diameter
    The radius of the hub along the height varies from diameter to zero at the top and follows a parabolic function
    '''
    numPoints = 100
    theta = np.linspace(0, 2*np.pi, numPoints)
    z_dists = np.linspace(0, dia/2, 20)

    for z_dist in z_dists:
        r = dia/2 * np.sqrt(1 - (2*z_dist/dia)**2)
        x = np.append(x, r * np.cos(theta))
        y = np.append(y, r * np.sin(theta))
        z = np.append(z, np.full_like(theta, z_dist))

    points = (x, y, z)

    return points
    

def monopileMesh(monopileData):

    # Reusing the tower mesh function for monopile
    points = towerMesh(monopileData)

    return points
    
def nacelleMesh(nacelleData, hubData):

    # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    
    # if the nacelle dict has length, width and height, we can create a box
    if 'length' in nacelleData['drivetrain'].keys():
        length = nacelleData['drivetrain']['length']
        width = nacelleData['drivetrain']['width']
        height = nacelleData['drivetrain']['height']
        x = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]).T
        y = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]).T
        z = np.array([-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2]).T
    else:
        # if not, we use the overhang, distance from tower top to hub, and hub diameter to create the box such that
        # height = 1.5 * distance from tower top to hub
        # length = 2 * overhang
        # width = 1.5 * hub diameter
        height = 1.5 * nacelleData['drivetrain']['distance_tt_hub']
        length = 2 * nacelleData['drivetrain']['overhang']
        width =  1.5 * hubData['diameter']
        x = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]).T
        y = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]).T
        z = np.array([-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2]).T
    
    points = (x, y, z)

    return points

def floatingPlatformMesh(semisubData):

    # creating a few sub functions


    # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    

def render_our_own_delaunay(points):
    '''
    Create and fill the VTK Data Object with your own data using VTK library and pyvista high level api

    Reference: https://tutorial.pyvista.org/tutorial/06_vtk/b_create_vtk.html
    https://docs.pyvista.org/examples/00-load/create-tri-surface
    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.reconstruct_surface#pyvista.PolyDataFilters.reconstruct_surface
    '''

    # Join the points
    x, y, z = points
    values = np.c_[x.ravel(), y.ravel(), z.ravel()]     # (6400, 3) where each column is x, y, z coords
    coords = numpy_to_vtk(values)
    cloud = pv.PolyData(coords)
    # mesh = cloud.delaunay_2d()          # From point cloud, apply a 2D Delaunary filter to generate a 2d surface from a set of points on a plane.
    mesh = cloud.delaunay_3d()

    mesh_state = to_mesh_state(mesh)

    return mesh_state, mesh


def main():    
    # fetch the turbine data from the yaml file
    # with open('../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_VolturnUS-S.yaml') as file:
    with open('../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_Monopile.yaml') as file:
        turbine_data = yaml.load(file, Loader=yaml.FullLoader)

    # fetch airfoil names from array of airfoils
    airfoils_by_names = {}
    for a in turbine_data['airfoils']:
        airfoils_by_names[a['name']] = a

    ## render the blade
    _, meshBlade, _ = render_blade(turbine_data, local=False)
    # meshBlade = meshBlade.merge(pv.Sphere(center=(0, 0, 0), radius=10))
    # meshBlade.plot(show_edges=True)

    # ## render the hub
    _, meshHub, _ = render_hub(turbine_data, local=False)
    # meshHub = meshHub.merge(pv.Sphere(center=(0, 0, 0), radius=10))
    # meshHub.plot(show_edges=True)

    # ## render the tower
    _, meshTower, _ = render_Tower(turbine_data, local=False)
    # meshTower = meshTower.merge(pv.Sphere(center=(0, 0, 0), radius=10))
    # meshTower.plot(show_edges=True)

    # ## render the monopile
    _, meshMonopile, _ = render_monopile(turbine_data, local=False)
    # meshMonopile = meshMonopile.merge(pv.Sphere(center=(0, 0, 0), radius=10))
    # meshMonopile.plot(show_edges=True)

    # # ## render the nacelle
    _, meshNacelle, _ = render_nacelle(turbine_data, local=False)
    # meshNacelle = meshNacelle.merge(pv.Sphere(center=(0, 0, 0), radius=10))
    # meshNacelle.plot(show_edges=True)

    meshTurbine = meshBlade.merge(meshHub).merge(meshTower).merge(meshMonopile).merge(meshNacelle).merge(pv.Sphere(center=(0, 0, 0), radius=10))
    meshTurbine.plot(show_edges=True)



# call main if this script is run
if __name__ == "__main__":
    main()