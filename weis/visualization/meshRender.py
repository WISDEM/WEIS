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

        # First rotate hub to align with x-axis (90 degrees about y), 
        # with additional 180 deg rotation for downwind configuration
        points = rotation_transformation(points, [0, np.pi/2 + np.pi*int(downwindScalar==-1), 0])

        # Apply tilt angle rotation
        tiltAngle = turbineData['components']['nacelle']['drivetrain']['uptilt'] #* downwindScalar # in rads
        points = rotation_transformation(points, [0, tiltAngle, 0])

        # For downwind configuration, rotate 180 degrees about z-axis
        if downwindScalar == -1:
            points = rotation_transformation(points, [0, 0, np.pi])

        # Tanslation along the z-axis to the hub height
        zTranslation = turbineData['assembly']['hub_height']
        # Translation in the x-axis (account for downwind configuration)
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
        xTranslation = turbineData['components']['nacelle']['drivetrain']['overhang']  * 0.5

        # If downwind, rotate nacelle 180 degrees about z-axis
        if downwindScalar == -1:
            points = rotation_transformation(points, [0, 0, np.pi])

            xTranslation = -xTranslation  # Reverse the x-translation for downwind configuration

        points = translation_transformation(np.array(points), [xTranslation, 0, zTranslation])

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

def render_floatingPlatform(turbineData, local=True):

    mesh = floatingPlatformMesh(turbineData['components']['floating_platform'],
                                turbineData['components']['mooring']) # this function unlike other components returns a mesh object

    if local:
        mesh_state = to_mesh_state(mesh)
    else:
        # Floating platform does not have any orientation or tilt or translations
        mesh_state = to_mesh_state(mesh)

    extremes = extractExtremes(mesh.points)

    return mesh_state, mesh, extremes


def render_turbine(turbineData, components):

    mesh_state_all = []
    extremes_all = []

    for idx, component in enumerate(components):
        if component == 'blade':
            mesh_state, mesh, extreme = render_blade(turbineData, local=False)
        elif component == 'hub':
            mesh_state, mesh, extreme = render_hub(turbineData, local=False)
        elif component == 'tower':
            mesh_state, mesh, extreme = render_Tower(turbineData, local=False)
        elif component == 'nacelle':
            mesh_state, mesh, extreme = render_nacelle(turbineData, local=False)
        elif component == 'monopile':
            mesh_state, mesh, extreme = render_monopile(turbineData, local=False)
        elif component == 'floating_platform':
            mesh_state, mesh, extreme = render_floatingPlatform(turbineData, local=False)
        else:
            raise ValueError(f"Component {component} not recognized")
        
        mesh_state_all.append(mesh_state)
        extremes_all.append(extreme)
        if idx == 0:
            mesh_all = mesh
        else:
            mesh_all = mesh_all.merge(mesh)

    return mesh_state_all, mesh_all, extremes_all



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

# def hubMesh(hubData):

#     # Clear previous arrays
#     x = np.array([])
#     y = np.array([])
#     z = np.array([]) 
    
#     dia = hubData['diameter']
#     h = 0.5 * dia
#     '''
#     let the height of the hub be 0.5 of the diameter
#     The radius of the hub along the height varies from diameter to zero at the top and follows a parabolic function
#     '''
#     numPoints = 100
#     theta = np.linspace(0, 2*np.pi, numPoints)
#     z_dists = np.linspace(0, dia/2, 20)

#     for z_dist in z_dists:
#         r = dia/2 * np.sqrt(1 - (2*z_dist/dia)**2)
#         x = np.append(x, r * np.cos(theta))
#         y = np.append(y, r * np.sin(theta))
#         z = np.append(z, np.full_like(theta, z_dist))

#     points = (x, y, z)

#     return points

def hubMesh(hubData):
    # Clear previous arrays
    x = np.array([])
    y = np.array([])
    z = np.array([]) 
    
    dia = 1.25 * hubData['diameter']  # Increased diameter multiplier
    total_length = dia #* 1.2  # Increased total length for better coverage
    offset = -0.2 * dia  # Offset to move the hub backwards
    
    '''
    Hub consists of a parabolic section that:
    1. Starts slightly behind z=0 to better cover blade roots
    2. Has maximum radius at z=0.3*length 
    3. Tapers to a point at the end
    '''
    
    numPoints = 100
    theta = np.linspace(0, 2*np.pi, numPoints)
    z_points = np.linspace(offset, total_length + offset, 30)
    
    for z_dist in z_points:
        # Modified parabolic profile 
        normalized_z = (z_dist - offset)/(total_length)
        # Profile peaks at z=0.3*length and tapers at both ends
        r = dia/2 * (1 - ((normalized_z - 0.3)/0.7)**2)
        r = max(0, r)  # Ensure radius doesn't go negative
        
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
    
    # Add an offset for better visual alignment
    xOffset = -0.2  # Shift nacelle back by to better align with hub
    
    # if the nacelle dict has length, width and height, we can create a box
    if 'length' in nacelleData['drivetrain'].keys():
        length = nacelleData['drivetrain']['length']
        width = nacelleData['drivetrain']['width']
        height = nacelleData['drivetrain']['height']
        x = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]).T + xOffset * length
        y = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]).T
        z = np.array([-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2]).T
    else:
        # if not, we use the overhang, distance from tower top to hub, and hub diameter to create the box such that
        # height = 1.5 * distance from tower top to hub
        # length = 1 * overhang
        # width = 1 * hub diameter
        height = 1.5 * nacelleData['drivetrain']['distance_tt_hub']
        length = 1 * nacelleData['drivetrain']['overhang']
        width = 1 * hubData['diameter']
        x = np.array([length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]).T + xOffset * length
        y = np.array([width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]).T
        z = np.array([-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2]).T
    
    points = (x, y, z)

    return points

def floatingPlatformMesh(semisubData, mooringData=None):

    # convert joints from array to dictionary
    semisubData['joints'] = {joint['name']: joint for joint in semisubData['joints']}

    # creating a few sub functions
    def cylind2cart(p):
        # r, theta, z
        x = np.array([0, 0, 0])
        x[0] = p[0] * np.cos(p[1])
        x[1] = p[0] * np.sin(p[1])
        x[2] = p[2]
        return x

    def getJointPoint(jointName):

        joint = semisubData['joints'][jointName]

        # search if joint has cylindrical key
        if 'cylindrical' in joint.keys():
            if joint['cylindrical']:
                return cylind2cart(joint['location'])
            else:
                return joint['location']
        else:
            return joint['location']

    # def meshCone(joint1, joint2, grid, value):
    #     grid = np.interp(grid, [0, 1], [joint1, joint2])


    # looping through the members
    for idx, member in enumerate(semisubData['members']):
        # fetch the joints
        joint1 = np.array(getJointPoint(member['joint1']))
        joint2 = np.array(getJointPoint(member['joint2']))

        # direction vector normalized
        direction = (joint2 - joint1) / np.linalg.norm(joint2 - joint1)


        if member['outer_shape']['shape'] == 'circular':
            # create a cylinder, if last and first diameters are the same, then use pv.Cylinder
            # else use towerMesh to create a cylinder
            if member['outer_shape']['outer_diameter']['values'][0] == member['outer_shape']['outer_diameter']['values'][-1]:
                center = joint1 + 0.5 * (joint2 - joint1)  # Calculate midpoint
                memberMesh = pv.Cylinder(center=center,
                                        direction=direction, 
                                        height=np.linalg.norm(joint2 - joint1), 
                                        radius=member['outer_shape']['outer_diameter']['values'][0]/2)
            else:
                # Create a tapered cylinder using towerMesh-like approach
                towerData = {
                    'outer_shape_bem': {
                        'outer_diameter': {
                            'grid': member['outer_shape']['outer_diameter']['grid'],
                            'values': member['outer_shape']['outer_diameter']['values']
                        },
                        'reference_axis': {
                            'z': {
                                'grid': member['outer_shape']['outer_diameter']['grid'],
                                'values': np.linspace(0, np.linalg.norm(joint2 - joint1), len(member['outer_shape']['outer_diameter']['grid']))
                            }
                        }
                    }
                }
                
                x, y, z = towerMesh(towerData)
                
                # Rotate and translate the points to align with member direction
                points = np.vstack((x, y, z))
                
                # Calculate rotation matrix from [0,0,1] to direction vector
                v = np.array([0, 0, 1])
                rot_axis = np.cross(v, direction)
                rot_angle = np.arccos(np.dot(v, direction))
                
                if np.linalg.norm(rot_axis) > 0:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)
                    rotMatrix = pv.transformations.axis_angle_rotation(rot_axis, np.degrees(rot_angle))
                    points = np.dot(rotMatrix[:3, :3], points)
                
                # Translate to joint1
                points = points + joint1.reshape(3, 1)
                
                memberMesh = pv.PolyData(points.T)
                memberMesh = memberMesh.delaunay_3d()

        elif member['outer_shape']['shape'] == 'rectangular':
            '''
            Definition of Rectangular member:
            shape: rectangular
                side_length_a:
                    grid: [0.0, 1.0]
                    values: [12.5, 12.5]
                side_length_b:
                    grid: [0.0, 1.0]
                    values: [7.0, 7.0]
            '''
            # For rectangular members, create a box using the side lengths
            # Get side lengths at grid points
            side_a = member['outer_shape']['side_length_a']['values']
            side_b = member['outer_shape']['side_length_b']['values']

            # Calculate the member direction and perpendicular vectors
            direction = (joint2 - joint1) / np.linalg.norm(joint2 - joint1)
            perpendicular1 = np.array([-direction[1], direction[0], 0])
            if np.linalg.norm(perpendicular1) < 1e-10:
                perpendicular1 = np.array([1, 0, 0])
            perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
            perpendicular2 = np.cross(direction, perpendicular1)
            
            # Create corners based on joint points
            corners = np.array([
                # Bottom face
                joint1 + (-side_a[0]/2 * perpendicular1) + (-side_b[0]/2 * perpendicular2),
                joint1 + (side_a[0]/2 * perpendicular1) + (-side_b[0]/2 * perpendicular2),
                joint1 + (side_a[0]/2 * perpendicular1) + (side_b[0]/2 * perpendicular2),
                joint1 + (-side_a[0]/2 * perpendicular1) + (side_b[0]/2 * perpendicular2),
                # Top face 
                joint2 + (-side_a[1]/2 * perpendicular1) + (-side_b[1]/2 * perpendicular2),
                joint2 + (side_a[1]/2 * perpendicular1) + (-side_b[1]/2 * perpendicular2),
                joint2 + (side_a[1]/2 * perpendicular1) + (side_b[1]/2 * perpendicular2),
                joint2 + (-side_a[1]/2 * perpendicular1) + (side_b[1]/2 * perpendicular2)
            ])

            # Create mesh from vertices
            faces = np.array([
                [4, 0, 1, 2, 3],  # bottom
                [4, 4, 5, 6, 7],  # top
                [4, 0, 1, 5, 4],  # front
                [4, 1, 2, 6, 5],  # right
                [4, 2, 3, 7, 6],  # back
                [4, 3, 0, 4, 7]   # left
            ])
            memberMesh = pv.PolyData(corners, faces)




        else:
            # warning that the shape is not recognized
            print(f"Shape {member['outer_shape']['shape']} not recognized")
            

        # if the member has a key named 'axial_joints', then calculate its location in space and add it to the list of joints for future reference
        if 'axial_joints' in member.keys():
            for axial_joint in member['axial_joints']:
                axial_joint_location = joint1 + axial_joint['grid'] * (joint2 - joint1)
                semisubData['joints'][axial_joint['name']] = {'name': axial_joint['name'],'location': axial_joint_location, 'cylindrical': False}


        if idx == 0:
            floatingMesh = memberMesh
        else:
            floatingMesh = floatingMesh.merge(memberMesh)

        # floatingMesh.plot(show_edges=True)


    # if the semisub has a mooring system, then we need to add the mooring lines
    if mooringData:
        
        # convert nodes from array to dictionary
        mooringData['nodes'] = {node['name']: node for node in mooringData['nodes']}

        # Looping over the mooring lines
        for line in mooringData['lines']:
            # fetch the nodes
            np.array(getJointPoint(member['joint1']))
            node1 = np.array(getJointPoint(mooringData['nodes'][line['node1']]['joint']))
            node2 = np.array(getJointPoint(mooringData['nodes'][line['node2']]['joint']))

            # check if the distance between the nodes is equal to the unstretched length of the mooring line
            if np.linalg.norm(node2 - node1) == line['unstretched_length']:
                mooringLineMesh = pv.Line(pointa=node1, pointb=node2)
            else:
                # Create catenary mooring line using parametric equations
                num_points = 50
                t = np.linspace(0, 1, num_points)
                # Calculate catenary parameter (a) based on endpoints and length
                dx = node2[0] - node1[0]
                dy = node2[1] - node1[1]
                dz = node2[2] - node1[2]
                L = line['unstretched_length']
                a = np.sqrt(L**2 - dz**2) / 2  # Approximate catenary parameter
                
                # Generate points along catenary
                x = node1[0] + dx * t
                y = node1[1] + dy * t
                # Changed the sign before 'a' to make the catenary curve downward
                z = node1[2] + dz * t + a * (np.cosh(t - 0.5) - np.cosh(-0.5))
                
                points = np.column_stack((x, y, z))
                mooringLineMesh = pv.lines_from_points(points)

            floatingMesh = floatingMesh.merge(mooringLineMesh)

    return floatingMesh

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
    with open('../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_VolturnUS-S.yaml') as file:
    # with open('../../examples/06_IEA-15-240-RWT/IEA-15-240-RWT_Monopile.yaml') as file:
        turbine_data = yaml.load(file, Loader=yaml.FullLoader)

    # render the turbine components
    # mesh_state, meshTurbine, extremes_all = render_turbine(turbine_data, ['blade', 'hub', 'tower', 'nacelle', 'monopile'])
    mesh_state, meshTurbine, extremes_all = render_turbine(turbine_data, ['blade', 'hub', 'tower', 'nacelle','floating_platform'])
    # mesh_state, meshTurbine, extremes_all = render_turbine(turbine_data, ['floating_platform'])
    meshTurbine.plot(show_edges=False)


# call main if this script is run
if __name__ == "__main__":
    main()