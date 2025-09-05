
import copy, numpy as np
import weis.inputs as inp
from math import degrees, radians, pi, sin, cos
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.inputs.validation import get_geometry_schema, write_geometry_yaml, load_geometry_yaml
from wisdem.inputs import validate_without_defaults, write_yaml
from ROSCO_toolbox.utilities import read_DISCON
from weis.aeroelasticse.FileTools import remove_numpy
import os

#=========================================================================================================================
# OTHER FUNCTIONS

def compute_relThk(x,y):

  if not x[0]:
    x = 1 - np.array(x)  # hacky way to get airfoils nice
  
  LE_index = np.argmin(x)
  x_upper = np.flip(x[0:LE_index+1]) # python skips the last item
  y_upper = np.flip(y[0:LE_index+1]) # python skips the last item
  
  x_lower = x[LE_index:-1] # python skips the last item
  y_lower = y[LE_index:-1] # python skips the last item

  thickness = [0 for a in range(len(x_upper))] # initialize the array

  # compute camber and thickness
  for z in range(len(x_upper)):
    yu = np.interp(x_upper[z],x_upper,y_upper)
    yl = np.interp(x_upper[z],x_lower,y_lower)
    camber = (yu + yl)/2
    thickness[z] = yu-camber

  max_thk = 2*np.round(max(abs(np.array(thickness))),2) # thickness is symmetric about the camber line
  return max_thk

def point_in_cylinder(point, cylinder_start, cylinder_end, cylinder_radius):
    # Extract coordinates of the point and cylinder endpoints
    x, y, z = point
    x1, y1, z1 = cylinder_start
    x2, y2, z2 = cylinder_end

    # Calculate the vector from the start point to the end point of the cylinder
    v = (x2 - x1, y2 - y1, z2 - z1)

    # Calculate the vector from the start point to the given point
    w = (x - x1, y - y1, z - z1)

    # Calculate the dot product of v and w
    dot_product = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

    # Calculate the squared length of v
    length_squared = v[0] ** 2 + v[1] ** 2 + v[2] ** 2

    # Calculate the parameter along the line of the closest point to the given point
    # t is also axial joint, return this?
    t = dot_product / length_squared

    # t is projection of the point onto the line, if it's within (0,1) it's within the cylinder
    if 0 <= t <= 1:
      # Find the closest point on the line to the given point
      closest_point = (x1 + t * v[0], y1 + t * v[1], z1 + t * v[2])

      # Calculate the distance between the closest point and the given point
      distance = np.sqrt((x - closest_point[0]) ** 2 + (y - closest_point[1]) ** 2 + (z - closest_point[2]) ** 2)

      # Check if the distance is within the cylinder radius
      if distance <= cylinder_radius:
          # Check if the closest point is within the cylinder's height range
          if min(z1, z2) <= closest_point[2] <= max(z1, z2):
              return True, t

    return False, t

# ===============================================================================================================
# Constants
this_dir = os.path.dirname(__file__)

# ===============================================================================================================
# Inputs

# OpenFAST
fast = InputReader_OpenFAST()
fast.FAST_InputFile = 'MHK_RM1_Floating.fst'   # FAST input file (ext=.fst)
#fast.FAST_InputFile = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
fast.FAST_directory = '/Users/dzalkind/Projects/FloatingRM1_Controls/OpenFAST'
fast.execute()
print('successfully imported fast.fst_vt')

# Read DISCON infiles
fast.fst_vt['DISCON_in'] = read_DISCON(fast.fst_vt['ServoDyn']['DLL_InFile'])

# WEIS
finput        = os.path.join(this_dir,'IEA-15-floating_blank.yaml')   # blank input here
merged_schema = get_geometry_schema()
weis_obj      = validate_without_defaults(finput, merged_schema)
print('successfully imported blank weis dictionary')

# ASSUMED Inputs

assumed_model_file = os.path.join(this_dir,'IEA-15-floating_wTMDs.yaml')
assumed_model = load_geometry_yaml(assumed_model_file)

# Custom changes
iea_3_file = os.path.join(this_dir,'../05_IEA-3.4-130-RWT/IEA-3p4-130-RWT.yaml')
iea_3 = load_geometry_yaml(iea_3_file)

# FOR VERIFICATION ONLY
#weis_complete = inp.load_geometry_yaml('/home/nmendoza/Projects/Ct-Opt/WEIS/examples/06_IEA-15-240-RWT/IEA-15-floating_wTMDs.yaml')

# =======================================================================================================================
# Conversion from OpenFAST to WEIS (by component)

# Environment
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the environment to WEIS geometry schema and dictionary .............', end="", flush=True)

weis_obj['environment']['air_density']          = fast.fst_vt['Fst']['AirDens']
weis_obj['environment']['air_dyn_viscosity']    = merged_schema['properties']['environment']['properties']['air_dyn_viscosity']['default']
weis_obj['environment']['speed_sound']          = fast.fst_vt['Fst']['SpdSound']
weis_obj['environment']['shear_exp']            = fast.fst_vt['InflowWind']['PLexp']

# Optional
weis_obj['environment']['gravity']              = fast.fst_vt['Fst']['Gravity']
weis_obj['environment']['weib_shape_parameter'] = merged_schema['properties']['environment']['properties']['weib_shape_parameter']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['environment']['water_density']        = fast.fst_vt['Fst']['WtrDens']
weis_obj['environment']['water_dyn_viscosity']  = merged_schema['properties']['environment']['properties']['water_dyn_viscosity']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['environment']['water_depth']          = fast.fst_vt['Fst']['WtrDpth']
weis_obj['environment']['soil_shear_modulus']   = merged_schema['properties']['environment']['properties']['soil_shear_modulus']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['environment']['soil_poisson']         = merged_schema['properties']['environment']['properties']['soil_poisson']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['environment']['air_pressure']         = fast.fst_vt['Fst']['Patm']
weis_obj['environment']['air_vapor_pressure']   = fast.fst_vt['Fst']['Pvap']

print('Done')


# Assembly
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the assembly properties to WEIS geometry schema and dictionary .............', end="", flush=True)

weis_obj['assembly']['turbine_class']     = merged_schema['properties']['assembly']['properties']['turbine_class']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['assembly']['turbulence_class']  = merged_schema['properties']['assembly']['properties']['turbulence_class']['default'] # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT
weis_obj['assembly']['rated_power']       = fast.fst_vt['DISCON_in']['VS_RtPwr']
weis_obj['assembly']['lifetime']          = merged_schema['properties']['assembly']['properties']['lifetime']['default']  # NOT AVAILABLE IN OPENFAST, WEIS DEFAULT

if fast.fst_vt['ElastoDyn']['GBRatio'] == 1:
  weis_obj['assembly']['drivetrain']      = 'direct_drive'
else:
  weis_obj['assembly']['drivetrain']      = 'geared'

if fast.fst_vt['ElastoDyn']['OverHang'] < 0:
  weis_obj['assembly']['rotor_orientation'] = 'upwind'
else:
  weis_obj['assembly']['rotor_orientation'] = 'downwind'

weis_obj['assembly']['number_of_blades']  = fast.fst_vt['ElastoDyn']['NumBl']
weis_obj['assembly']['rotor_diameter']    = 2*fast.fst_vt['ElastoDyn']['TipRad']
weis_obj['assembly']['hub_height']        = fast.fst_vt['ElastoDyn']['TowerHt'] + fast.fst_vt['ElastoDyn']['Twr2Shft']+abs(fast.fst_vt['ElastoDyn']['OverHang'])*sin(radians(abs(fast.fst_vt['ElastoDyn']['ShftTilt'])))
weis_obj['assembly']['marine_hydro']      = bool(fast.fst_vt['Fst']['MHK'])

print('Done')


# Airfoils
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the airfoils to WEIS geometry schema and dictionary .............', end="", flush=True)
numAF                = range(fast.fst_vt['AeroDyn15']['NumAFfiles'])
AF_obj               = weis_obj['airfoils'][0] # AF_obj is now a pointer to the first index of the dictionary
weis_obj['airfoils'] = [copy.deepcopy(AF_obj) for x in range(len(numAF))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
airfoil_names        = [os.path.split(afn)[1].split('.dat')[0] for afn in fast.fst_vt['AeroDyn15']['AFNames']]  # assumes .dat extension!

for i in numAF:
    #print('i = ',i)
    # coordinates
    weis_obj['airfoils'][i]['coordinates']['x']           = fast.fst_vt['AeroDyn15']['af_coord'][i]['x'].tolist()
    weis_obj['airfoils'][i]['coordinates']['y']           = fast.fst_vt['AeroDyn15']['af_coord'][i]['y'].tolist()
    
    x = weis_obj['airfoils'][i]['coordinates']['x']
    if not x[0]:
      x = 1 - np.array(x)
      x = x.tolist()
      weis_obj['airfoils'][i]['coordinates']['x'] = x
    
    # properties
    weis_obj['airfoils'][i]['name']                       = airfoil_names[i]
    weis_obj['airfoils'][i]['aerodynamic_center']         = float(fast.fst_vt['AeroDyn15']['ac'][i])
    weis_obj['airfoils'][i]['relative_thickness']         = float(compute_relThk(weis_obj['airfoils'][i]['coordinates']['x'],weis_obj['airfoils'][i]['coordinates']['y']))
    # polars
    AoA = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Alpha']
    weis_obj['airfoils'][i]['polars'][0]['c_l']['grid']   = [A * pi/180 for A in AoA]
    weis_obj['airfoils'][i]['polars'][0]['c_l']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cl']
    weis_obj['airfoils'][i]['polars'][0]['c_d']['grid']   = [A * pi/180 for A in AoA]
    weis_obj['airfoils'][i]['polars'][0]['c_d']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cd']
    weis_obj['airfoils'][i]['polars'][0]['c_m']['grid']   = [A * pi/180 for A in AoA]
    if fast.fst_vt['AeroDyn15']['InCol_Cm']:
      weis_obj['airfoils'][i]['polars'][0]['c_m']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cm']
    else:
      weis_obj['airfoils'][i]['polars'][0]['c_m']['values'] = np.zeros_like(AoA).tolist()

    # Reynolds number
    weis_obj['airfoils'][i]['polars'][0]['re'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Re']

print('Done')


# Blades
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the blade planform properties to WEIS geometry schema and dictionary .........', end="", flush=True)

# Blade nodes = nondimensionalize
blade_length = fast.fst_vt['AeroDynBlade']['BlSpn'][-1]
BlSpn = fast.fst_vt['AeroDynBlade']['BlSpn']
blade_fraction = [L / blade_length for L in BlSpn]

# Airfoil Positions
weis_obj['components']['blade']['outer_shape_bem']['airfoil_position']['grid']      = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['airfoil_position']['labels']    = [airfoil_names[int(id)-1] for id in fast.fst_vt['AeroDynBlade']['BlAFID']]
# Chord
weis_obj['components']['blade']['outer_shape_bem']['chord']['grid']                 = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['chord']['values']               = fast.fst_vt['AeroDynBlade']['BlChord']
# Twist
weis_obj['components']['blade']['outer_shape_bem']['twist']['grid']                 = blade_fraction
twist = fast.fst_vt['AeroDynBlade']['BlTwist'] # convert to radians
weis_obj['components']['blade']['outer_shape_bem']['twist']['values']               = [T * pi/180 for T in twist]
# Pitch Axis
weis_obj['components']['blade']['outer_shape_bem']['pitch_axis']['grid']            = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['pitch_axis']['values']          = np.interp(blade_fraction,fast.fst_vt['ElastoDynBlade']['BlFract'],fast.fst_vt['ElastoDynBlade']['PitchAxis'])
# Reference Axis (normal prebend is negative in weis)
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid']   = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['x']['values'] = [-x for x in fast.fst_vt['AeroDynBlade']['BlCrvAC']]
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid']   = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['y']['values'] = fast.fst_vt['AeroDynBlade']['BlSwpAC']
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid']   = blade_fraction
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['z']['values'] = fast.fst_vt['AeroDynBlade']['BlSpn']

print('Done')


# Hub
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the hub properties to WEIS geometry schema and dictionary .........', end="", flush=True)

# Required
weis_obj['components']['hub']['diameter']         = 2*fast.fst_vt['ElastoDyn']['HubRad']
weis_obj['components']['hub']['cone_angle']       = radians(fast.fst_vt['ElastoDyn']['PreCone(1)']) # assumes all three blades are at the same cone angle (very reasonable and practical assumption)
weis_obj['components']['hub']['drag_coefficient'] = 0.0 # NOT AVAILABLE IN OPENFAST

# Optional
#weis_obj['components']['hub']['clearance_hub_spinner'] = 
#weis_obj['components']['hub']['flange_ID2OD'] = 
#weis_obj['components']['hub']['flange_OD2hub_D'] = 
#weis_obj['components']['hub']['flange_t2shell_t'] = 
#weis_obj['components']['hub']['hub_blade_spacing_margin'] = 
#weis_obj['components']['hub']['hub_material'] = 
#weis_obj['components']['hub']['hub_stress_concentration'] = 
#weis_obj['components']['hub']['n_front_brackets'] = 
#weis_obj['components']['hub']['n_rear_brackets'] = 
#weis_obj['components']['hub']['pitch_system_scaling_factor'] = 
#weis_obj['components']['hub']['spin_hole_incr'] = 
#weis_obj['components']['hub']['spinner_gust_ws'] = 
#weis_obj['components']['hub']['spinner_material'] = 

print('Done')


# Nacelle
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the drivetrain and generator properties to WEIS geometry schema and dictionary .........', end="", flush=True)

# Drivetrain
weis_obj['components']['nacelle']['drivetrain']['uptilt']                                  = radians(-1*fast.fst_vt['ElastoDyn']['ShftTilt'])
weis_obj['components']['nacelle']['drivetrain']['distance_tt_hub']                         = fast.fst_vt['ElastoDyn']['Twr2Shft']+abs(fast.fst_vt['ElastoDyn']['OverHang'])*sin(radians(abs(fast.fst_vt['ElastoDyn']['ShftTilt'])))
weis_obj['components']['nacelle']['drivetrain']['distance_hub_mb']                         = 0
weis_obj['components']['nacelle']['drivetrain']['distance_mb_mb']                          = 0
weis_obj['components']['nacelle']['drivetrain']['overhang']                                = abs(fast.fst_vt['ElastoDyn']['OverHang'])
weis_obj['components']['nacelle']['drivetrain']['generator_length']                        = 0
weis_obj['components']['nacelle']['drivetrain']['generator_radius_user']                   = 0
weis_obj['components']['nacelle']['drivetrain']['generator_mass_user']                     = 0
weis_obj['components']['nacelle']['drivetrain']['generator_rpm_efficiency_user']['grid']   = []
weis_obj['components']['nacelle']['drivetrain']['generator_rpm_efficiency_user']['values'] = []
weis_obj['components']['nacelle']['drivetrain']['gear_ratio']                              = fast.fst_vt['ElastoDyn']['GBRatio']
weis_obj['components']['nacelle']['drivetrain']['gearbox_length_user']                     = 0
weis_obj['components']['nacelle']['drivetrain']['gearbox_radius_user']                     = 0
weis_obj['components']['nacelle']['drivetrain']['gearbox_mass_user']                       = 0
weis_obj['components']['nacelle']['drivetrain']['gearbox_efficiency']                      = fast.fst_vt['ElastoDyn']['GBoxEff'] / 100
weis_obj['components']['nacelle']['drivetrain']['damping_ratio']                           = 0 #fast.fst_vt['ElastoDyn']['DTTorDmp']???  Not quite.  I'm not sure we can get it with the OpenFAST info
weis_obj['components']['nacelle']['drivetrain']['lss_diameter']                            = []
weis_obj['components']['nacelle']['drivetrain']['lss_wall_thickness']                      = []
weis_obj['components']['nacelle']['drivetrain']['lss_material']                            = ''
weis_obj['components']['nacelle']['drivetrain']['hss_length']                              = 0
weis_obj['components']['nacelle']['drivetrain']['hss_diameter']                            = []
weis_obj['components']['nacelle']['drivetrain']['hss_wall_thickness']                      = []
weis_obj['components']['nacelle']['drivetrain']['hss_material']                            = ''
weis_obj['components']['nacelle']['drivetrain']['nose_diameter']                           = []
weis_obj['components']['nacelle']['drivetrain']['nose_wall_thickness']                     = []
weis_obj['components']['nacelle']['drivetrain']['bedplate_wall_thickness']['grid']         = []
weis_obj['components']['nacelle']['drivetrain']['bedplate_wall_thickness']['values']       = []
weis_obj['components']['nacelle']['drivetrain']['bedplate_flange_width']                   = 0
weis_obj['components']['nacelle']['drivetrain']['bedplate_flange_thickness']               = 0
weis_obj['components']['nacelle']['drivetrain']['bedplate_web_thickness']                  = 0
weis_obj['components']['nacelle']['drivetrain']['bedplate_material']                       = ''
#weis_obj['components']['nacelle']['drivetrain']['brake_mass_user']                         = 0
#weis_obj['components']['nacelle']['drivetrain']['hvac_mass_coefficient']                   = 0
#weis_obj['components']['nacelle']['drivetrain']['converter_mass_user']                     = 0
#weis_obj['components']['nacelle']['drivetrain']['transformer_mass_user']                   = 0
weis_obj['components']['nacelle']['drivetrain']['mb1Type']                                 = ''
weis_obj['components']['nacelle']['drivetrain']['mb2Type']                                 = ''
weis_obj['components']['nacelle']['drivetrain']['uptower']                                 = True
weis_obj['components']['nacelle']['drivetrain']['gear_configuration']                      = ''
weis_obj['components']['nacelle']['drivetrain']['planet_numbers']                          = []

# Generator
weis_obj['components']['nacelle']['generator']['mass_coefficient'] = 0
weis_obj['components']['nacelle']['generator']['generator_type']   = ''
weis_obj['components']['nacelle']['generator']['B_r']              = 0
weis_obj['components']['nacelle']['generator']['P_Fe0e']           = 0
weis_obj['components']['nacelle']['generator']['P_Fe0h']           = 0
weis_obj['components']['nacelle']['generator']['S_N']              = 0
weis_obj['components']['nacelle']['generator']['S_Nmax']           = 0
weis_obj['components']['nacelle']['generator']['alpha_p']          = 0
weis_obj['components']['nacelle']['generator']['b_r_tau_r']        = 0
weis_obj['components']['nacelle']['generator']['b_ro']             = 0
weis_obj['components']['nacelle']['generator']['b_s_tau_s']        = 0
weis_obj['components']['nacelle']['generator']['b_so']             = 0
weis_obj['components']['nacelle']['generator']['cofi']             = 0
weis_obj['components']['nacelle']['generator']['freq']             = 1
weis_obj['components']['nacelle']['generator']['h_i']              = 0
weis_obj['components']['nacelle']['generator']['h_sy0']            = 0
weis_obj['components']['nacelle']['generator']['h_w']              = 0
weis_obj['components']['nacelle']['generator']['k_fes']            = 0
weis_obj['components']['nacelle']['generator']['k_fillr']          = 0
weis_obj['components']['nacelle']['generator']['k_fills']          = 0
weis_obj['components']['nacelle']['generator']['k_s']              = 0
weis_obj['components']['nacelle']['generator']['m']                = 0
weis_obj['components']['nacelle']['generator']['mu_0']             = 0
weis_obj['components']['nacelle']['generator']['mu_r']             = 0
weis_obj['components']['nacelle']['generator']['p']                = 0
weis_obj['components']['nacelle']['generator']['phi']              = 0
weis_obj['components']['nacelle']['generator']['q1']               = 0
weis_obj['components']['nacelle']['generator']['q3']               = 0
weis_obj['components']['nacelle']['generator']['ratio_mw2pp']      = 0
weis_obj['components']['nacelle']['generator']['resist_Cu']        = 0
weis_obj['components']['nacelle']['generator']['sigma']            = 0
weis_obj['components']['nacelle']['generator']['y_tau_p']          = 0
weis_obj['components']['nacelle']['generator']['y_tau_pr']         = 0
weis_obj['components']['nacelle']['generator']['I_0']              = 0
weis_obj['components']['nacelle']['generator']['d_r']              = 0
weis_obj['components']['nacelle']['generator']['h_m']              = 0
weis_obj['components']['nacelle']['generator']['h_0']              = 0
weis_obj['components']['nacelle']['generator']['h_s']              = 0
weis_obj['components']['nacelle']['generator']['len_s']            = 0
weis_obj['components']['nacelle']['generator']['n_r']              = 0
weis_obj['components']['nacelle']['generator']['rad_ag']           = 0
weis_obj['components']['nacelle']['generator']['t_wr']             = 0
weis_obj['components']['nacelle']['generator']['n_s']              = 0
weis_obj['components']['nacelle']['generator']['b_st']             = 0
weis_obj['components']['nacelle']['generator']['d_s']              = 0
weis_obj['components']['nacelle']['generator']['t_ws']             = 0
weis_obj['components']['nacelle']['generator']['rho_Copper']       = 0
weis_obj['components']['nacelle']['generator']['rho_Fe']           = 0
weis_obj['components']['nacelle']['generator']['rho_Fes']          = 0
weis_obj['components']['nacelle']['generator']['rho_PM']           = 0
#weis_obj['components']['nacelle']['generator']['C_Cu']             = 0 # OpenFAST doesn't have costs
#weis_obj['components']['nacelle']['generator']['C_Fe']             = 0 # OpenFAST doesn't have costs
#weis_obj['components']['nacelle']['generator']['C_Fes']            = 0 # OpenFAST doesn't have costs
#weis_obj['components']['nacelle']['generator']['C_PM']             = 0 # OpenFAST doesn't have costs

print('Done')


# Tower
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the tower properties to WEIS geometry schema and dictionary .........', end="", flush=True)

# Outer Geometry

# Tower nodes = nondimensionalize
num_tower_aero_nodes = fast.fst_vt['AeroDyn15']['NumTwrNds']
tower_height         = fast.fst_vt['AeroDyn15']['TwrElev'][-1]
TwrElev              = fast.fst_vt['AeroDyn15']['TwrElev']

weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['x']['grid']   = [H / tower_height for H in TwrElev] # non-dimensional
weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['x']['values'] = [0 for i in range(num_tower_aero_nodes)] # x is positive upwind to downwind
weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['y']['grid']   = [H / tower_height for H in TwrElev] # non-dimensional
weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['y']['values'] = [0 for i in range(num_tower_aero_nodes)] # y follows the right-hand-rule
weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']   = [H / tower_height for H in TwrElev] # non-dimensional
weis_obj['components']['tower']['outer_shape_bem']['reference_axis']['z']['values'] = TwrElev # z is positive upwards

weis_obj['components']['tower']['outer_shape_bem']['outer_diameter']['grid']      = [H / tower_height for H in TwrElev]
weis_obj['components']['tower']['outer_shape_bem']['outer_diameter']['values']    = fast.fst_vt['AeroDyn15']['TwrDiam']
weis_obj['components']['tower']['outer_shape_bem']['drag_coefficient']['grid']    = [H / tower_height for H in TwrElev]
weis_obj['components']['tower']['outer_shape_bem']['drag_coefficient']['values']  = fast.fst_vt['AeroDyn15']['TwrCd']

# Internal Structure
num_tower_struct_nodes = fast.fst_vt['ElastoDynTower']['NTwInpSt']
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['x']['grid']      = fast.fst_vt['ElastoDynTower']['HtFract'] # non-dimensional
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['x']['values']    = [0 for j in range(num_tower_struct_nodes)] # x is positive upwind to downwind
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['y']['grid']      = fast.fst_vt['ElastoDynTower']['HtFract'] # non-dimensional
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['y']['values']    = [0 for j in range(num_tower_struct_nodes)] # y follows the right-hand-rule
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['z']['grid']      = fast.fst_vt['ElastoDynTower']['HtFract'] # non-dimensional
# weis_obj['components']['tower']['internal_structure_2d_fem']['reference_axis']['z']['values']    = [T * tower_height for T in fast.fst_vt['ElastoDynTower']['HtFract']] # z is positive upwards

# Reference axes in blank input are linked, this could be problematic if an assumed geometry has a different reference axis

weis_obj['components']['tower']['internal_structure_2d_fem']['outfitting_factor']                = fast.fst_vt['ElastoDynTower']['AdjTwMa'] # tower mass scaling factor

weis_obj['components']['tower']['internal_structure_2d_fem']['layers'][0]['name']                = '' # not available at this time
weis_obj['components']['tower']['internal_structure_2d_fem']['layers'][0]['material']            = ''  # not available at this time
weis_obj['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['grid']   = fast.fst_vt['ElastoDynTower']['HtFract'] # non-dimensional
weis_obj['components']['tower']['internal_structure_2d_fem']['layers'][0]['thickness']['values'] = [] # not available at this time

# Elastic Properties (Multi-body)? - optional

print('Done')


# Monopile
# -----------------------------------------------------------------------------------------------------------------------
# Logic to determine if the system is fixed-bottom or floating
if fast.fst_vt['Fst']['CompSub'] == 1: # if there is no mooring, then its fixed-bottom
  print('Converting the monopile properties to WEIS geometry schema and dictionary .........', end="", flush=True)

  # Required

  #Non-dimensionalize the reference axis
  monopile_length = abs(fast.fst_vt['SubDyn']['JointZss'][0] - fast.fst_vt['SubDyn']['JointZss'][-1])
  temp = [m / monopile_length for m in fast.fst_vt['SubDyn']['JointZss']]
  monopile_joints = [n - (fast.fst_vt['SubDyn']['JointZss'][0]/monopile_length) for n in temp]

  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['x']['grid']   = monopile_joints
  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['x']['values'] = fast.fst_vt['SubDyn']['JointXss']
  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['y']['grid']   = monopile_joints
  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['y']['values'] = fast.fst_vt['SubDyn']['JointYss']
  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['z']['grid']   = monopile_joints
  weis_obj['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values'] = fast.fst_vt['SubDyn']['JointZss']
  weis_obj['components']['monopile']['outer_shape_bem']['outer_diameter']['grid']        = monopile_joints
  weis_obj['components']['monopile']['outer_shape_bem']['drag_coefficient']['grid']      = monopile_joints

  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['x']['grid']   = monopile_joints
  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['x']['values'] = fast.fst_vt['SubDyn']['JointXss']
  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['y']['grid']   = monopile_joints
  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['y']['values'] = fast.fst_vt['SubDyn']['JointYss']
  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['z']['grid']   = monopile_joints
  weis_obj['components']['monopile']['internal_structure_2d_fem']['reference_axis']['z']['values'] = fast.fst_vt['SubDyn']['JointZss']
  weis_obj['components']['monopile']['internal_structure_2d_fem']['outfitting_factor']             = 0
  weis_obj['components']['monopile']['internal_structure_2d_fem']['layers'][0]['name']             = '' # not available at this time
  weis_obj['components']['monopile']['internal_structure_2d_fem']['layers'][0]['material']         = '' # not available at this time
  weis_obj['components']['monopile']['internal_structure_2d_fem']['layers'][0]['thickness']['grid']= monopile_joints # there's only one layer

  # Other distributed properties
  weis_obj['components']['monopile']['outer_shape_bem']['outer_diameter']['values'] = [0 for j in range(fast.fst_vt['SubDyn']['NJoints'])]
  weis_obj['components']['monopile']['internal_structure_2d_fem']['layers'][0]['thickness']['values'] = [0 for j in range(fast.fst_vt['SubDyn']['NJoints'])]

  for j in range(fast.fst_vt['SubDyn']['NJoints']):
    try:
      idx = fast.fst_vt['SubDyn']['MJointID1'].index(fast.fst_vt['SubDyn']['JointID'][j])
      propset = fast.fst_vt['SubDyn']['MPropSetID1'][idx]
      idx2 = fast.fst_vt['SubDyn']['PropSetID1'].index(propset)
    except: # for the last one because there are always more joints than members
      idx = fast.fst_vt['SubDyn']['MJointID2'].index(fast.fst_vt['SubDyn']['JointID'][j])
      propset = fast.fst_vt['SubDyn']['MPropSetID2'][idx]
      idx2 = fast.fst_vt['SubDyn']['PropSetID1'].index(propset)
    
    weis_obj['components']['monopile']['outer_shape_bem']['outer_diameter']['values'][j] = fast.fst_vt['SubDyn']['XsecD'][idx2]
    weis_obj['components']['monopile']['internal_structure_2d_fem']['layers'][0]['thickness']['values'][j] = fast.fst_vt['SubDyn']['XsecT'][idx2]

  # Get the drag coefficient from HydroDyn - DO WE WANT AXIAL OR SIMPLE COEFFICIENTS???
  weis_obj['components']['monopile']['outer_shape_bem']['drag_coefficient']['values']    = [0 for j in range(fast.fst_vt['SubDyn']['NJoints'])]

  # Optional
  for cm in range(fast.fst_vt['SubDyn']['NCMass']): # iterate through concentrated masses aka point loads
    if fast.fst_vt['SubDyn']['CMJointID'] == fast.fst_vt['SubDyn']['RJointID']: # base joint
      weis_obj['components']['monopile']['gravity_foundation_mass'] = fast.fst_vt['SubDyn']['JMass'][cm]
    if fast.fst_vt['SubDyn']['CMJointID'] == fast.fst_vt['SubDyn']['IJointID']: # interface joint with the transition piece
      weis_obj['components']['monopile']['transition_piece_mass']   = fast.fst_vt['SubDyn']['JMass'][cm]
      #weis_obj['components']['monopile']['transition_piece_cost']   = 0 # COST NOT AVAILABLE IN OPENFAST!
  
  # weis_obj['components']['monopile']['elastic_properties_mb'] = 

  print('Done')


# Floating Platform
# -----------------------------------------------------------------------------------------------------------------------
# Logic to determine if the system is fixed-bottom or floating
if fast.fst_vt['Fst']['CompMooring'] > 0: # if there is mooring, then its floating so parameterize the floating hull/platform  
  print('Converting the floating platform properties to WEIS geometry schema and dictionary .........', end="", flush=True)

  # Required
  
  # Joints
  numJoints = fast.fst_vt['HydroDyn']['NJoints']  # total number of joints from HydroDyn
  J_obj     = weis_obj['components']['floating_platform']['joints'][0] # J_obj is now a pointer to the first index of the dictionary

  # First get all joints
  ptfm_joints = [copy.deepcopy(J_obj) for x in range(numJoints)]

  max_height = max(fast.fst_vt['HydroDyn']['Jointzi'])
  for j in range(fast.fst_vt['HydroDyn']['NJoints']):
    ptfm_joints[j]['name']        = f"joint_{fast.fst_vt['HydroDyn']['JointID'][j]}"
    ptfm_joints[j]['location']    = [fast.fst_vt['HydroDyn']['Jointxi'][j], fast.fst_vt['HydroDyn']['Jointyi'][j], fast.fst_vt['HydroDyn']['Jointzi'][j]]
    ptfm_joints[j]['cylindrical'] = False # are cylindrical coordinates used to describe the location of this joint?
    #ptfm_joints[j]['reactions']   = [] # joint DOFs
    # does the transition between tower and platform happen at this joint?
    if (abs(fast.fst_vt['HydroDyn']['Jointxi'][j]) < 0.001) and (abs(fast.fst_vt['HydroDyn']['Jointyi'][j]) < 0.001) and (abs(fast.fst_vt['HydroDyn']['Jointzi'][j]) == max_height):
      ptfm_joints[j]['transition']  = True
    else: 
      ptfm_joints[j]['transition']  = False

  j = j + 1
  anchor_count = 1
  fairlead_count = 1
  n_moor_joints = len(fast.fst_vt['MoorDyn']['Point_ID'])
  
  moor_joints = []

  # Add the joints (nodes) from MoorDyn
  for j in range(n_moor_joints): 
    moor_joint = copy.deepcopy(J_obj)
    if fast.fst_vt['MoorDyn']['Attachment'][j].lower() == 'fixed':
      moor_joint['name']        = 'anchor' + str(anchor_count)
      anchor_count = anchor_count + 1
    elif fast.fst_vt['MoorDyn']['Attachment'][j].lower() == 'vessel':
      moor_joint['name']        = 'fairlead' + str(fairlead_count)
      fairlead_count = fairlead_count + 1
    else:
      moor_joint['name']        = ''
      
    moor_joint['location']    = [fast.fst_vt['MoorDyn']['X'][j], fast.fst_vt['MoorDyn']['Y'][j], fast.fst_vt['MoorDyn']['Z'][j]]
    moor_joint['cylindrical'] = False # are cylindrical coordinates used to describe the location of this joint?
    #moor_joints[j]['reactions']   = [] # joint DOFs
    moor_joints.append(moor_joint)

  # Members
  numMembers = fast.fst_vt['HydroDyn']['NMembers']
  M_obj      = weis_obj['components']['floating_platform']['members'][0] # M_obj is now a pointer to the first index of the dictionary
  empty_axial_joint = M_obj['axial_joints'][0].copy()
  M_obj['axial_joints'] = []  # leave axial joints empty 

  weis_obj['components']['floating_platform']['members'] = [copy.deepcopy(M_obj) for x in range(numMembers)] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
  for m in range(numMembers):
    weis_obj['components']['floating_platform']['members'][m]['name']               = f"member_{int(fast.fst_vt['HydroDyn']['MemberID'][m])}"
    weis_obj['components']['floating_platform']['members'][m]['joint1']             = f"joint_{fast.fst_vt['HydroDyn']['MJointID1'][m]}"
    weis_obj['components']['floating_platform']['members'][m]['joint2']             = f"joint_{fast.fst_vt['HydroDyn']['MJointID2'][m]}"
    #weis_obj['components']['floating_platform']['members'][m]['Ca']                 = 0 # ONLY AVAILABLE FOR JOINTS
    #weis_obj['components']['floating_platform']['members'][m]['Cd']                 = 0 # ONLY AVAILABLE FOR JOINTS
    #weis_obj['components']['floating_platform']['members'][m]['Cp']                 = 0 # ONLY AVAILABLE FOR JOINTS

    idxProp = fast.fst_vt['HydroDyn']['PropSetID'].index(fast.fst_vt['HydroDyn']['MPropSetID1'][m]) # get which property set it is

    weis_obj['components']['floating_platform']['members'][m]['outer_shape']['shape]']                   = 'circular' # fixed for now because OpenFAST can't do polygonal members
    weis_obj['components']['floating_platform']['members'][m]['outer_shape']['outer_diameter']['grid']   = [0, 1]
    weis_obj['components']['floating_platform']['members'][m]['outer_shape']['outer_diameter']['values'] = [fast.fst_vt['HydroDyn']['PropD'][idxProp], fast.fst_vt['HydroDyn']['PropD'][idxProp]]

    weis_obj['components']['floating_platform']['members'][m]['internal_structure']['layers'][0]['name']                = str(fast.fst_vt['HydroDyn']['MPropSetID1'][m])
    weis_obj['components']['floating_platform']['members'][m]['internal_structure']['layers'][0]['material']            = '' # NOT AVAILABLE IN OPENFAST
    weis_obj['components']['floating_platform']['members'][m]['internal_structure']['layers'][0]['thickness']['grid']   = [0, 1]
    weis_obj['components']['floating_platform']['members'][m]['internal_structure']['layers'][0]['thickness']['values'] = [fast.fst_vt['HydroDyn']['PropThck'][idxProp], fast.fst_vt['HydroDyn']['PropThck'][idxProp]]
    
  # TODO: axial joints
  members = weis_obj['components']['floating_platform']['members']
  new_members = copy.deepcopy(members)
  ptfm_joint_names = [joint['name'] for joint in ptfm_joints]
  moor_joint_names = [joint['name'] for joint in moor_joints]
  all_joint_names = ptfm_joint_names + moor_joint_names

  # For each member (keep copy of new_members handy)
  for mem_i, new_mem_i in zip(members, new_members):
    # Find endpoints
    mem_i_joint_1 = ptfm_joints[ptfm_joint_names.index(mem_i['joint1'])]['location']
    mem_i_joint_2 = ptfm_joints[ptfm_joint_names.index(mem_i['joint2'])]['location']

    # Do these endpoints fall within or near the cylinder of another member
    for mem_j, new_mem_j in zip(members, new_members):
      if mem_j == mem_i:
        continue

      margin = 1e-2

      cyl_radius = (mem_j['outer_shape']['outer_diameter']['values'][0] + margin)/2   # assuming straight cols for now
      mem_j_joint_1 = ptfm_joints[ptfm_joint_names.index(mem_j['joint1'])]['location']
      mem_j_joint_2 = ptfm_joints[ptfm_joint_names.index(mem_j['joint2'])]['location']

      # Is the 1st joint of mem_i in mem_j?
      pic = point_in_cylinder(mem_i_joint_1, mem_j_joint_1, mem_j_joint_2, cyl_radius)
      if pic[0]:
        print(f"Member {mem_i['name']} joint1 ({mem_i['joint1']}) falls in member {mem_j['name']}")
        # Make axial joint
        ax_joint = empty_axial_joint.copy()
        ax_joint['name'] = f"{mem_i['name']}_connect_a"
        ax_joint['grid'] = pic[1]
        
        # Add to mem_j, member where joint was found
        new_mem_j['axial_joints'].append(ax_joint)
        
        # Rename joint of mem_i to axial joint name
        new_mem_i['joint1'] = ax_joint['name']  # reassign mem
        
        # Remove joint from list
        all_joint_names.remove(mem_i['joint1'])

      # Is the 2nd joint of mem_i in mem_j?
      pic = point_in_cylinder(mem_i_joint_2, mem_j_joint_1, mem_j_joint_2, cyl_radius)
      if pic[0]:
        print(f"Member {mem_i['name']} joint2 ({mem_i['joint2']}) falls in member {mem_j['name']}")
        # Make axial joint
        ax_joint = empty_axial_joint.copy()
        ax_joint['name'] = f"{mem_i['name']}_connect_b"
        ax_joint['grid'] = pic[1]
        
        # Add to mem_j, member where joint was found
        new_mem_j['axial_joints'].append(ax_joint)

        # Rename joint of mem_i to axial joint name
        new_mem_i['joint2'] = ax_joint['name']
        
        # Remove joint from list
        all_joint_names.remove(mem_i['joint2'])

  # Check if mooring points lie on members
  for mj in moor_joints:
    # Is mooring joint in a member?
    for mem_i, new_mem_i in zip(members, new_members):
      mem_i_joint_1 = ptfm_joints[ptfm_joint_names.index(mem_i['joint1'])]['location']
      mem_i_joint_2 = ptfm_joints[ptfm_joint_names.index(mem_i['joint2'])]['location']
      cyl_radius = (mem_i['outer_shape']['outer_diameter']['values'][0] + margin)/2   # assuming straight cols for now
      pic = point_in_cylinder(mj['location'], mem_i_joint_1, mem_i_joint_2, cyl_radius)
      if pic[0]:
        print(f"Mooring joint {mj['name']} falls in member {mem_i['name']}")
        ax_joint = empty_axial_joint.copy()
        ax_joint['name'] = mj['name']
        ax_joint['grid'] = pic[1]
        new_mem_i['axial_joints'].append(ax_joint)
        all_joint_names.remove(mj['name'])



  # # Remove duplicated axial joints?  Leave out for now. I think WISDEM will handle this
  # for mem_i in new_members:
  #   joint_locs = []
  #   joint_names = []
  #   for aj in mem_i['axial_joints']:
  #     if aj['grid'] not in joint_locs:
  #       joint_locs.append(aj['grid'])
  #       joint_names.append(aj['name'])
  #     else:  # duplicate, need to reassign member endpoint
  #       for mem_j in new_members:
  #         if mem_j['joint1'] == aj['name']:
  #           mem_j['joint1'] = joint_names[joint_locs.index(aj['grid'])]
  #           mem_i['axial_joints'].remove(aj)
  #           print('here')

  # Make joints without axial joints
  floating_joints = []
  for pj in ptfm_joints:
    if pj['name'] in all_joint_names:
      floating_joints.append(pj)

  for mj in moor_joints:
    if mj['name'] in all_joint_names:
      floating_joints.append(mj)

  # Apply floating_joints and new_members to floating WEIS definition
  weis_obj['components']['floating_platform']['joints'] = floating_joints
  weis_obj['components']['floating_platform']['members'] = new_members


  # Optional
  #weis_obj['components']['floating_platform']['rigid_bodies']['joint1']             = ''
  #weis_obj['components']['floating_platform']['rigid_bodies']['mass']               = 0
  #weis_obj['components']['floating_platform']['rigid_bodies']['cost']               = 0 # COSTS ARE NOT AVAILABLE IN OPENFAST!
  #weis_obj['components']['floating_platform']['rigid_bodies']['cm_offset']          = []
  #weis_obj['components']['floating_platform']['rigid_bodies']['moments_of_inertia'] = []

  #weis_obj['components']['floating_platform']['transition_piece_mass']              = 0
  #weis_obj['components']['floating_platform']['transition_piece_cost']              = 0 # COSTS ARE NOT AVAILABLE IN OPENFAST!

  print('Done')


# Mooring
# -----------------------------------------------------------------------------------------------------------------------
# Logic to determine if the system is fixed-bottom or floating
if fast.fst_vt['Fst']['CompMooring'] > 0: # if there is mooring, use it!
  print('Converting the mooring properties to WEIS geometry schema and dictionary .........', end="", flush=True)

  # Lines
  numL                = range(len(fast.fst_vt['MoorDyn']['Line_ID']))
  L_obj               = weis_obj['components']['mooring']['lines'][0] # L_obj is now a pointer to the first index of the dictionary
  weis_obj['components']['mooring']['lines'] = [copy.deepcopy(L_obj) for x in range(len(numL))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
  for l in numL:
    weis_obj['components']['mooring']['lines'][l]['name']               = fast.fst_vt['MoorDyn']['Line_ID'][l]
    weis_obj['components']['mooring']['lines'][l]['line_type']          = fast.fst_vt['MoorDyn']['LineType'][l]
    weis_obj['components']['mooring']['lines'][l]['unstretched_length'] = fast.fst_vt['MoorDyn']['UnstrLen'][l]
    weis_obj['components']['mooring']['lines'][l]['node1']              = fast.fst_vt['MoorDyn']['AttachA'][l]
    weis_obj['components']['mooring']['lines'][l]['node2']              = fast.fst_vt['MoorDyn']['AttachB'][l]

  # Line Types
  numLT                = range(len(fast.fst_vt['MoorDyn']['Name']))
  LT_obj               = weis_obj['components']['mooring']['line_types'][0] # LT_obj is now a pointer to the first index of the dictionary
  weis_obj['components']['mooring']['line_types'] = [copy.deepcopy(LT_obj) for x in range(len(numLT))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
  for lt in numLT:
    weis_obj['components']['mooring']['line_types'][lt]['name']     = fast.fst_vt['MoorDyn']['Name'][lt]
    weis_obj['components']['mooring']['line_types'][lt]['diameter'] = fast.fst_vt['MoorDyn']['Diam'][lt]
    weis_obj['components']['mooring']['line_types'][lt]['type']     = '' # MATERIAL NOT AVAILABLE IN OPENFAST: must be one of lower(chain, chain_stud, nylon, polyester, polypropylene, wire_fiber, fiber, wire, wire_wire, iwrc, custom]

  # Nodes (same as Points)
  numN                = range(len(fast.fst_vt['MoorDyn']['Point_ID']))
  N_obj               = weis_obj['components']['mooring']['nodes'][0] # LT_obj is now a pointer to the first index of the dictionary
  mooring_nodes = [copy.deepcopy(N_obj) for x in range(len(numN))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
 
  num_anchors         = anchor_count - 1
  anchor_obj          = weis_obj['components']['mooring']['anchor_types'][0] # anchor_obj is now a pointer to the first index of the dictionary
  weis_obj['components']['mooring']['anchor_types'] = [copy.deepcopy(anchor_obj) for x in range(num_anchors)] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*

  anchor_count = 0

  # Make each mooring joint a node
  for n, mj in enumerate(moor_joints):
    mooring_nodes[n]['name']        = mj['name']
    mooring_nodes[n]['node_type']   = fast.fst_vt['MoorDyn']['Attachment'][n].lower()
    mooring_nodes[n]['joint']       = mj['name']
    # mooring_nodes[n]['node_mass']   = fast.fst_vt['MoorDyn']['M'][n]
    # mooring_nodes[n]['node_volume'] = fast.fst_vt['MoorDyn']['V'][n]
    # mooring_nodes[n]['location']    = [fast.fst_vt['MoorDyn']['X'][n], fast.fst_vt['MoorDyn']['Y'][n], fast.fst_vt['MoorDyn']['Z'][n]]
    if mooring_nodes[n]['node_type'] == 'fixed':
      # then also need joint, anchor_type
      mooring_nodes[n]['anchor_type'] = 'drag_embedment' # must be of lower(drag_embedment, suction, plate, micropile, sepla, custom)
      # weis_obj['components']['mooring']['anchor_types'][anchor_count]['name'] = 'anchor'+str(anchor_count+1)
      # weis_obj['components']['mooring']['anchor_types'][anchor_count]['type'] = '' # must be one of lower(drag_embedment, suction, plate, micropile, sepla, custom)
      # if its a custom anchor type, then will also need mass, cost, max_lateral_load, max_vertical_load
      # if weis_obj['components']['mooring']['anchor_types'][anchor_count]['type'].lower() == 'custom':
      #   weis_obj['components']['mooring']['anchor_types'][anchor_count]['mass'] = 0.0
      #   weis_obj['components']['mooring']['anchor_types'][anchor_count]['cost'] = 0.0
      #   weis_obj['components']['mooring']['anchor_types'][anchor_count]['max_lateral_load'] = 0.0
      #   weis_obj['components']['mooring']['anchor_types'][anchor_count]['max_vertical_load'] = 0.0
      anchor_count = anchor_count + 1
    elif mooring_nodes[n]['node_type'] == 'vessel':
      # then also need joint, fairlead_type ()
      mooring_nodes[n]['fairlead_type'] = 'rigid' # must be one of ['rigid','actuated','ball']

  weis_obj['components']['mooring']['nodes'] = mooring_nodes


  # Lines
  numL                = range(len(fast.fst_vt['MoorDyn']['Line_ID']))
  L_obj               = weis_obj['components']['mooring']['lines'][0] # LT_obj is now a pointer to the first index of the dictionary
  weis_obj['components']['mooring']['lines'] = [copy.deepcopy(L_obj) for x in range(len(numL))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*
  node_names = [node['name'] for node in mooring_nodes]
  for l in numL:
    weis_obj['components']['mooring']['lines'][l]['name']               = f"line_{fast.fst_vt['MoorDyn']['Line_ID'][l]}"
    weis_obj['components']['mooring']['lines'][l]['line_type']          = fast.fst_vt['MoorDyn']['LineType'][l]
    weis_obj['components']['mooring']['lines'][l]['unstretched_length'] = fast.fst_vt['MoorDyn']['UnstrLen'][l]
    weis_obj['components']['mooring']['lines'][l]['node1']              = node_names[fast.fst_vt['MoorDyn']['AttachA'][l]-1]
    weis_obj['components']['mooring']['lines'][l]['node2']              = node_names[fast.fst_vt['MoorDyn']['AttachB'][l]-1]

  # Anchor Types
  weis_obj['components']['mooring']['anchor_types'][0]['name'] = 'drag_embedment'
  weis_obj['components']['mooring']['anchor_types'][0]['type'] = 'drag_embedment' # must be one of [drag_embedment, suction, plate, micropile, sepla, Drag_Embedment, Suction, Plate, Micropile, Sepla, DRAG_EMBEDMENT, SUCTION, PLATE, MICROPILE, SEPLA, custom, Custom, CUSTOM]
  # if its a custom type, then will also need mass, cost, max_lateral_load, max_vertical_load

  print('Done')


# Control
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the control parameters to WEIS geometry schema and dictionary .........', end="", flush=True)

weis_obj['control']['supervisory']['Vin']        = fast.fst_vt['DISCON_in']['PS_WindSpeeds'][0]
weis_obj['control']['supervisory']['Vout']       = fast.fst_vt['DISCON_in']['PS_WindSpeeds'][-1]
# weis_obj['control']['supervisory']['maxTS']      = fast.fst_vt['DISCON_in']['VS_TSRopt'] * fast.fst_vt['InflowWind']['HWindSpeed']

weis_obj['control']['torque']['tsr']             = fast.fst_vt['DISCON_in']['VS_TSRopt'] 
# weis_obj['control']['torque']['VS_minspd']       = fast.fst_vt['DISCON_in']['VS_MinOMSpd'] / fast.fst_vt['DISCON_in']['WE_GearboxRatio'] # both WEIS and OpenFAST are in rad/s
weis_obj['control']['torque']['VS_maxspd']       = fast.fst_vt['DISCON_in']['VS_RefSpd'] / fast.fst_vt['DISCON_in']['WE_GearboxRatio']  # both in rad/s, WEIS is low speed, ROSCO is high speed
weis_obj['control']['torque']['max_torque_rate'] = fast.fst_vt['DISCON_in']['VS_MaxRat']   # both WEIS and OpenFAST are in Nm/s

weis_obj['control']['pitch']['max_pitch_rate']   = fast.fst_vt['DISCON_in']['PC_MaxRat']
weis_obj['control']['pitch']['min_pitch']        = fast.fst_vt['DISCON_in']['PC_MinPit']

print('Done')


# Materials
# -----------------------------------------------------------------------------------------------------------------------
# Go with database approach - fixed for now

# Initialize database
numMat                = range(20) # can be whatever size, 20 is arbritrarily big enough
Mat_obj               = weis_obj['materials'][0] # Mat_obj is now a pointer to the first index of the dictionary
weis_obj['materials'] = [copy.deepcopy(Mat_obj) for x in range(len(numMat))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*

# GelCoat
weis_obj['materials'][0]['name']             = 'gelcoat'
weis_obj['materials'][0]['description']      = ''
weis_obj['materials'][0]['source']           = ''
weis_obj['materials'][0]['orth']             = 0
weis_obj['materials'][0]['rho']              = 1235.0
weis_obj['materials'][0]['E']                = 3.440e+009
weis_obj['materials'][0]['G']                = 1.323e+009
weis_obj['materials'][0]['nu']               = 0.3
weis_obj['materials'][0]['alpha']            = 0.0
weis_obj['materials'][0]['Xt']               = 74
weis_obj['materials'][0]['Xc']               = 87
weis_obj['materials'][0]['Xy']               = 0
weis_obj['materials'][0]['S']                = 2.126E7
weis_obj['materials'][0]['GIc']              = 303
weis_obj['materials'][0]['GIIc']             = 3446
weis_obj['materials'][0]['alp0']             = 53
weis_obj['materials'][0]['ply_t']            = 5.0E-4
weis_obj['materials'][0]['m']                = 0
weis_obj['materials'][0]['waste']            = 0.25
weis_obj['materials'][0]['unit_cost']        = 7.23
weis_obj['materials'][0]['component_id']     = 0
weis_obj['materials'][0]['fvf']              = 0
weis_obj['materials'][0]['fwf']              = 0
weis_obj['materials'][0]['fiber_density']    = 0
weis_obj['materials'][0]['area_density_dry'] = 0
weis_obj['materials'][0]['roll_mass']        = 0

weis_obj['materials'][1]['name']             = 'steel'
weis_obj['materials'][1]['description']      = 'Steel of the tower and monopile ASTM A572 Grade 50'
weis_obj['materials'][1]['source']           = 'http://www.matweb.com/search/DataSheet.aspx?MatGUID=9ced5dc901c54bd1aef19403d0385d7f'
weis_obj['materials'][1]['orth']             = 0
weis_obj['materials'][1]['rho']              = 7800
weis_obj['materials'][1]['E']                = 200.e+009
weis_obj['materials'][1]['G']                = 79.3e+009
weis_obj['materials'][1]['nu']               = 0.3
weis_obj['materials'][1]['alpha']            = 0.0
weis_obj['materials'][1]['Xt']               = 450.e+006
weis_obj['materials'][1]['Xc']               = 450.e+006
weis_obj['materials'][1]['Xy']               = 345.e+6
weis_obj['materials'][1]['S']                = 0
weis_obj['materials'][1]['GIc']              = 0
weis_obj['materials'][1]['GIIc']             = 0
weis_obj['materials'][1]['alp0']             = 0
weis_obj['materials'][1]['ply_t']            = 0
weis_obj['materials'][1]['m']                = 3
weis_obj['materials'][1]['waste']            = 0
weis_obj['materials'][1]['unit_cost']        = 0.7
weis_obj['materials'][1]['component_id']     = 0
weis_obj['materials'][1]['fvf']              = 0
weis_obj['materials'][1]['fwf']              = 0
weis_obj['materials'][1]['fiber_density']    = 0
weis_obj['materials'][1]['area_density_dry'] = 0
weis_obj['materials'][1]['roll_mass']        = 0

weis_obj['materials'][2]['name']             = 'steel_drive'
weis_obj['materials'][2]['description']      = 'Steel of the drivetrain ASTM 4140 40Cr1Mo28'
weis_obj['materials'][2]['source']           = 'http://www.matweb.com/search/DataSheet.aspx?MatGUID=38108bfd64c44b4c9c6a02af78d5b6c6'
weis_obj['materials'][2]['orth']             = 0
weis_obj['materials'][2]['rho']              = 7850
weis_obj['materials'][2]['E']                = 205.e+009
weis_obj['materials'][2]['G']                = 80.0e+009
weis_obj['materials'][2]['nu']               = 0.3
weis_obj['materials'][2]['alpha']            = 0
weis_obj['materials'][2]['Xt']               = 814.e+006
weis_obj['materials'][2]['Xc']               = 814.e+006
weis_obj['materials'][2]['Xy']               = 485.e+6
weis_obj['materials'][2]['S']                = 0
weis_obj['materials'][2]['GIc']              = 0
weis_obj['materials'][2]['GIIc']             = 0
weis_obj['materials'][2]['alp0']             = 0
weis_obj['materials'][2]['ply_t']            = 0
weis_obj['materials'][2]['m']                = 3
weis_obj['materials'][2]['waste']            = 0
weis_obj['materials'][2]['unit_cost']        = 0.9
weis_obj['materials'][2]['component_id']     = 0
weis_obj['materials'][2]['fvf']              = 0
weis_obj['materials'][2]['fwf']              = 0
weis_obj['materials'][2]['fiber_density']    = 0
weis_obj['materials'][2]['area_density_dry'] = 0
weis_obj['materials'][2]['roll_mass']        = 0

weis_obj['materials'][3]['name']             = 'cast_iron'
weis_obj['materials'][3]['description']      = 'Cast iron for hub and nacelle components'
weis_obj['materials'][3]['source']           = ''
weis_obj['materials'][3]['orth']             = 0
weis_obj['materials'][3]['rho']              = 7200
weis_obj['materials'][3]['E']                = 118.e+009
weis_obj['materials'][3]['G']                = 47.6e+009
weis_obj['materials'][3]['nu']               = 0.3
weis_obj['materials'][3]['alpha']            = 0
weis_obj['materials'][3]['Xt']               = 310.e+006
weis_obj['materials'][3]['Xc']               = 310.e+006
weis_obj['materials'][3]['Xy']               = 265.e+6
weis_obj['materials'][3]['S']                = 0
weis_obj['materials'][3]['GIc']              = 0
weis_obj['materials'][3]['GIIc']             = 0
weis_obj['materials'][3]['alp0']             = 0
weis_obj['materials'][3]['ply_t']            = 0
weis_obj['materials'][3]['m']                = 3
weis_obj['materials'][3]['waste']            = 0
weis_obj['materials'][3]['unit_cost']        = 0.5
weis_obj['materials'][3]['component_id']     = 0
weis_obj['materials'][3]['fvf']              = 0
weis_obj['materials'][3]['fwf']              = 0
weis_obj['materials'][3]['fiber_density']    = 0
weis_obj['materials'][3]['area_density_dry'] = 0
weis_obj['materials'][3]['roll_mass']        = 0

weis_obj['materials'][4]['name']             = 'glass_uni'
weis_obj['materials'][4]['description']      = 'Vectorply E-LT-5500, Epikote MGS RIMR 135/Epicure MGS RIMH 1366 epoxy'
weis_obj['materials'][4]['source']           = 'MSU composites database 3D property tests, Engineering Mechanics of Composite Materials, Daniel, I & Ishai, O., 1994, pg. 34'
weis_obj['materials'][4]['orth']             = 1
weis_obj['materials'][4]['rho']              = 1940.0
weis_obj['materials'][4]['E']                = [4.46E10, 1.7E10, 1.67E10]
weis_obj['materials'][4]['G']                = [3.27E9, 3.48E9, 3.5E9]
weis_obj['materials'][4]['nu']               = [0.262, 0.35, 0.264]
weis_obj['materials'][4]['alpha']            = []
weis_obj['materials'][4]['Xt']               = [6.092E8, 3.81E7, 1.529E7]
weis_obj['materials'][4]['Xc']               = [4.7471E8, 1.1264E8, 1.1322E8]
weis_obj['materials'][4]['Xy']               = 0
weis_obj['materials'][4]['S']                = [1.891E7, 1.724E7, 1.316E7]
weis_obj['materials'][4]['GIc']              = 303
weis_obj['materials'][4]['GIIc']             = 3446
weis_obj['materials'][4]['alp0']             = 53
weis_obj['materials'][4]['ply_t']            = 0.005
weis_obj['materials'][4]['m']                = 10
weis_obj['materials'][4]['waste']            = 0.05
weis_obj['materials'][4]['unit_cost']        = 1.87
weis_obj['materials'][4]['component_id']     = 5
weis_obj['materials'][4]['fvf']              = 0.57
weis_obj['materials'][4]['fwf']              = 0.7450682696347697
weis_obj['materials'][4]['fiber_density']    = 2535.5
weis_obj['materials'][4]['area_density_dry'] = 7.227162215457267
weis_obj['materials'][4]['roll_mass']        = 0

weis_obj['materials'][5]['name']             = 'glass_biax'
weis_obj['materials'][5]['description']      = 'Vectorply E-LT-5500, Epikote MGS RIMR 135/Epicure MGS RIMH 1366 epoxy'
weis_obj['materials'][5]['source']           = 'MSU composites database 3D property tests, Engineering Mechanics of Composite Materials, Daniel, I & Ishai, O., 1994, pg. 34'
weis_obj['materials'][5]['orth']             = 1
weis_obj['materials'][5]['rho']              = 1940.0
weis_obj['materials'][5]['E']                = [1.11E10, 1.11E10, 1.67E10]
weis_obj['materials'][5]['G']                = [1.353E10, 3.49E9, 3.49E9]
weis_obj['materials'][5]['nu']               = [0.5, 0.0, 0.066]
weis_obj['materials'][5]['alpha']            = []
weis_obj['materials'][5]['Xt']               = [4.29E7, 4.26E7, 1.53E7]
weis_obj['materials'][5]['Xc']               = [7.07E7, 7.07E7, 1.132E8]
weis_obj['materials'][5]['Xy']               = 0
weis_obj['materials'][5]['S']                = [1.034E8, 1.72E7, 1.32E7]
weis_obj['materials'][5]['GIc']              = 303
weis_obj['materials'][5]['GIIc']             = 3446
weis_obj['materials'][5]['alp0']             = 53
weis_obj['materials'][5]['ply_t']            = 0.001
weis_obj['materials'][5]['m']                = 10
weis_obj['materials'][5]['waste']            = 0.15
weis_obj['materials'][5]['unit_cost']        = 3.0
weis_obj['materials'][5]['component_id']     = 3
weis_obj['materials'][5]['fvf']              = 0.57
weis_obj['materials'][5]['fwf']              = 0.7450682696347697
weis_obj['materials'][5]['fiber_density']    = 2535.5
weis_obj['materials'][5]['area_density_dry'] = 1.4454324430914534
weis_obj['materials'][5]['roll_mass']        = 181.4368

weis_obj['materials'][6]['name']             = 'glass_triax'
weis_obj['materials'][6]['description']      = 'Vectorply E-LT-5500, Epikote MGS RIMR 135/Epicure MGS RIMH 1366 epoxy'
weis_obj['materials'][6]['source']           = 'MSU composites database 3D property tests, Engineering Mechanics of Composite Materials, Daniel, I & Ishai, O., 1994, pg. 34'
weis_obj['materials'][6]['orth']             = 1.0
weis_obj['materials'][6]['rho']              = 1940.0
weis_obj['materials'][6]['E']                = [2.87E10, 1.66E10, 1.67E10]
weis_obj['materials'][6]['G']                = [8.4E9, 3.49E9, 3.49E9]
weis_obj['materials'][6]['nu']               = [0.5, 0.0, 0.17]
weis_obj['materials'][6]['alpha']            = []
weis_obj['materials'][6]['Xt']               = [3.96E8, 7.64E7, 1.53E7]
weis_obj['materials'][6]['Xc']               = [4.489E8, 1.747E8, 1.132E8]
weis_obj['materials'][6]['Xy']               = 0
weis_obj['materials'][6]['S']                = [1.034E8, 1.72E7, 1.32E7]
weis_obj['materials'][6]['GIc']              = 303
weis_obj['materials'][6]['GIIc']             = 3446
weis_obj['materials'][6]['alp0']             = 53
weis_obj['materials'][6]['ply_t']            = 0.001
weis_obj['materials'][6]['m']                = 10
weis_obj['materials'][6]['waste']            = 0.15
weis_obj['materials'][6]['unit_cost']        = 2.86
weis_obj['materials'][6]['component_id']     = 2
weis_obj['materials'][6]['fvf']              = 0.57
weis_obj['materials'][6]['fwf']              = 0.7450682696347697
weis_obj['materials'][6]['fiber_density']    = 2535.5
weis_obj['materials'][6]['area_density_dry'] = 1.4454324430914534
weis_obj['materials'][6]['roll_mass'] = 181.4368

weis_obj['materials'][7]['name'] = 'CarbonUD'
weis_obj['materials'][7]['description'] = ''
weis_obj['materials'][7]['source'] = ''
weis_obj['materials'][7]['orth'] = 1
weis_obj['materials'][7]['rho'] = 1220.0
weis_obj['materials'][7]['E'] = [114500000000.0, 8390000000.0, 8390000000.0]
weis_obj['materials'][7]['G'] = [5990000000.0, 5990000000.0, 5990000000.0]
weis_obj['materials'][7]['nu'] = [0.27, 0.27, 0.27]
weis_obj['materials'][7]['alpha'] = []
weis_obj['materials'][7]['Xt'] = [1546.e6, 0.0, 0.0]
weis_obj['materials'][7]['Xc'] = [1047.e6, 0.0, 0.0]
weis_obj['materials'][7]['Xy'] = 0
weis_obj['materials'][7]['S'] = [0.0, 0.0, 0.0]
weis_obj['materials'][7]['GIc'] = 0
weis_obj['materials'][7]['GIIc'] = 0
weis_obj['materials'][7]['alp0'] = 0
weis_obj['materials'][7]['ply_t'] = 0.005158730158730159
weis_obj['materials'][7]['m'] = 16.1
weis_obj['materials'][7]['waste'] = 0.05
weis_obj['materials'][7]['unit_cost'] = 30.0
weis_obj['materials'][7]['component_id'] = 4
weis_obj['materials'][7]['fvf'] = 0.1076923076923077
weis_obj['materials'][7]['fwf'] = 0.15889029003783103
weis_obj['materials'][7]['fiber_density'] = 1800.0
weis_obj['materials'][7]['area_density_dry'] = 1.0
weis_obj['materials'][7]['roll_mass']        = 0

weis_obj['materials'][8]['name']             = 'medium_density_foam'
weis_obj['materials'][8]['description']      = 'Airex C70.130 PVC Foam'
weis_obj['materials'][8]['source']           = 'https://www.3accorematerials.com/uploads/documents/TDS-AIREX-C70-E_1106.pdf'
weis_obj['materials'][8]['orth']             = 0.0
weis_obj['materials'][8]['rho']              = 130.0
weis_obj['materials'][8]['E']                = 1.292E8
weis_obj['materials'][8]['G']                = 4.8946969696969695E7
weis_obj['materials'][8]['nu']               = 0.32
weis_obj['materials'][8]['alpha']            = []
weis_obj['materials'][8]['Xt']               = 2083000.0
weis_obj['materials'][8]['Xc']               = 1563000.0
weis_obj['materials'][8]['Xy']               = 0
weis_obj['materials'][8]['S']                = 1250000.0
weis_obj['materials'][8]['GIc']              = 303
weis_obj['materials'][8]['GIIc']             = 3446
weis_obj['materials'][8]['alp0']             = 53
weis_obj['materials'][8]['ply_t']            = 0
weis_obj['materials'][8]['m']                = 0
weis_obj['materials'][8]['waste']            = 0.2
weis_obj['materials'][8]['unit_cost']        = 13
weis_obj['materials'][8]['component_id']     = 1
weis_obj['materials'][8]['fvf']              = 0
weis_obj['materials'][8]['fwf']              = 0
weis_obj['materials'][8]['fiber_density']    = 0
weis_obj['materials'][8]['area_density_dry'] = 0
weis_obj['materials'][8]['roll_mass']        = 0

weis_obj['materials'][9]['name']             = 'resin'
weis_obj['materials'][9]['description']      = 'epoxy'
weis_obj['materials'][9]['source']           = ''
weis_obj['materials'][9]['orth']             = 0
weis_obj['materials'][9]['rho']              = 1150.0
weis_obj['materials'][9]['E']                = 1.e+6
weis_obj['materials'][9]['G']                = 312500.0
weis_obj['materials'][9]['nu']               = 0.3
weis_obj['materials'][9]['alpha']            = 0
weis_obj['materials'][9]['Xt']               = 0
weis_obj['materials'][9]['Xc']               = 0
weis_obj['materials'][9]['Xy']               = 0
weis_obj['materials'][9]['S']                = 0
weis_obj['materials'][9]['GIc']              = 0
weis_obj['materials'][9]['GIIc']             = 0
weis_obj['materials'][9]['alp0']             = 0
weis_obj['materials'][9]['ply_t']            = 0
weis_obj['materials'][9]['m']                = 0
weis_obj['materials'][9]['waste']            = 0
weis_obj['materials'][9]['unit_cost']        = 3.63
weis_obj['materials'][9]['component_id']     = 0
weis_obj['materials'][9]['fvf']              = 0
weis_obj['materials'][9]['fwf']              = 0
weis_obj['materials'][9]['fiber_density']    = 0
weis_obj['materials'][9]['area_density_dry'] = 0
weis_obj['materials'][9]['roll_mass']        = 0

weis_obj['materials'][10]['name']             = 'adhesive'
weis_obj['materials'][10]['description']      = 'Sample adhesive'
weis_obj['materials'][10]['source']           = 'https://www.nrel.gov/docs/fy19osti/73585.pdf'
weis_obj['materials'][10]['orth']             = 0
weis_obj['materials'][10]['rho']              = 1100.0
weis_obj['materials'][10]['E']                = 4.56e+006
weis_obj['materials'][10]['G']                = 1520000.0
weis_obj['materials'][10]['nu']               = 0.49
weis_obj['materials'][10]['alpha']            = 0.0
weis_obj['materials'][10]['Xt']               = 0.69e+006
weis_obj['materials'][10]['Xc']               = 0.4e+006
weis_obj['materials'][10]['Xy']               = 0
weis_obj['materials'][10]['S']                = 0.31e+006
weis_obj['materials'][10]['GIc']              = 0
weis_obj['materials'][10]['GIIc']             = 0
weis_obj['materials'][10]['alp0']             = 0
weis_obj['materials'][10]['ply_t']            = 0
weis_obj['materials'][10]['m']                = 0
weis_obj['materials'][10]['waste']            = 0
weis_obj['materials'][10]['unit_cost']        = 9.0
weis_obj['materials'][10]['component_id']     = 0
weis_obj['materials'][10]['fvf']              = 0
weis_obj['materials'][10]['fwf']              = 0
weis_obj['materials'][10]['fiber_density']    = 0
weis_obj['materials'][10]['area_density_dry'] = 0
weis_obj['materials'][10]['roll_mass']        = 0

# Required user inputs, use defaults
# -----------------------------------------------------------------------------------------------------------------------


# Drivetrain
drive_props = merged_schema['properties']['components']['properties']['nacelle']['properties']['drivetrain']['properties']

weis_obj['components']['nacelle']['drivetrain']['mb1Type']                                 = drive_props['mb1Type']['default']
weis_obj['components']['nacelle']['drivetrain']['mb2Type']                                 = drive_props['mb2Type']['default']

gen_props = merged_schema['properties']['components']['properties']['nacelle']['properties']['generator']['properties']

weis_obj['components']['nacelle']['generator']['generator_type']                           = gen_props['generator_type']['default']


# Mooring
if fast.fst_vt['Fst']['CompMooring'] > 0: # if there is mooring, use it!

  # Line Types
  for lt in numLT:
    weis_obj['components']['mooring']['line_types'][lt]['type']     = 'chain' # MATERIAL NOT AVAILABLE IN OPENFAST: [chain, chain_stud, nylon, polyester, polypropylene, wire_fiber, fiber, wire, wire_wire, iwrc, Chain, Chain_Stud, Nylon, Polyester, Polypropylene, Wire, Wire_Fiber, Fiber, Wire, Wire_Wire, IWRC, CHAIN, CHAIN_STUD, NYLON, POLYESTER, POLYPROPYLENE, WIRE, WIRE_FIBER, FIBER, WIRE, WIRE_WIRE, custom, Custom, CUSTOM]

  # Nodes (same as Points)
  numN                = range(len(fast.fst_vt['MoorDyn']['Point_ID']))
  for n in numN:
    if fast.fst_vt['MoorDyn']['Attachment'][0].lower() == 'fixed':
      # then also need joint, anchor_type
      # mooring_nodes[n]['joint'] = ''   # This comes from HydroDyn
      mooring_nodes[n]['anchor_type'] = 'rigid' # must be one from list below
    elif fast.fst_vt['MoorDyn']['Attachment'][0].lower() == 'vessel':
      # then also need joint, fairlead_type ()
      # mooring_nodes[n]['joint'] = ''   # Also from hydrodyn?? May
      mooring_nodes[n]['fairlead_type'] = 'rigid' # must be one of ['rigid','actuated','ball']

  weis_obj['components']['mooring']['anchor_types'][0]['name'] = 'drag_embedment'
  weis_obj['components']['mooring']['anchor_types'][0]['type'] = 'drag_embedment' # must be one of [drag_embedment, suction, plate, micropile, sepla, Drag_Embedment, Suction, Plate, Micropile, Sepla, DRAG_EMBEDMENT, SUCTION, PLATE, MICROPILE, SEPLA, custom, Custom, CUSTOM]



#=========================================================================================================================
# OUTPUTS

# Print out the final, new weis geometry yaml input file
project_name = fast.FAST_InputFile.split('.')[0]
print('Write the clean output geometry yaml file .........', end="", flush=True)
weis_obj = remove_numpy(weis_obj)
write_geometry_yaml(weis_obj, os.path.join(this_dir,project_name + '_CLEAN.yaml'))
print('Done')

#=========================================================================================================================
# ASSUMPTIONS

# Insert assumed variable values below here


# Turbine System Costs
# -----------------------------------------------------------------------------------------------------------------------
weis_obj['costs'] = assumed_model['costs'].copy()


# weis_obj['costs']['wake_loss_factor']                      = 0.15
# weis_obj['costs']['fixed_charge_rate']                     = 0.056
# weis_obj['costs']['bos_per_kW']                            = 4053.
# weis_obj['costs']['opex_per_kW']                           = 137.
# weis_obj['costs']['turbine_number']                        = 40.
# weis_obj['costs']['labor_rate']                            = 58.8
# weis_obj['costs']['painting_rate']                         = 30.0
# weis_obj['costs']['blade_mass_cost_coeff']                 = 14.6
# weis_obj['costs']['hub_mass_cost_coeff']                   = 3.9
# weis_obj['costs']['pitch_system_mass_cost_coeff']          = 22.1
# weis_obj['costs']['spinner_mass_cost_coeff']               = 11.1
# weis_obj['costs']['lss_mass_cost_coeff']                   = 11.9
# weis_obj['costs']['bearing_mass_cost_coeff']               = 4.5
# weis_obj['costs']['gearbox_mass_cost_coeff']               = 12.9
# weis_obj['costs']['hss_mass_cost_coeff']                   = 6.8
# weis_obj['costs']['generator_mass_cost_coeff']             = 12.4
# weis_obj['costs']['bedplate_mass_cost_coeff']              = 2.9
# weis_obj['costs']['yaw_mass_cost_coeff']                   = 8.3
# weis_obj['costs']['converter_mass_cost_coeff']             = 18.8
# weis_obj['costs']['transformer_mass_cost_coeff']           = 18.8
# weis_obj['costs']['hvac_mass_cost_coeff']                  = 124.0
# weis_obj['costs']['cover_mass_cost_coeff']                 = 5.7
# weis_obj['costs']['elec_connec_machine_rating_cost_coeff'] = 41.85
# weis_obj['costs']['platforms_mass_cost_coeff']             = 17.1
# weis_obj['costs']['tower_mass_cost_coeff']                 = 2.9
# weis_obj['costs']['controls_machine_rating_cost_coeff']    = 21.15
# weis_obj['costs']['crane_cost']                            = 12e3

# Balance of System Costs
# -----------------------------------------------------------------------------------------------------------------------
weis_obj['bos'] = assumed_model['bos'].copy()

# weis_obj['bos']['plant_turbine_spacing']             = 7
# weis_obj['bos']['plant_row_spacing']                 = 7
# weis_obj['bos']['commissioning_pct']                 = 0.01
# weis_obj['bos']['decommissioning_pct']               = 0.15
# weis_obj['bos']['distance_to_substation']            = 1.0
# weis_obj['bos']['distance_to_interconnection']       = 8.5
# weis_obj['bos']['interconnect_voltage']              = 130.
# weis_obj['bos']['distance_to_site']                  = 115.
# weis_obj['bos']['distance_to_landfall']              = 50.
# weis_obj['bos']['port_cost_per_month']               = 2e6
# weis_obj['bos']['site_auction_price']                = 100e6
# weis_obj['bos']['site_assessment_plan_cost']         = 1e6
# weis_obj['bos']['site_assessment_cost']              = 25e6
# weis_obj['bos']['construction_operations_plan_cost'] = 2.5e6
# weis_obj['bos']['boem_review_cost']                  = 0.0
# weis_obj['bos']['design_install_plan_cost']          = 2.5e6

# Hub info
weis_obj['components']['hub'] = assumed_model['components']['hub']
weis_obj['components']['hub']['diameter']         = 2*fast.fst_vt['ElastoDyn']['HubRad']
weis_obj['components']['hub']['cone_angle']       = radians(fast.fst_vt['ElastoDyn']['PreCone(1)']) # assumes all three blades are at the same cone angle (very reasonable and practical assumption)
weis_obj['components']['hub']['drag_coefficient'] = 0.0 # NOT AVAILABLE IN OPENFAST

# Generator info: all generator info provided above looked like placeholders
nacelle_model = iea_3
weis_obj['components']['nacelle']['generator'] = nacelle_model['components']['nacelle']['generator']

# Drivetrain info: some can be inferred from OpenFAST model, try to keep that
drivetrain_params = [
  'distance_hub_mb',
  'distance_mb_mb',
  'generator_length',
  'generator_radius_user',
  'generator_mass_user',
  'gearbox_length_user',
  'gearbox_radius_user',
  'gearbox_mass_user',
  'damping_ratio',
  'lss_diameter',
  'lss_wall_thickness',
  'lss_material',
  'hss_length',
  'hss_diameter',
  'hss_wall_thickness',
  'hss_material',
  'nose_diameter',
  'nose_wall_thickness',
  'bedplate_flange_width',
  'bedplate_flange_thickness',
  'bedplate_web_thickness',
  'bedplate_material',
  'brake_mass_user',
  'hvac_mass_coefficient',
  'converter_mass_user',
  'transformer_mass_user',
  'mb1Type',
  'mb2Type',
  'uptower',
  'gear_configuration',
  'planet_numbers'
]

weis_drivetrain = weis_obj['components']['nacelle']['drivetrain']
assumed_drivetrain = nacelle_model['components']['nacelle']['drivetrain']
for param in drivetrain_params:
  weis_drivetrain[param] = assumed_drivetrain[param]


weis_drivetrain['generator_rpm_efficiency_user']['grid']    = assumed_drivetrain['generator_rpm_efficiency_user']['grid']
weis_drivetrain['generator_rpm_efficiency_user']['values']  = assumed_drivetrain['generator_rpm_efficiency_user']['values'] 
weis_drivetrain['bedplate_wall_thickness']['grid']          = assumed_drivetrain['bedplate_wall_thickness']['grid']
weis_drivetrain['bedplate_wall_thickness']['values']        = assumed_drivetrain['bedplate_wall_thickness']['values']

# Blade Structure
# -----------------------------------------------------------------------------------------------------------------------
weis_obj['components']['blade']['internal_structure_2d_fem'] = assumed_model['components']['blade']['internal_structure_2d_fem'].copy()

# Tower Structure
# -----------------------------------------------------------------------------------------------------------------------
weis_obj['components']['tower']['internal_structure_2d_fem'] = assumed_model['components']['tower']['internal_structure_2d_fem'].copy()
# Note that reference axis of internal_structure_2d_fem could change outer_shape_bem because they are linked in yaml


# Platform internal structure
if fast.fst_vt['Fst']['CompMooring'] > 0: # if there is mooring, then parameterize the floating hull/platform  
  
  assumed_members = assumed_model['components']['floating_platform']['members']
  members = weis_obj['components']['floating_platform']['members']
  
  for member, assumed_member in zip(members,assumed_members):
    member['internal_structure'] = assumed_member['internal_structure'].copy()

# Anchor types
weis_obj['components']['mooring']['anchor_types'] = assumed_model['components']['mooring']['anchor_types']


# Use assumed model materials
weis_obj['materials'] = assumed_model['materials']


# Tidy WEIS object

# Remove mooring node locations
for node in weis_obj['components']['mooring']['nodes']:
  del(node['location'])

# Remove monopile
del(weis_obj['components']['monopile'])

# More customizations
weis_obj['control']['supervisory']['Vin']        = 0.5
weis_obj['control']['supervisory']['Vout']       = 4.
# weis_obj['control']['supervisory']['maxTS']      = fast.fst_vt['DISCON_in']['VS_TSRopt'] * fast.fst_vt['InflowWind']['HWindSpeed']

weis_obj['control']['torque']['tsr']             = fast.fst_vt['DISCON_in']['VS_TSRopt'] 
weis_obj['control']['torque']['VS_minspd']       = 0.
weis_obj['assembly']['rated_power']              = 500000  # There's an error in current DISCON



# Print out the final, weis geometry yaml input file with assumptions
print('Writing the output geometry yaml file with assumptions, approximations, and estimates .........', end="", flush=True)
write_geometry_yaml(weis_obj, os.path.join(this_dir,project_name + '_ASSUMED.yaml'))

print('Done')

# Write yaml without floating platform for testing
del(weis_obj['components']['floating_platform'])
del(weis_obj['components']['mooring'])
write_geometry_yaml(weis_obj, os.path.join(this_dir,project_name + '_ASSUMED_NoPtfm.yaml'))

print('Done')

exit()
