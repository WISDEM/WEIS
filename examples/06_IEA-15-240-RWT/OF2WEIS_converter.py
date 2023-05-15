
import copy, numpy as np
import weis.inputs as inp
from math import degrees, radians, pi
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from weis.inputs.validation import get_geometry_schema
from wisdem.inputs import validate_without_defaults

#=========================================================================================================================
# OTHER FUNCTIONS

def compute_relThk(x,y):
  
  LE_index = np.argmin(x)
  x_upper = np.flip(x[0:LE_index+1]) # python skips the last item
  y_upper = np.flip(y[0:LE_index+1]) # python skips the last item
  
  x_lower = x[LE_index:-1] # python skips the last item
  y_lower = y[LE_index:-1] # python skips the last item

  thickness = [0 for a in range(len(x_upper))] # initialize the array

  # compute camber and thickness
  for z in range(len(x_upper)):
    if i == 50:
        print('debug me')
    yu = np.interp(x_upper[z],x_upper,y_upper)
    yl = np.interp(x_upper[z],x_lower,y_lower)
    camber = (yu + yl)/2
    thickness[z] = yu-camber

  max_thk = 2*np.round(max(abs(np.array(thickness))),2) # thickness is symmetric about the camber line
  return max_thk

# ===============================================================================================================
# Inputs

# OpenFAST
fast = InputReader_OpenFAST()
fast.FAST_InputFile = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
fast.FAST_directory = '/home/nmendoza/Projects/Ct-Opt/OpenFAST/openfast/examples/IEA-15MW-RWT'   # Path to fst directory files
fast.execute()
print('successfully imported fast.fst_vt')

# WEIS
#weis_schema = get_geometry_schema()
#print(weis_schema['properties']['components']['properties']['blade']['properties']['outer_shape_bem']['properties']['chord']['properties']['values']['$ref'])

# FOR VERIFICATION ONLY
#weis_complete = inp.load_geometry_yaml('/home/nmendoza/Projects/Ct-Opt/WEIS/examples/06_IEA-15-240-RWT/IEA-15-floating_wTMDs.yaml')

finput        = '/home/nmendoza/Projects/Ct-Opt/WEIS/examples/06_IEA-15-240-RWT/IEA-15-floating_blank.yaml'
merged_schema = get_geometry_schema()
weis_obj      = validate_without_defaults(finput, merged_schema)
print('successfully imported blank weis dictionary')

# =======================================================================================================================
# Conversion from OpenFAST to WEIS (by component)

# Airfoils
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the airfoils to WEIS geometry schema and dictionary .............')
numAF                = range(fast.fst_vt['AeroDyn15']['NumAFfiles'])
AF_obj               = weis_obj['airfoils'][0] # AF_obj is now a pointer to the first index of the dictionary
weis_obj['airfoils'] = [copy.deepcopy(AF_obj) for x in range(len(numAF))] # deepcopy recursively copies the dictionary structure, creates duplicate, *ALWAYS USE DEEPCOPY*

for i in numAF:
    #print('i = ',i)
    # coordinates
    weis_obj['airfoils'][i]['coordinates']['x']           = fast.fst_vt['AeroDyn15']['af_coord'][i]['x']
    weis_obj['airfoils'][i]['coordinates']['y']           = fast.fst_vt['AeroDyn15']['af_coord'][i]['y']
    # properties
    weis_obj['airfoils'][i]['name']                       = str(fast.fst_vt['AeroDynBlade']['BlAFID'][i])
    weis_obj['airfoils'][i]['aerodynamic_center']         = fast.fst_vt['AeroDyn15']['ac'][i]
    weis_obj['airfoils'][i]['relative_thickness']         = compute_relThk(weis_obj['airfoils'][i]['coordinates']['x'],weis_obj['airfoils'][i]['coordinates']['y'])
    # polars
    AoA = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Alpha']
    weis_obj['airfoils'][i]['polars'][0]['c_l']['grid']   = [A * pi/180 for A in AoA]
    weis_obj['airfoils'][i]['polars'][0]['c_l']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cl']
    weis_obj['airfoils'][i]['polars'][0]['c_d']['grid']   = [A * pi/180 for A in AoA]
    weis_obj['airfoils'][i]['polars'][0]['c_d']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cd']
    weis_obj['airfoils'][i]['polars'][0]['c_m']['grid']   = [A * pi/180 for A in AoA]
    weis_obj['airfoils'][i]['polars'][0]['c_m']['values'] = fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cm']

print('Done')

# Blades
# -----------------------------------------------------------------------------------------------------------------------
print('Converting the blade planform properties to WEIS geometry schema and dictionary .........')

# Blade nodes = nondimensionalize
blade_length = fast.fst_vt['AeroDynBlade']['BlSpn'][-1]
BlSpn = fast.fst_vt['AeroDynBlade']['BlSpn']
# Airfoil Positions
weis_obj['components']['blade']['outer_shape_bem']['airfoil_position']['grid']      = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['airfoil_position']['labels']    = str(fast.fst_vt['AeroDynBlade']['BlAFID'])
# Chord
weis_obj['components']['blade']['outer_shape_bem']['chord']['grid']                 = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['chord']['values']               = fast.fst_vt['AeroDynBlade']['BlChord']
# Twist
weis_obj['components']['blade']['outer_shape_bem']['twist']['grid']                 = [L / blade_length for L in BlSpn]
twist = fast.fst_vt['AeroDynBlade']['BlTwist'] # convert to radians
weis_obj['components']['blade']['outer_shape_bem']['twist']['values']               = [T * pi/180 for T in twist]
# Pitch Axis
weis_obj['components']['blade']['outer_shape_bem']['pitch_axis']['grid']            = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['pitch_axis']['values']          = fast.fst_vt['ElastoDynBlade']['PitchAxis']
# Reference Axis (normal prebend is negative in weis)
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid']   = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['x']['values'] = [-x for x in fast.fst_vt['AeroDynBlade']['BlCrvAC']]
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid']   = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['y']['values'] = fast.fst_vt['AeroDynBlade']['BlSwpAC']
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid']   = [L / blade_length for L in BlSpn]
weis_obj['components']['blade']['outer_shape_bem']['reference_axis']['z']['values'] = fast.fst_vt['AeroDynBlade']['BlSpn']

print('Done')

# Hub
# -----------------------------------------------------------------------------------------------------------------------

# Nacelle
# -----------------------------------------------------------------------------------------------------------------------

# Tower
# -----------------------------------------------------------------------------------------------------------------------

# Monopile
# -----------------------------------------------------------------------------------------------------------------------

# Floating Platform
# -----------------------------------------------------------------------------------------------------------------------

# Mooring
# -----------------------------------------------------------------------------------------------------------------------


#=========================================================================================================================
# OUTPUTS

# Print out the final, new weis geometry yaml file
#write_geometry_yaml(weis_obj, '/home/nmendoza/Projects/Ct-Opt/WEIS/examples/06_IEA-15-240-RWT/IEA-15-floating_NEW.yaml')