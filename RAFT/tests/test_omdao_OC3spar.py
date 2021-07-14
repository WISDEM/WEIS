import numpy as np
import openmdao.api as om
import numpy as np
import os
import yaml

import raft
from raft.omdao_raft import RAFT_OMDAO
from common import check, test

# -----------------------------------
# OMDAO
# -----------------------------------
w = np.arange(0.05, 5, 0.05)  # frequency range (to be set by modeling options yaml)

# -------------------------
# options
# -------------------------
opt = {}

opt['modeling'] = {}
opt['modeling']['nfreq'] = len(w)
opt['modeling']['potModMaster'] = 1
opt['modeling']['XiStart'] = 0
opt['modeling']['nIter'] = 0
opt['modeling']['dlsMax'] = 5.0

opt['turbine'] = {}
opt['turbine']['npts'] = 11
opt['turbine']['shape'] = 'circ'
opt['turbine']['scalar_thicknesses'] = False
opt['turbine']['scalar_diameters'] = False
opt['turbine']['scalar_coefficients'] = True

# TODO: this is a little awkward - what happens if nmembers != length of array?
opt['members'] = {}
opt['members']['nmembers'] = 1
opt['members']['npts'] = np.array([4])
opt['members']['npts_lfill'] = np.array([3])
opt['members']['npts_rho_fill'] = np.array([3])
opt['members']['ncaps'] = np.array([1])
opt['members']['nreps'] = np.array([1])
opt['members']['shape'] = np.array(['circ']) # can be 'circ', 'rect', or 'square'
opt['members']['scalar_diameters'] = np.array([False])
opt['members']['scalar_thicknesses'] = np.array([True])
opt['members']['scalar_coefficients'] = np.array([True])

opt['mooring'] = {}
opt['mooring']['nlines'] = 3
opt['mooring']['nline_types'] = 1
opt['mooring']['nconnections'] = 6

prob = om.Problem()
prob.model = RAFT_OMDAO(modeling_options=opt['modeling'],
                        turbine_options=opt['turbine'],
                        mooring_options=opt['mooring'],
                        member_options=opt['members'])
prob.setup()

# -------------------------
# inputs
# -------------------------
prob['frequency_range'] = w

# -------------------------
# turbine
# -------------------------
prob['turbine_mRNA'] = 350000
prob['turbine_IxRNA'] = 35444067
prob['turbine_IrRNA'] = 26159984.0
prob['turbine_xCG_RNA'] = 0
prob['turbine_hHub'] = 90.0
prob['turbine_Fthrust'] = 800.0E3
prob['turbine_yaw_stiffness'] = 98340000.0
# tower
prob['turbine_tower_rA'] = [0, 0, 10]
prob['turbine_tower_rB'] = [0, 0, 87.6]
prob['turbine_tower_gamma'] = 0.0
prob['turbine_tower_stations'] = [10, 17.76, 25.52, 33.28, 41.04, 48.8, 56.56, 64.32, 72.08, 79.84, 87.6]
prob['turbine_tower_d'] = [6.5, 6.237, 5.974, 5.711, 5.448, 5.185, 4.922, 4.659, 4.396, 4.133, 3.870]
prob['turbine_tower_t'] = [0.027, 0.0262, 0.0254, 0.0246, 0.0238, 0.023, 0.0222, 0.0214, 0.0206, 0.0198, 0.0190]
prob['turbine_tower_Cd'] = 0.0
prob['turbine_tower_Ca'] = 0.0
prob['turbine_tower_CdEnd'] = 0.0
prob['turbine_tower_CaEnd'] = 0.0
prob['turbine_tower_rho_shell'] = 8500

# -------------------------
# platform
# -------------------------
# member 1
prob['platform_member1_rA'] = [0, 0, -120]
prob['platform_member1_rB'] = [0, 0, 10]
prob['platform_member1_gamma'] = 0.0
prob['platform_member1_potMod'] = True
prob['platform_member1_stations'] = [-120, -12, -4, 10]
prob['platform_member1_d'] = [9.4, 9.4, 6.5, 6.5]
prob['platform_member1_t'] = 0.027
prob['platform_member1_Cd'] = 0.8
prob['platform_member1_Ca'] = 1.0
prob['platform_member1_CdEnd'] = 0.6
prob['platform_member1_CaEnd'] = 0.6
prob['platform_member1_rho_shell'] = 8500
prob['platform_member1_l_fill'] = [52.9, 0.0, 0.0]
prob['platform_member1_rho_fill'] = [1800.0, 0.0, 0.0]
prob['platform_member1_cap_stations'] = [-120]
prob['platform_member1_cap_t'] = [0.2]
prob['platform_member1_cap_d_in'] = [0]

# -------------------------
# mooring
# -------------------------
prob['mooring_water_depth'] = 320
# connection points
prob['mooring_point1_name'] = 'line1_anchor'
prob['mooring_point1_type'] = 'fixed'
prob['mooring_point1_location'] = [853.87, 0.0, -320.0]
prob['mooring_point2_name'] = 'line2_anchor'
prob['mooring_point2_type'] = 'fixed'
prob['mooring_point2_location'] = [-426.935, 739.47311, -320.0]
prob['mooring_point3_name'] = 'line3_anchor'
prob['mooring_point3_type'] = 'fixed'
prob['mooring_point3_location'] = [-426.935, -739.47311, -320.0]
prob['mooring_point4_name'] = 'line1_vessel'
prob['mooring_point4_type'] = 'vessel'
prob['mooring_point4_location'] = [5.2, 0, -70.0]
prob['mooring_point5_name'] = 'line2_vessel'
prob['mooring_point5_type'] = 'vessel'
prob['mooring_point5_location'] = [-2.6, 4.5033, -70.0]
prob['mooring_point6_name'] = 'line3_vessel'
prob['mooring_point6_type'] = 'vessel'
prob['mooring_point6_location'] = [-2.6, -4.5033, -70.0]
# lines
prob['mooring_line1_endA'] = 'line1_anchor'
prob['mooring_line1_endB'] = 'line1_vessel'
prob['mooring_line1_type'] = 'chain'
prob['mooring_line1_length'] = 902.2
prob['mooring_line2_endA'] = 'line2_anchor'
prob['mooring_line2_endB'] = 'line2_vessel'
prob['mooring_line2_type'] = 'chain'
prob['mooring_line2_length'] = 902.2
prob['mooring_line3_endA'] = 'line3_anchor'
prob['mooring_line3_endB'] = 'line3_vessel'
prob['mooring_line3_type'] = 'chain'
prob['mooring_line3_length'] = 902.2
# line types
prob['mooring_line_type1_name'] = 'chain'
prob['mooring_line_type1_diameter'] = 0.09
prob['mooring_line_type1_mass_density'] = 77.7066
prob['mooring_line_type1_stiffness'] = 384.243e6
prob['mooring_line_type1_breaking_load'] = 1e8
prob['mooring_line_type1_cost'] = 100.0
prob['mooring_line_type1_transverse_added_mass'] = 1.0
prob['mooring_line_type1_tangential_added_mass'] = 0.0
prob['mooring_line_type1_transverse_drag'] = 1.6
prob['mooring_line_type1_tangential_drag'] = 0.1

prob.run_model()


# -----------------------------------
# YAML input
# -----------------------------------

fname_design = os.path.join(os.path.join('..', 'designs'), 'OC3spar.yaml')

# open the design YAML file and parse it into a dictionary for passing to raft
with open(fname_design) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)
design['potModMaster'] = 1

# grab the depth (currently needs to be passed separately)
depth = float(design['mooring']['water_depth'])

# set up frequency range for computing response over
w = np.arange(0.05, 5, 0.05)  # frequency range (to be set by modeling options yaml)

# Create and run the model
model = raft.Model(design, w=w, depth=depth)

model.setEnv(spectrum="unit")

model.calcSystemProps()

model.solveEigen()

model.calcMooringAndOffsets()

model.solveDynamics()

results = model.calcOutputs()
print('-----------------')
testPass = test(prob, results)

print('Test ' + ('FAILED' if not testPass else 'PASSED'))
