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
opt['modeling']['XiStart'] = 0.1        # default
opt['modeling']['nIter'] = 15           # default
opt['modeling']['dlsMax'] = 5.0         # default

opt['turbine'] = {}
opt['turbine']['npts'] = 20
opt['turbine']['shape'] = 'circ'
opt['turbine']['scalar_thicknesses'] = False
opt['turbine']['scalar_diameters'] = False
opt['turbine']['scalar_coefficients'] = True

# TODO: this is a little awkward - what happens if nmembers != length of array?
opt['members'] = {}
opt['members']['nmembers'] = 4
opt['members']['npts'] = np.array([2, 2, 2, 2])
opt['members']['npts_lfill'] = np.array([0, 1, 1, 0])
opt['members']['npts_rho_fill'] = np.array([0, 1, 1, 0])
opt['members']['ncaps'] = np.array([1, 1, 0, 0])
opt['members']['nreps'] = np.array([1, 3, 3, 3])
opt['members']['shape'] = np.array(['circ', 'circ', 'rect', 'circ']) # can be 'circ', 'rect', or 'square'
#opt['members']['scalar_diameters'] = np.array([True, True, False, True])
# leaving above commented line to show that the third member is rectangular, but still has 'scalar diameters' since the side lengths are the same at endA and endB
# in this sense, 'scalar diameter' refers to either a scalar diameter (circ) or a list of size 2 (rect) that is consistent throughout the member
opt['members']['scalar_diameters'] = np.array([True, True, True, True])
opt['members']['scalar_thicknesses'] = np.array([True, True, True, True])
opt['members']['scalar_coefficients'] = np.array([True, True, True, True])

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
prob['turbine_mRNA'] = 991000
prob['turbine_IxRNA'] = 0
prob['turbine_IrRNA'] = 0
prob['turbine_xCG_RNA'] = 0
prob['turbine_hHub'] = 150.0
prob['turbine_Fthrust'] = 1500.0E3
prob['turbine_yaw_stiffness'] = 0.0
# tower
prob['turbine_tower_rA'] = [0, 0, 15]
prob['turbine_tower_rB'] = [0, 0, 144.582]
prob['turbine_tower_gamma'] = 0.0
prob['turbine_tower_stations'] = [15, 28, 28.001, 41, 41.001, 54, 54.001, 67, 67.001, 80, 80.001, 93, 93.001, 106, 106.001, 119, 119.001, 132, 132.001, 144.582]
prob['turbine_tower_d'] = [10, 9.964, 9.964, 9.967, 9.967, 9.927, 9.927, 9.528, 9.528, 9.149, 9.149, 8.945, 8.945, 8.735, 8.735, 8.405, 8.405, 7.321, 7.321, 6.5]
prob['turbine_tower_t'] = [0.082954, 0.082954, 0.083073, 0.083073, 0.082799, 0.082799, 0.0299, 0.0299, 0.027842, 0.027842, 0.025567, 0.025567, 0.022854, 0.022854, 0.02025, 0.02025, 0.018339, 0.018339, 0.021211, 0.021211]
prob['turbine_tower_Cd'] = 0.0
prob['turbine_tower_Ca'] = 0.0
prob['turbine_tower_CdEnd'] = 0.0
prob['turbine_tower_CaEnd'] = 0.0
prob['turbine_tower_rho_shell'] = 7850.0

# -------------------------
# platform
# -------------------------
# member 1
prob['platform_member1_rA'] = [0, 0, -20]
prob['platform_member1_rB'] = [0, 0, 15]
prob['platform_member1_gamma'] = 0.0
prob['platform_member1_potMod'] = True
prob['platform_member1_stations'] = [0, 1]
prob['platform_member1_d'] = 10.0
prob['platform_member1_t'] = 0.05
prob['platform_member1_Cd'] = 0.8
prob['platform_member1_Ca'] = 1.0
prob['platform_member1_CdEnd'] = 0.6
prob['platform_member1_CaEnd'] = 0.6
prob['platform_member1_rho_shell'] = 7850
prob['platform_member1_cap_stations'] = [0]
prob['platform_member1_cap_t'] = [0.001]
prob['platform_member1_cap_d_in'] = [0]
# member 2
prob['platform_member2_heading'] = [60, 180, 300]
prob['platform_member2_rA'] = [51.75, 0, -20.0]
prob['platform_member2_rB'] = [51.75, 0, 15]
prob['platform_member2_gamma'] = 0.0
prob['platform_member2_potMod'] = True
prob['platform_member2_stations'] = [0, 1]
prob['platform_member2_d'] = 12.5
prob['platform_member2_t'] = 0.05
prob['platform_member2_Cd'] = 0.8
prob['platform_member2_Ca'] = 1.0
prob['platform_member2_CdEnd'] = 0.6
prob['platform_member2_CaEnd'] = 0.6
prob['platform_member2_rho_shell'] = 7850
prob['platform_member2_l_fill'] = 1.4
prob['platform_member2_rho_fill'] = 5000
prob['platform_member2_cap_stations'] = [0]
prob['platform_member2_cap_t'] = [0.001]
prob['platform_member2_cap_d_in'] = [0]
# member 3
prob['platform_member3_heading'] = [60, 180, 300]
prob['platform_member3_rA'] = [5, 0, -16.5]
prob['platform_member3_rB'] = [45.5, 0, -16.5]
prob['platform_member3_gamma'] = 0.0
prob['platform_member3_potMod'] = False
prob['platform_member3_stations'] = [0, 1]
prob['platform_member3_d'] = [12.5, 7.0]
prob['platform_member3_t'] = .05
prob['platform_member3_Cd'] = 0.8
prob['platform_member3_Ca'] = 1.0
prob['platform_member3_CdEnd'] = 0.6
prob['platform_member3_CaEnd'] = 0.6
prob['platform_member3_rho_shell'] = 7850
prob['platform_member3_l_fill'] = 43.0
prob['platform_member3_rho_fill'] = 1025.0
# member 4
prob['platform_member4_heading'] = [60, 180, 300]
prob['platform_member4_rA'] = [5, 0, 14.545]
prob['platform_member4_rB'] = [45.5, 0, 14.545]
prob['platform_member4_gamma'] = 0.0
prob['platform_member4_potMod'] = False
prob['platform_member4_stations'] = [0, 1]
prob['platform_member4_d'] = 0.91
prob['platform_member4_t'] = 0.01
prob['platform_member4_Cd'] = 0.8
prob['platform_member4_Ca'] = 1.0
prob['platform_member4_CdEnd'] = 0.6
prob['platform_member4_CaEnd'] = 0.6
prob['platform_member4_rho_shell'] = 7850

# -------------------------
# mooring
# -------------------------
prob['mooring_water_depth'] = 200
# connection points
prob['mooring_point1_name'] = 'line1_anchor'
prob['mooring_point1_type'] = 'fixed'
prob['mooring_point1_location'] = [-837., 0.0, -200.0]
prob['mooring_point2_name'] = 'line2_anchor'
prob['mooring_point2_type'] = 'fixed'
prob['mooring_point2_location'] = [418, 725, -200.0]
prob['mooring_point3_name'] = 'line3_anchor'
prob['mooring_point3_type'] = 'fixed'
prob['mooring_point3_location'] = [418, -725, -200.0]
prob['mooring_point4_name'] = 'line1_vessel'
prob['mooring_point4_type'] = 'vessel'
prob['mooring_point4_location'] = [-58., 0, -14.0]
prob['mooring_point5_name'] = 'line2_vessel'
prob['mooring_point5_type'] = 'vessel'
prob['mooring_point5_location'] = [29, 50, -14.0]
prob['mooring_point6_name'] = 'line3_vessel'
prob['mooring_point6_type'] = 'vessel'
prob['mooring_point6_location'] = [29, -50, -14.0]
# lines
prob['mooring_line1_endA'] = 'line1_anchor'
prob['mooring_line1_endB'] = 'line1_vessel'
prob['mooring_line1_type'] = 'chain'
prob['mooring_line1_length'] = 850
prob['mooring_line2_endA'] = 'line2_anchor'
prob['mooring_line2_endB'] = 'line2_vessel'
prob['mooring_line2_type'] = 'chain'
prob['mooring_line2_length'] = 850
prob['mooring_line3_endA'] = 'line3_anchor'
prob['mooring_line3_endB'] = 'line3_vessel'
prob['mooring_line3_type'] = 'chain'
prob['mooring_line3_length'] = 850
# line types
prob['mooring_line_type1_name'] = 'chain'
prob['mooring_line_type1_diameter'] = 0.185
prob['mooring_line_type1_mass_density'] = 685.0
prob['mooring_line_type1_stiffness'] = 3270e6
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

fname_design = os.path.join(os.path.join('..', 'designs'), 'VolturnUS-S.yaml')

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
