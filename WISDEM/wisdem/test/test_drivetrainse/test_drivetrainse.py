import unittest
from wisdem.drivetrainse.drivetrain import DrivetrainSE
import openmdao.api as om
import numpy as np
import traceback, sys

def set_common(prob, opt):
    prob['n_blades'] = 3
    prob['rotor_diameter'] = 120.0
    prob['machine_rating'] = 5e3
    prob['D_top'] = 6.5

    prob['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
    prob['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))

    prob['lss_E']    = prob['hub_E']   = prob['bedplate_E'] = 210e9
    prob['lss_G']    = prob['hub_G']   = prob['bedplate_G'] = 80.8e9
    prob['lss_rho']  = prob['hub_rho'] = prob['bedplate_rho'] = 7850.
    prob['lss_Xy']   = prob['hub_Xy']   = prob['bedplate_Xy'] = 250e6
    prob['hub_mat_cost'] = prob['bedplate_mat_cost'] = prob['spinner_mat_cost'] = 3.
    prob['spinner_Xt'] = 60.e6
    #prob['lss_cost'] = prob['hss_cost'] =
    if not opt['drivetrainse']['direct']:
        prob['hss_E']    = prob['bedplate_E']
        prob['hss_G']    = prob['bedplate_G']
        prob['hss_rho']  = prob['bedplate_rho']
        prob['hss_Xy']   = prob['bedplate_Xy']
        
    prob['blade_mass']                  = 17000.
    prob['pitch_system.BRFM']           = 1.e+6
    prob['pitch_system_scaling_factor'] = 0.54

    prob['blade_root_diameter']         = 4.
    prob['flange_t2shell_t']            = 4.
    prob['flange_OD2hub_D']             = 0.5
    prob['flange_ID2flange_OD']         = 0.8
    prob['hub_rho']                     = 7200.
    prob['hub_Xy']                      = 200.e+6
    prob['hub_stress_concentration']    = 2.5

    prob['n_front_brackets']            = 3
    prob['n_rear_brackets']             = 3
    prob['clearance_hub_spinner']       = 0.5
    prob['spin_hole_incr']              = 1.2
    prob['spinner_gust_ws']             = 70.

    prob['hub_diameter']                = 6.235382907247958
    prob['rated_rpm']                   = 10            #8.68                # rpm 9.6
    prob['blades_I']                    = 1e3*np.ones(6)
    prob['blades_I'][0]                 = 199200777.51 * 30. * 5 / 10. / np.pi
    
    return prob



class TestGroup(unittest.TestCase):
    
    def testDirectDrive_withGen(self):

        opt = {}
        opt['drivetrainse'] = {}
        opt['drivetrainse']['n_height'] = npts = 10
        opt['drivetrainse']['direct'] = True
        opt['drivetrainse']['hub'] = {}
        opt['drivetrainse']['hub']['hub_gamma'] = 2.0
        opt['drivetrainse']['hub']['spinner_gamma'] = 1.5
        opt['drivetrainse']['gamma_f'] = 1.35
        opt['drivetrainse']['gamma_m'] = 1.3
        opt['drivetrainse']['gamma_n'] = 1.0
        opt['servose'] = {}
        opt['servose']['n_pc'] = 20
        opt['GeneratorSE'] = {}
        opt['GeneratorSE']['type'] = 'pmsg_outer'
        opt['flags'] = {}
        opt['flags']['generator'] = True
        
        prob = om.Problem()
        prob.model = DrivetrainSE(modeling_options=opt, topLevelFlag=True, n_dlcs=1)
        prob.setup()
        prob = set_common(prob, opt)

        prob['upwind'] = True

        prob['L_12'] = 2.0
        prob['L_h1'] = 1.0
        prob['L_generator'] = 3.25
        prob['overhang'] = 6.25
        prob['drive_height'] = 4.875
        prob['tilt'] = 4.0
        prob['access_diameter'] = 0.9

        myones = np.ones(5)
        prob['lss_diameter'] = 3.3*myones
        prob['nose_diameter'] = 2.2*myones
        prob['lss_wall_thickness'] = 0.45*myones
        prob['nose_wall_thickness'] = 0.1*myones
        prob['bedplate_wall_thickness'] = 0.06*np.ones(npts)

        prob['rated_torque']        = 10.25e6       #rev 1 9.94718e6
        prob['generator.P_mech']         = 10.71947704e6 #rev 1 9.94718e6
        prob['generator.rad_ag']            = 4.0           # rev 1  4.92
        prob['generator.len_s']          = 1.7           # rev 2.3
        prob['generator.h_s']            = 0.7            # rev 1 0.3
        prob['generator.p']              = 70            #100.0    # rev 1 160
        prob['generator.h_m']            = 0.005         # rev 1 0.034
        prob['generator.h_ys']           = 0.04          # rev 1 0.045
        prob['generator.h_yr']           = 0.06          # rev 1 0.045
        prob['generator.b']              = 2.
        prob['generator.c']              = 5.0
        prob['generator.B_tmax']         = 1.9
        prob['generator.E_p']            = 3300/np.sqrt(3)
        prob['generator.D_nose']         = 2*1.1             # Nose outer radius
        prob['generator.D_shaft']        = 2*1.34            # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
        prob['generator.t_r']            = 0.05          # Rotor disc thickness
        prob['generator.h_sr']           = 0.04          # Rotor cylinder thickness
        prob['generator.t_s']            = 0.053         # Stator disc thickness
        prob['generator.h_ss']           = 0.04          # Stator cylinder thickness
        prob['generator.u_allow_pcent']  = 8.5            # % radial deflection
        prob['generator.y_allow_pcent']  = 1.0            # % axial deflection
        prob['generator.z_allow_deg']    = 0.05           # torsional twist
        prob['generator.sigma']          = 60.0e3         # Shear stress
        prob['generator.B_r']            = 1.279
        prob['generator.ratio_mw2pp']    = 0.8
        prob['generator.h_0']            = 5e-3
        prob['generator.h_w']            = 4e-3
        prob['generator.k_fes']          = 0.8
        prob['generator.C_Cu']         = 4.786         # Unit cost of Copper $/kg
        prob['generator.C_Fe']         = 0.556         # Unit cost of Iron $/kg
        prob['generator.C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
        prob['generator.C_PM']         =   95.0
        prob['generator.rho_Fe']       = 7700.0        # Steel density Kg/m3
        prob['generator.rho_Fes']      = 7850          # structural Steel density Kg/m3
        prob['generator.rho_Copper']   = 8900.0        # copper density Kg/m3
        prob['generator.rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)
        
    def testDirectDrive_withSimpleGen(self):

        opt = {}
        opt['drivetrainse'] = {}
        opt['drivetrainse']['n_height'] = npts = 10
        opt['drivetrainse']['direct'] = True
        opt['drivetrainse']['hub'] = {}
        opt['drivetrainse']['hub']['hub_gamma'] = 2.0
        opt['drivetrainse']['hub']['spinner_gamma'] = 1.5
        opt['drivetrainse']['gamma_f'] = 1.35
        opt['drivetrainse']['gamma_m'] = 1.3
        opt['drivetrainse']['gamma_n'] = 1.0
        opt['servose'] = {}
        opt['servose']['n_pc'] = 20
        opt['flags'] = {}
        opt['flags']['generator'] = False

        prob = om.Problem()
        prob.model = DrivetrainSE(modeling_options=opt, topLevelFlag=True, n_dlcs=1)
        prob.setup()
        prob = set_common(prob, opt)

        prob['upwind'] = True

        prob['L_12'] = 2.0
        prob['L_h1'] = 1.0
        prob['L_generator'] = 3.25
        prob['overhang'] = 6.25
        prob['drive_height'] = 4.875
        prob['tilt'] = 4.0
        prob['access_diameter'] = 0.9

        myones = np.ones(5)
        prob['lss_diameter'] = 3.3*myones
        prob['nose_diameter'] = 2.2*myones
        prob['lss_wall_thickness'] = 0.45*myones
        prob['nose_wall_thickness'] = 0.1*myones
        prob['bedplate_wall_thickness'] = 0.06*np.ones(npts)

        prob['generator_efficiency_user'] = np.zeros((20,2))
        
        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        
    def testGeared_withGen(self):

        opt = {}
        opt['drivetrainse'] = {}
        opt['drivetrainse']['n_height'] = npts = 10
        opt['drivetrainse']['direct'] = False
        opt['drivetrainse']['hub'] = {}
        opt['drivetrainse']['hub']['hub_gamma'] = 2.0
        opt['drivetrainse']['hub']['spinner_gamma'] = 1.5
        opt['drivetrainse']['gamma_f'] = 1.35
        opt['drivetrainse']['gamma_m'] = 1.3
        opt['drivetrainse']['gamma_n'] = 1.0
        opt['GeneratorSE'] = {}
        opt['GeneratorSE']['type'] = 'dfig'
        opt['servose'] = {}
        opt['servose']['n_pc'] = 20
        opt['flags'] = {}
        opt['flags']['generator'] = True

        prob = om.Problem()
        prob.model = DrivetrainSE(modeling_options=opt, topLevelFlag=True, n_dlcs=1)
        prob.setup()
        prob = set_common(prob, opt)

        prob['upwind'] = True

        prob['L_12'] = 2.0
        prob['L_h1'] = 1.0        
        prob['overhang'] = 2.0
        prob['drive_height'] = 4.875
        prob['L_hss'] = 1.5
        prob['L_generator'] = 1.25
        prob['L_gearbox'] = 1.1
        prob['tilt'] = 5.0

        myones = np.ones(5)
        prob['lss_diameter'] = 2.3*myones
        prob['lss_wall_thickness'] = 0.05*myones
        myones = np.ones(3)
        prob['hss_diameter'] = 2.0*myones
        prob['hss_wall_thickness'] = 0.05*myones

        prob['bedplate_flange_width'] = 1.5
        prob['bedplate_flange_thickness'] = 0.05
        #prob['bedplate_web_height'] = 1.0
        prob['bedplate_web_thickness'] = 0.05

        prob['planet_numbers'] = np.array([3, 3, 0])
        prob['gear_configuration'] = 'eep'
        #prob['shaft_factor'] = 'normal'
        prob['gear_ratio'] = 90.0
        prob['gearbox_efficiency'] = 0.955

        prob['generator.rho_Fe']         = 7700.0
        prob['generator.rho_Fes']        = 7850.0
        prob['generator.rho_Copper']     = 8900.0
        prob['generator.rho_PM']         = 7450.0
        prob['generator.B_r']            = 1.2
        prob['generator.E']              = 2e11
        prob['generator.G']              = 79.3e9
        prob['generator.P_Fe0e']         = 1.0
        prob['generator.P_Fe0h']         = 4.0
        prob['generator.S_N']            = -0.002
        prob['generator.alpha_p']        = 0.5*np.pi*0.7
        prob['generator.b_r_tau_r']      = 0.45
        prob['generator.b_ro']           = 0.004
        prob['generator.b_s_tau_s']      = 0.45
        prob['generator.b_so']           = 0.004
        prob['generator.cofi']           = 0.85
        prob['generator.freq']           = 60
        prob['generator.h_i']            = 0.001
        prob['generator.h_sy0']          = 0.0
        prob['generator.h_w']            = 0.005
        prob['generator.k_fes']          = 0.9
        prob['generator.k_s']            = 0.2
        prob['generator.m']     = 3
        prob['generator.mu_0']           = np.pi*4e-7
        prob['generator.mu_r']           = 1.06
        prob['generator.p']              = 3.0
        prob['generator.phi']            = np.deg2rad(90)
        prob['generator.ratio_mw2pp']    = 0.7
        prob['generator.resist_Cu']      = 1.8e-8*1.4
        prob['generator.sigma']          = 40e3
        prob['generator.v']              = 0.3
        prob['generator.y_tau_p']        = 1.0
        prob['generator.y_tau_pr']       = 10. / 12
        prob['generator.cofi']               = 0.9
        prob['generator.y_tau_p']            = 12./15.
        prob['generator.sigma']              = 21.5e3
        prob['generator.rad_ag']             = 0.61
        prob['generator.len_s']              = 0.49
        prob['generator.h_s']                = 0.08
        prob['generator.I_0']                = 40.0
        prob['generator.B_symax']            = 1.3
        prob['generator.S_Nmax']             = -0.2
        prob['generator.h_0']                = 0.01
        prob['generator.k_fillr']        = 0.55
        prob['generator.k_fills']        = 0.65
        prob['generator.q1']    = 5
        prob['generator.q2']    = 4
        
        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        
    def testGeared_withSimpleGen(self):

        opt = {}
        opt['drivetrainse'] = {}
        opt['drivetrainse']['n_height'] = npts = 10
        opt['drivetrainse']['direct'] = False
        opt['drivetrainse']['hub'] = {}
        opt['drivetrainse']['hub']['hub_gamma'] = 2.0
        opt['drivetrainse']['hub']['spinner_gamma'] = 1.5
        opt['drivetrainse']['gamma_f'] = 1.35
        opt['drivetrainse']['gamma_m'] = 1.3
        opt['drivetrainse']['gamma_n'] = 1.0
        opt['flags'] = {}
        opt['flags']['generator'] = False
        opt['servose'] = {}
        opt['servose']['n_pc'] = 20

        prob = om.Problem()
        prob.model = DrivetrainSE(modeling_options=opt, topLevelFlag=True, n_dlcs=1)
        prob.setup()
        prob = set_common(prob, opt)

        prob['upwind'] = True

        prob['L_12'] = 2.0
        prob['L_h1'] = 1.0        
        prob['overhang'] = 2.0
        prob['drive_height'] = 4.875
        prob['L_hss'] = 1.5
        prob['L_generator'] = 1.25
        prob['L_gearbox'] = 1.1
        prob['tilt'] = 5.0

        myones = np.ones(5)
        prob['lss_diameter'] = 2.3*myones
        prob['lss_wall_thickness'] = 0.05*myones
        myones = np.ones(3)
        prob['hss_diameter'] = 2.0*myones
        prob['hss_wall_thickness'] = 0.05*myones

        prob['bedplate_flange_width'] = 1.5
        prob['bedplate_flange_thickness'] = 0.05
        #prob['bedplate_web_height'] = 1.0
        prob['bedplate_web_thickness'] = 0.05

        prob['planet_numbers'] = np.array([3, 3, 0])
        prob['gear_configuration'] = 'eep'
        #prob['shaft_factor'] = 'normal'
        prob['gear_ratio'] = 90.0

        prob['generator_efficiency_user'] = np.zeros((20,2))

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)



def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGroup))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
