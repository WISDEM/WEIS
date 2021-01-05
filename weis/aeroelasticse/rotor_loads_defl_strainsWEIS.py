from openmdao.api import Group
from wisdem.rotorse.rotor_structure import BladeCurvature, TotalLoads, RunFrame3DD, TipDeflection, DesignConstraints


class RotorLoadsDeflStrainsWEIS(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        # Load blade with rated conditions and compute aerodynamic forces
        promoteListAeroLoads =  ['r', 'theta', 'chord', 'Rtip', 'Rhub', 'hub_height', 'precone', 'tilt', 'airfoils_aoa', 'airfoils_Re', 'airfoils_cl', 'airfoils_cd', 'airfoils_cm', 'nBlades', 'rho', 'mu', 'Omega_load','pitch_load']
        # Add centrifugal and gravity loading to aero loading
        promotes=['tilt','theta','rhoA','z','totalCone','z_az']
        self.add_subsystem('curvature',         BladeCurvature(modeling_options = modeling_options),  promotes=['r','precone','precurve','presweep','3d_curv','x_az','y_az','z_az'])
        promoteListTotalLoads = ['r', 'theta', 'tilt', 'rhoA', '3d_curv', 'z_az']
        self.add_subsystem('tot_loads_gust',        TotalLoads(modeling_options = modeling_options),      promotes=promoteListTotalLoads)
        promoteListFrame3DD = ['x_az','y_az','z_az','theta','r','A','EA','EIxx','EIyy','EIxy','GJ','rhoA','rhoJ','x_ec','y_ec','xu_strain_spar','xl_strain_spar','yu_strain_spar','yl_strain_spar','xu_strain_te','xl_strain_te','yu_strain_te','yl_strain_te']
        self.add_subsystem('frame',     RunFrame3DD(modeling_options = modeling_options),      promotes=promoteListFrame3DD)
        self.add_subsystem('tip_pos',   TipDeflection(),                                  promotes=['tilt','pitch_load'])
        self.add_subsystem('constr',    DesignConstraints(modeling_options = modeling_options, opt_options = opt_options))

        # Total loads to strains
        self.connect('tot_loads_gust.Px_af', 'frame.Px_af')
        self.connect('tot_loads_gust.Py_af', 'frame.Py_af')
        self.connect('tot_loads_gust.Pz_af', 'frame.Pz_af')

        # Blade distributed deflections to tip deflection
        self.connect('frame.dx', 'tip_pos.dx_tip', src_indices=[-1])
        self.connect('frame.dy', 'tip_pos.dy_tip', src_indices=[-1])
        self.connect('frame.dz', 'tip_pos.dz_tip', src_indices=[-1])
        self.connect('3d_curv',  'tip_pos.3d_curv_tip', src_indices=[-1])

        # Strains from frame3dd to constraint
        self.connect('frame.strainU_spar', 'constr.strainU_spar')
        self.connect('frame.strainL_spar', 'constr.strainL_spar')
        self.connect('frame.flap_mode_freqs', 'constr.flap_mode_freqs')
        self.connect('frame.edge_mode_freqs', 'constr.edge_mode_freqs')

         
