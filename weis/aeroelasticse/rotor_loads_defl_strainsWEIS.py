from openmdao.api import Group, ExplicitComponent
from wisdem.rotorse.rotor_structure import ComputeStrains, DesignConstraints, BladeRootSizing
import numpy as np

class RotorLoadsDeflStrainsWEIS(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        self.add_subsystem("m2pa", MtoPrincipalAxes(modeling_options=modeling_options), promotes=['alpha', 'M1', 'M2'])
        self.add_subsystem("strains", ComputeStrains(modeling_options=modeling_options), promotes=['alpha', 'M1', 'M2'])
        self.add_subsystem("constr", DesignConstraints(modeling_options=modeling_options, opt_options=opt_options))
        self.add_subsystem("brs", BladeRootSizing(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))

        # Strains from frame3dd to constraint
        self.connect('strains.strainU_spar', 'constr.strainU_spar')
        self.connect('strains.strainL_spar', 'constr.strainL_spar')
         
class MtoPrincipalAxes(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        
        self.add_input(
            "alpha",
            val=np.zeros(n_span),
            units="deg",
            desc="Angle between blade c.s. and principal axes",
        )

        self.add_input(
            "Mx",
            val=np.zeros(n_span),
            units="N*m/m",
            desc="distribution along blade span of edgewise bending moment",
        )
        self.add_input(
            "My",
            val=np.zeros(n_span),
            units="N*m/m",
            desc="distribution along blade span of flapwise bending moment",
        )

        self.add_output(
            "M1",
            val=np.zeros(n_span),
            units="N*m/m",
            desc="distribution along blade span of bending moment w.r.t principal axis 1",
        )
        self.add_output(
            "M2",
            val=np.zeros(n_span),
            units="N*m/m",
            desc="distribution along blade span of bending moment w.r.t principal axis 2",
        )

    def compute(self, inputs, outputs):
        
        alpha = inputs['alpha']

        Mx = inputs['Mx']
        My = inputs['My']

        ca = np.cos(np.deg2rad(alpha))
        sa = np.sin(np.deg2rad(alpha))

        def rotate(x, y):
            x2 = x * ca + y * sa
            y2 = -x * sa + y * ca
            return x2, y2
        
        M1, M2 = rotate(-My, -Mx)

        outputs['M1'] = M1
        outputs['M2'] = M2