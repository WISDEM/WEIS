from openmdao.api import ExplicitComponent, Group, IndepVarComp
import numpy as np

def assign_TMD_values(wt_opt,wt_init,opt_options):
    '''
    Assign initial TMD values
    '''
    for i_TMD, TMD in enumerate(wt_init['TMDs']):
        wt_opt['TMDs.mass'][i_TMD]         = TMD['mass']
        wt_opt['TMDs.stiffness'][i_TMD]    = TMD['stiffness']
        wt_opt['TMDs.damping'][i_TMD]      = TMD['damping']

        # check if natural frequency and damping ratio are defined, print warning if overwriting input
        if TMD['natural_frequency'] > 0:
            wt_opt['TMDs.stiffness'][i_TMD] = TMD['natural_frequency']**2 * TMD['mass']
            if wt_opt['TMDs.stiffness'][i_TMD] > 0:
                print("TMD Warning: the natural frequency of {} is overwriting the stiffness".format(TMD['name']))

        if TMD['damping_ratio'] > 0:
            if 'natural_frequency' in TMD:
                wt_opt['TMDs.damping'][i_TMD] = 2 * TMD['damping_ratio'] * TMD['natural_frequency'] * TMD['mass']
            else:
                raise Exception('You must define the natural_frequency if you define the damping_ratio')
            if wt_opt['TMDs.damping'][i_TMD] > 0:
                print("TMD Warning: the damping ratio of {} is overwriting the damping".format(TMD['name']))

    # IVC groups
    for i_group, tmd_group in enumerate(opt_options['design_variables']['TMDs']['groups']):
        # Set initial values to first in group
        if 'mass' in opt_options['design_variables']['TMDs']['groups'][i_group]:
            wt_opt[f'TMDs.TMD_IVCs.group_{i_group}_mass'] = tmd_group['mass']['initial']

        if 'stiffness' in opt_options['design_variables']['TMDs']['groups'][i_group]:
            wt_opt[f'TMDs.TMD_IVCs.group_{i_group}_stiffness'] = tmd_group['stiffness']['initial']
        
        if 'natural_frequency' in opt_options['design_variables']['TMDs']['groups'][i_group]:
            wt_opt[f'TMDs.TMD_IVCs.group_{i_group}_natural_frequency'] = tmd_group['natural_frequency']['initial']

        if 'damping' in opt_options['design_variables']['TMDs']['groups'][i_group]:
            wt_opt[f'TMDs.TMD_IVCs.group_{i_group}_damping'] = tmd_group['damping']['initial']
        
        if 'damping_ratio' in opt_options['design_variables']['TMDs']['groups'][i_group]:
            wt_opt[f'TMDs.TMD_IVCs.group_{i_group}_damping_ratio'] = tmd_group['damping_ratio']['initial']

    return wt_opt

class TMD_group(Group):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']
        
        self.add_subsystem('TMDs',
            TMDs(modeling_options = modeling_options, opt_options=opt_options),
            promotes=
                [
                    'mass',
                    'stiffness',
                    'damping',
                ]
        )

        if opt_options['design_variables']['TMDs']['flag']:
            self.add_subsystem('TMD_IVCs',
                TMD_IVCs(modeling_options = modeling_options, opt_options=opt_options),
            )
            
            if 'TMDs' in opt_options['design_variables']:
                n_groups = len(opt_options['design_variables']['TMDs']['groups'])
                for i_group in range(n_groups):
                    self.connect(f'TMD_IVCs.group_{i_group}_mass',              f'TMDs.group_{i_group}_mass')
                    self.connect(f'TMD_IVCs.group_{i_group}_stiffness',         f'TMDs.group_{i_group}_stiffness')
                    self.connect(f'TMD_IVCs.group_{i_group}_damping',           f'TMDs.group_{i_group}_damping')
                    self.connect(f'TMD_IVCs.group_{i_group}_natural_frequency', f'TMDs.group_{i_group}_natural_frequency')
                    self.connect(f'TMD_IVCs.group_{i_group}_damping_ratio',     f'TMDs.group_{i_group}_damping_ratio')


class TMDs(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.opt_options = self.options['opt_options']
                
        n_TMDs = self.modeling_options['TMDs']['n_TMDs']
        self.n_groups = len(self.opt_options['design_variables']['TMDs']['groups'])

        for i_group in range(self.n_groups):
            self.add_input(f'group_{i_group}_mass',                 val=0, units='kg',        desc='TMD Mass')
            self.add_input(f'group_{i_group}_stiffness',            val=0, units='N/m',       desc='TMD Stiffnes')
            self.add_input(f'group_{i_group}_damping',              val=0, units='N/(m/s)',   desc='TMD Damping')
            self.add_input(f'group_{i_group}_natural_frequency',    val=-1, units='rad/s',     desc='TMD natural frequencies')
            self.add_input(f'group_{i_group}_damping_ratio',        val=-1,                    desc='TMD damping ratios')

        self.add_output('mass',                 val=np.zeros(n_TMDs), units='kg',           desc='TMD Mass')
        self.add_output('stiffness',            val=np.zeros(n_TMDs), units='N/m',          desc='TMD Stiffnes')
        self.add_output('damping',              val=np.zeros(n_TMDs), units='N/(m/s)',      desc='TMD Damping')

    def compute(self, inputs, outputs):
        '''
        Map TMD ivc groups to list of TMDs
        Someday, we'll do things like mapping the relative locations to global coords
        '''

        # only update outputs if they are a design variable
        if self.opt_options['design_variables']['TMDs']['flag']:

            group_mapping = self.modeling_options['TMDs']['group_mapping']  
            
            # for each group
            for i_group, tmd_group in enumerate(self.opt_options['design_variables']['TMDs']['groups']):
                # which turbines are in group i
                tmds_in_group = group_mapping[i_group]
                for i_TMD in tmds_in_group:

                    # original values
                    omega   = np.sqrt(outputs['stiffness'][i_TMD]/outputs['mass'][i_TMD])
                    zeta    = outputs['damping'][i_TMD] / (2 * omega * outputs['mass'][i_TMD])

                    # mass
                    if 'mass' in tmd_group:
                        outputs['mass'][i_TMD] = inputs[f'group_{i_group}_mass']
                        if 'const_omega' in tmd_group['mass'] and tmd_group['mass']['const_omega']:
                            outputs['stiffness'][i_TMD] = omega**2 * outputs['mass'][i_TMD]
                        if 'const_zeta' in tmd_group['mass'] and tmd_group['mass']['const_zeta']:
                            outputs['damping'][i_TMD] = 2 * omega * zeta * outputs['mass'][i_TMD]

                    # stiffness:
                    if 'stiffness' in tmd_group:
                        outputs['stiffness'][i_TMD] = inputs[f'group_{i_group}_stiffness']
                    
                    if 'natural_frequency' in tmd_group:
                        outputs['stiffness'][i_TMD] = inputs[f'group_{i_group}_natural_frequency']**2 * outputs['mass'][i_TMD]
                        if 'const_zeta' in tmd_group['natural_frequency'] and tmd_group['natural_frequency']['const_zeta']:
                            outputs['damping'][i_TMD] = 2 * inputs[f'group_{i_group}_natural_frequency'] * zeta * outputs['mass'][i_TMD]

                    # damping:
                    if 'damping' in tmd_group:
                        outputs['damping'][i_TMD] = inputs[f'group_{i_group}_damping']
                    
                    if 'damping_ratio' in tmd_group:
                        omega   = np.sqrt(outputs['stiffness'][i_TMD]/outputs['mass'][i_TMD])  # recalculate in case stiffness has changed
                        outputs['damping'][i_TMD] = 2 * inputs[f'group_{i_group}_damping_ratio'] * omega * outputs['mass'][i_TMD]

            # print('here')  # to check mapping

class TMD_IVCs(IndepVarComp):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options = self.options['opt_options']

        n_TMDs = modeling_options['TMDs']['n_TMDs']
        
        n_groups = len(opt_options['design_variables']['TMDs']['groups'])

        for i_group in range(n_groups):
            self.add_output(f'group_{i_group}_mass',               val=-1, units='kg',        desc='TMD Mass')
            self.add_output(f'group_{i_group}_stiffness',          val=-1, units='N/m',       desc='TMD Stiffnes')
            self.add_output(f'group_{i_group}_damping',            val=-1, units='N/(m/s)',   desc='TMD Damping')
            self.add_output(f'group_{i_group}_natural_frequency',  val=-1, units='rad/s',     desc='TMD natural frequency')
            self.add_output(f'group_{i_group}_damping_ratio',      val=-1,                    desc='TMD damping ratio')

