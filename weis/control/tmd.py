from openmdao.api import ExplicitComponent, Group, IndepVarComp
import numpy as np

def assign_TMD_values(wt_opt,wt_init):
    '''
    Assign initial TMD values
    only one TMD supported for now, but it's an array in wt_init for future use
    '''
    TMD = wt_init['TMDs']
    for i_TMD, TMD in enumerate(wt_init['TMDs']):
        wt_opt['TMDs.group_mass'][i_TMD]         = TMD['mass']
        wt_opt['TMDs.group_stiffness'][i_TMD]    = TMD['stiffness']
        wt_opt['TMDs.group_damping'][i_TMD]      = TMD['damping']

        # check if natural frequency and damping ratio are defined, print warning if overwriting input
        if TMD['natural_frequency'] > 0:
            wt_opt['TMDs.group_natural_frequency'][i_TMD] = TMD['natural_frequency']
            if wt_opt['TMDs.group_stiffness'][i_TMD] > 0:
                print("TMD Warning: the natural frequency of {} is overwriting the stiffness".format(TMD['name']))

        if TMD['damping_ratio'] > 0:
            wt_opt['TMDs.group_damping_ratio'][i_TMD] = TMD['damping_ratio']
            if wt_opt['TMDs.group_damping'][i_TMD] > 0:
                print("TMD Warning: the damping ratio of {} is overwriting the damping".format(TMD['name']))

    return wt_opt

class TMD_group(Group):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        self.add_subsystem('TMD_IVCs',
            TMD_IVCs(modeling_options = modeling_options, opt_options=opt_options),
            promotes=
                [
                    'group_mass',
                    'group_stiffness',
                    'group_damping',
                    'group_natural_frequency',
                    'group_damping_ratio',
                ]
        )
        
        self.add_subsystem('TMDs',
            TMDs(modeling_options = modeling_options, opt_options=opt_options),
            promotes=
                [
                    'mass',
                    'stiffness',
                    'damping',
                ]
        )
        

        self.connect('group_mass','TMDs.group_mass')
        self.connect('group_stiffness','TMDs.group_stiffness')
        self.connect('group_damping','TMDs.group_damping')
        self.connect('group_natural_frequency','TMDs.group_natural_frequency')
        self.connect('group_damping_ratio','TMDs.group_damping_ratio')






class TMDs(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.opt_options = self.options['opt_options']
                
        n_TMDs = self.modeling_options['TMDs']['n_TMDs']
        n_groups = n_TMDs   # temporaray

        self.add_input('group_mass',                 val=np.zeros(n_groups), units='kg',        desc='TMD Mass')
        self.add_input('group_stiffness',            val=np.zeros(n_groups), units='N/m',       desc='TMD Stiffnes')
        self.add_input('group_damping',              val=np.zeros(n_groups), units='N/(m/s)',   desc='TMD Damping')
        self.add_input('group_natural_frequency',    val=-np.ones(n_groups), units='rad/s',     desc='TMD natural frequencies')
        self.add_input('group_damping_ratio',        val=-np.ones(n_groups),                    desc='TMD damping ratios')

        self.add_output('mass',                 val=np.zeros(n_TMDs), units='kg',           desc='TMD Mass')
        self.add_output('stiffness',            val=np.zeros(n_TMDs), units='N/m',          desc='TMD Stiffnes')
        self.add_output('damping',              val=np.zeros(n_TMDs), units='N/(m/s)',      desc='TMD Damping')

    def compute(self, inputs, outputs):
        '''
        Map TMD ivc groups to list of TMDs
        Someday, we'll do things like mapping the relative locations to global coords
        '''
        print('here')
        n_groups = self.modeling_options['TMDs']['n_TMDs']       # for now
        
        group_mapping = self.modeling_options['TMDs']['group_mapping']  
        # for each group
        for i_group in range(n_groups):
            # which turbines are in group i
            tmds_in_group = group_mapping[i_group]
            for i_TMD in tmds_in_group:
                
                # mass
                outputs['mass'][i_TMD] = inputs['group_mass'][i_group]

                # stiffness:
                if inputs['group_natural_frequency'][i_group] < 0:      # default is -1
                    outputs['stiffness'][i_TMD] = inputs['group_stiffness'][i_group]
                else:
                    outputs['stiffness'][i_TMD] = inputs['group_natural_frequency'][i_group]**2 * inputs['group_mass'][i_group]

                # damping:
                if inputs['group_damping_ratio'][i_group] < 0:      # default is -1
                    outputs['damping'][i_TMD] = inputs['group_damping'][i_group]
                else:
                    outputs['damping'][i_TMD] = 2 * inputs['group_damping_ratio'][i_group] * inputs['group_natural_frequency'][i_group] * inputs['group_mass'][i_group]

        print('here')  # to check mapping

class TMD_IVCs(IndepVarComp):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.opt_options = self.options['opt_options']

        n_TMDs = self.modeling_options['TMDs']['n_TMDs']
        n_groups = n_TMDs   # temporaray

        self.add_output('group_mass',               val=np.zeros(n_groups), units='kg',        desc='TMD Mass')
        self.add_output('group_stiffness',          val=np.zeros(n_groups), units='N/m',       desc='TMD Stiffnes')
        self.add_output('group_damping',            val=np.zeros(n_groups), units='N/(m/s)',   desc='TMD Damping')
        self.add_output('group_natural_frequency',  val=-np.ones(n_groups), units='rad/s',     desc='TMD natural frequency')
        self.add_output('group_damping_ratio',      val=-np.ones(n_groups),                    desc='TMD damping ratio')

