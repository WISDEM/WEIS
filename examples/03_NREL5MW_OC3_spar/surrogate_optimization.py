import openmdao.api as om

# We'll use the component that was defined in the last tutorial
from openmdao.test_suite.components.paraboloid import Paraboloid

import os, glob
import matplotlib.pyplot as plt
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI
import numpy as np
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml
from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

from scipy.interpolate import Rbf
from numpy.random import Generator, PCG64

class Paraboloid_New(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = 1e-6 * ((x-3.0)**2 + x*y + (y+4.0)**2 - 3.0)

class WT_DOE(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        # Load DOE results
        folder_output   = '/Users/dzalkind/Tools/WEIS-4/optimizations/03_ballast_DOE/03_ballast_DOE/'
        data_out        = load_yaml(os.path.join(folder_output,'doe_summary.yaml'))
        omega           = np.array(data_out['rec_out']['tune_rosco_ivc.PC_omega'])
        vol_b           = np.array(data_out['rec_out']['floating.memgrp0.ballast_volume'])/1000

        # Filter Data
        use_ind = np.bitwise_and(np.array(vol_b) > .75, np.array(vol_b) < 1.5)

        self.sigma           = {}
        self.sigma['rotor_overspeed']   = 1e-4 #1e-3
        self.sigma['platform_mass']     = 0
        self.sigma['Max_PtfmPitch']     = 1e-3 #1e-2
        

        # Extract the data.
        x = omega[use_ind]
        y = vol_b[use_ind]
        z = np.array(data_out['rec_out']['aeroelastic.rotor_overspeed'])[use_ind]

        # Make an n-dimensional interpolator.
        self.rbfi_overspeed = Rbf(x, y, z)

        z = np.array(data_out['rec_out']['floatingse.platform_mass'])[use_ind]

        # Make an n-dimensional interpolator.
        self.rbfi_mass = Rbf(x, y, z)

        z = np.array(data_out['rec_out']['aeroelastic.Max_PtfmPitch'])[use_ind]

        # Make an n-dimensional interpolator.
        self.rbfi_max_pitch = Rbf(x, y, z)

        # declare inputs/outputs
        self.add_input('pc_omega', val=0.0)
        self.add_input('vol_b', val=0.0)

        self.add_output('overspeed', val=0.0)
        self.add_output('ptfm_mass', val=0.0)
        self.add_output('max_ptfm_pitch', val=0.0)


    def setup_partials(self):
        pass

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """

        # Inputs
        pc_omega = inputs['pc_omega']
        vol_b = inputs['vol_b'] / 1000

        # Random Seed
        seed = (1e6 * pc_omega * vol_b).astype(int)
        # seed = int(str(int(1e6 * pc_omega[0])) + str(int(1e6 * vol_b[0])))

        rg = Generator(PCG64(seed))
        
        # Predict on the regular grid.
        outputs['max_ptfm_pitch']   = self.rbfi_max_pitch(pc_omega, vol_b) + self.sigma['Max_PtfmPitch']  * rg.standard_normal()
        outputs['ptfm_mass']        = self.rbfi_mass(pc_omega, vol_b) + self.sigma['platform_mass']   * rg.standard_normal()
        outputs['overspeed']        = self.rbfi_overspeed(pc_omega, vol_b) + self.sigma['rotor_overspeed']   * rg.standard_normal()

        print('here')
        

    # def compute_partials(self, inputs, partials):
    #     """
    #     Jacobian for our paraboloid.
    #     """
    #     x = inputs['x']
    #     y = inputs['y']

    #     partials['f_xy', 'x'] = 2.0*x - 6.0 + y
    #     partials['f_xy', 'y'] = 2.0*y + 8.0 + x

# Test WT DOE
wt = WT_DOE()
# wt.setup()

# build the model
prob = om.Problem()
prob.model.add_subsystem('wt', WT_DOE(), promotes_inputs=['pc_omega', 'vol_b'])

# # define the component whose output will be constrained
# prob.model.add_subsystem('const', om.ExecComp('g = .001*(x + y)'), promotes_inputs=['x', 'y'])

# # Design variables 'x' and 'y' span components, so we need to provide a common initial
# # value for them.

optimizer = 'LN_COBYLA'
record = True
plot_record = True
warm_start = False
print_debug = True

# # setup the optimization
if optimizer == 'COBYLA':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'
    # prob.driver.options['optimizer'] = 'SNOPT'
# prob.driver.options['optimizer'] = "Nelder-Mead"
    # prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings["rhobeg"] = .1
    prob.driver.opt_settings["test"] = .01
    # prob.driver.opt_settings["maxfun"] = 1000
    prob.driver.opt_settings['maxiter'] = 1000
    prob.driver.opt_settings['tol'] = 1e-6
    prob.driver.opt_settings['disp'] = True
    prob.driver.opt_settings['catol'] = 2e-4

if optimizer == 'SLSQP':
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.opt_settings["maxfun"] = 1000
    prob.driver.opt_settings['maxiter'] = 100
    prob.driver.opt_settings['tol'] = 1e-3
    prob.driver.opt_settings['disp'] = True
    
    step_size = 1e-3
    prob.model.approx_totals(method="fd", step=step_size, form='forward')


    prob.model.set_input_defaults('pc_omega', .35)
    # prob.model.set_input_defaults('vol_b', 7500)
    prob.model.set_input_defaults('vol_b', 1.1)    
elif optimizer == 'GA':
    prob.driver = om.SimpleGADriver()
    prob.driver.options['bits'] = {'pc_omega':3,'vol_b':3}
    prob.driver.options['max_gen'] = 3
    prob.driver.options['penalty_parameter'] = 6e6
    prob.driver.options['penalty_exponent'] = 3.

elif optimizer == 'SNOPT':
    prob.driver = om.pyOptSparseDriver()
    # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9
    step_size = 1e-4
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.model.approx_totals(method="fd", step=step_size, form='forward')
    prob.driver.opt_settings["Major iterations limit"] = int(100)
    prob.driver.opt_settings['Nonderivative linesearch'] = None

elif optimizer == 'DIRECT':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'GN_ORIG_DIRECT'
    prob.driver.options['maxiter'] = 50
    # prob.driver.options['tol'] = 1e-6
    # prob.driver.options['xtol'] = 1e-3

elif optimizer == 'MMA':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LD_MMA'
    step_size = 1e-3
    prob.model.approx_totals(method="fd", step=step_size, form='forward')
    prob.driver.options['tol'] = 1e-3
    prob.driver.options['xtol'] = 1e-2

elif optimizer == 'ISRES':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'GN_ISRES'
    prob.driver.options['maxiter'] = 1000
    prob.driver.options['numgen'] = 10


elif optimizer == 'LN_COBYLA':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options['tol'] = 1e-3
    prob.driver.options['xtol'] = 1e-2
    print('here')
elif optimizer == 'MLSL':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options['tol'] = 1e-5
    print('here')
elif optimizer == 'AGS':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'GN_AGS'
    # prob.driver.options['tol'] = 1e-5
    print('here')
elif optimizer == 'CCSAQ':
    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LD_CCSAQ'
    step_size = 1e-3
    prob.model.approx_totals(method="fd", step=step_size, form='forward')
    prob.driver.options['tol'] = 1e-5
    prob.driver.options['xtol'] = 1e-3
    print('here')



# prob.driver.rhobeg = .2
# prob.driver.max_iter = 1000



prob.model.add_design_var('pc_omega', lower=0.1, upper=0.5)
# prob.model.add_design_var('vol_b', lower=5.5, upper=9)
prob.model.add_design_var('vol_b', lower=0.75e3, upper=1.5e3,ref=1e3)
prob.model.add_objective('wt.ptfm_mass',ref=1e3)

# # to add the constraint to the model
prob.model.add_constraint('wt.max_ptfm_pitch', upper=5.5)
prob.model.add_constraint('wt.overspeed', lower=0, upper=.2)

# add debug
if print_debug:
    prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs", "totals"]


prob.setup()

if record:
    recorder = om.SqliteRecorder(os.path.join('/Users/dzalkind/Tools/WEIS-4/optimizations/play', 'record.sql'))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)

    # prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

# Set initial conditions
# prob.set_val('pc_omega',0.35)
# prob.set_val('vol_b',6.5)
prob.set_val('pc_omega',0.35)
prob.set_val('vol_b',1100)

if warm_start:
    prob.set_val('pc_omega',0.21234841)
    prob.set_val('vol_b',1100)  

# prob.model.approx_totals(method='fd')
prob.run_driver()

# # minimum value
print('mass = ' + str(prob.get_val('wt.ptfm_mass')))
print('overspeed = ' + str(prob.get_val('wt.overspeed')))
print('max pitch = ' + str(prob.get_val('wt.max_ptfm_pitch')))
print('ballast = ' + str(prob.get_val('vol_b')))
print('omega_pc = ' + str(prob.get_val('pc_omega')))



if record and plot_record:  # set up recorder
    cr = om.CaseReader(os.path.join('/Users/dzalkind/Tools/WEIS-4/optimizations/play', 'record.sql'))

    cases = cr.list_cases()

    rec_data = {}
    design_vars = {}
    responses   = {}
    iterations = []
    for i, casei in enumerate(cases):
        iterations.append(i)
        it_data = cr.get_case(casei)

        # Collect DVs and responses separately for DOE
        for design_var in [it_data.get_design_vars()]:
            for dv in design_var:
                if i == 0:
                    design_vars[dv] = []
                design_vars[dv].append(design_var[dv])

        for response in [it_data.get_responses()]:
            for resp in response:
                if i == 0:
                    responses[resp] = []
                responses[resp].append(response[resp])

        # parameters = it_data.get_responses()
        # Collect all parameters for convergence plots
        for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
            for j, param in enumerate(parameters.keys()):
                if i == 0:
                    rec_data[param] = []
                rec_data[param].append(parameters[param])

    if True:
        for param in rec_data.keys():
            if param != "tower.layer_thickness" and param != "tower.diameter":
                fig, ax = plt.subplots(1, 1, figsize=(5.3, 4))
                ax.plot(iterations, rec_data[param])
                ax.set(xlabel="Number of Iterations", ylabel=param)
                fig_name = "Convergence_trend_" + param + ".png"
                # fig.savefig(os.path.join(folder_output, fig_name))
                # plt.close(fig)

        # Iteration plot
        circle_style = dict(marker='o',markersize=20,linestyle='none',markerfacecolor='white',mew=2,markeredgecolor='k')
        for dv in design_vars.keys():
            if dv != "tower.layer_thickness" and dv != "tower.diameter":
                fig, ax = plt.subplots(len(responses),1, figsize=(5, len(responses)*4))
                for i_resp, resp in enumerate(responses):  
                    for it, (d, r) in enumerate(zip(design_vars[dv],responses[resp])):
                        ax[i_resp].plot(d, r,**circle_style)
                        iter_style = dict(marker='$'+str(it)+'$',markersize=10,linestyle='none',markerfacecolor='k',markeredgecolor='k')
                        ax[i_resp].plot(d, r,**iter_style)
                        ax[i_resp].set(xlabel=dv,ylabel=resp)
                        ax[i_resp].grid(True)
                        fig_name = "DesignVar_iterations_" + dv + ".png"

        # 2D iterations
        dv1,dv2 = design_vars.keys()
        fig, ax = plt.subplots(1,1, figsize=(5, 4))
        for it, (d1, d2) in enumerate(zip(design_vars[dv1],design_vars[dv2])):
            ax.plot(d1, d2,**circle_style)
            iter_style = dict(marker='$'+str(it)+'$',markersize=10,linestyle='none',markerfacecolor='k',markeredgecolor='k')
            ax.plot(d1, d2,**iter_style)
            ax.set(xlabel=dv,ylabel=resp)
            ax.grid(True)
            fig_name = "DesignVar_2D.png"
            # fig.savefig(os.path.join(folder_output, fig_name),bbox_inches='tight')
            # plt.close(fig)

if plot_record:
    fig, ax = plt.subplots(1,1)
    ax.scatter(np.concatenate(design_vars['pc_omega']),np.concatenate(design_vars['vol_b']))

if plot_record:
    plt.show()