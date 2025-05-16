# This test script is for running vizFileGen.py
# Use pytest-order to customize the order in which tests are run.
# Input vizFile Generation => Run the app with Input yaml file => Test WEIS Output Viz => Test WEIS Input Viz

import os
import subprocess
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
import dash_bootstrap_components as dbc

# Import all of the names of callback functions to tests
from weis import weis_main
from weis.visualization.appServer.app.mainApp import app        # Needed to prevent dash.exceptions.PageError: `dash.register_page()` must be called after app instantiation
from weis.visualization.appServer.app.pages.visualize_openfast import read_default_variables, define_graph_cfg_layout, save_openfast, update_graph_layout
from weis.visualization.appServer.app.pages.visualize_opt import read_variables, preprocess_data, define_preprocess_layout, complete_raft_sublayout, toggle_conv_layout, update_graphs, toggle_iteration_with_dlc_layout
from weis.visualization.utils import parse_yaml, convert_dict_values_to_list

# Input vizFile Generation
run_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( run_dir ) ) ) ) )
modeling_options = os.path.join(weis_dir, 'examples','04_frequency_domain_analysis_design','iea22_raft_opt_modeling.yaml')
analysis_options = os.path.join(weis_dir, 'examples','04_frequency_domain_analysis_design','iea22_raft_opt_analysis.yaml')
wt_input = os.path.join(weis_dir, 'examples','00_setup','ref_turbines','IEA-22-280-RWT_Floater.yaml')
vizFilepath = os.path.join(weis_dir, 'weis','visualization','appServer','app','tests','input','testIEA22RAFT.yaml')
vizExec = os.path.join(weis_dir, 'weis','visualization','appServer', 'share','vizFileGen.py')

global opt_options
opt_options = read_variables( parse_yaml(vizFilepath) )

def test_vizFile_generation(request):
    root_dir = request.config.rootdir    
    print(f'Moving back to root directory..{root_dir}\n')
    os.chdir(root_dir)
    subprocess.run(['python', vizExec, '--modeling_options', modeling_options, '--analysis_options', analysis_options, '--wt_input', wt_input, '--output', vizFilepath], cwd=root_dir)


# Optimization Visualization Test
def test_read_variables():
    assert opt_options['opt_type'] == 'RAFT'


def test_preprocess_data():

    def run_callback(trigger, nClickSQL, nClickRAFT, log_sql_path, raft_dir_path, log_data, prep_data):
        context_value.set(AttributeDict(**{"triggered_inputs": [trigger]}))
        return preprocess_data(nClickSQL, nClickRAFT, log_sql_path, raft_dir_path, log_data, prep_data)
    
    # Load SQL
    ctx = copy_context()
    output_sql = ctx.run(run_callback, trigger={'prop_id': 'load-sql.n_clicks'}, nClickSQL=1, nClickRAFT=0, log_sql_path=opt_options['log_file_path'], raft_dir_path=opt_options['raft_design_dir'], log_data={}, prep_data={'log_flag': False, 'raft_flag': False})
    assert output_sql[-1]['log_flag'] == True

    # For future reference..
    global df_dict
    df_dict = output_sql[-2]

    # Load RAFT Designs
    output_plot = ctx.run(run_callback, trigger={'prop_id': 'load-raft.n_clicks'}, nClickSQL=1, nClickRAFT=1, log_sql_path=opt_options['log_file_path'], raft_dir_path=opt_options['raft_design_dir'], log_data=df_dict, prep_data={'log_flag': True, 'raft_flag': False})
    assert output_plot[-1]['raft_flag'] == True
    # assert len(os.listdir(os.path.join(opt_options['raft_design_dir'], '..','raft_plots'))) > 0        # Make sure raft plots have been created... => With some reason, under pytest, plt.savefig doesn't work. Hence, disable test_toggle_iteration_with_dlc_layout() for now.


def test_define_preprocess_layout():
    output = define_preprocess_layout(opt_options)
    assert isinstance(output, dbc.Card)


def test_complete_raft_sublayout():
    output = complete_raft_sublayout(opt_options)
    assert output == (opt_options['raft_design_dir'], opt_options['raft_design_dir'])


def test_toggle_conv_layout():
    prep_data={'log_flag': True, 'raft_flag': True}
    output = toggle_conv_layout(prep_data, opt_options, df_dict)

    assert output[0] == True


def test_update_graphs():
    signaly = opt_options['conv_y']
    fig = update_graphs(signaly, [convert_dict_values_to_list(df_dict[0])])         # With some reason, its value types are not correct.. Need to correct value type for further unit test..

    assert len(fig['data']) == len(signaly)


def disable_test_toggle_iteration_with_dlc_layout():

    clickData_iteration_1 = {'points': [{'curveNumber': 0, 'pointNumber': 1, 'pointIndex': 1, 'x': 1, 'y': -25, 'bbox': {'x0': 284.5, 'x1': 290.5, 'y0': 379.89, 'y1': 385.89}}]}

    output = toggle_iteration_with_dlc_layout(clickData=clickData_iteration_1,
                                              prep_data={'log_flag': True, 'raft_flag': True}, 
                                              opt_options=opt_options, 
                                              x_chan_option=None, 
                                              y_chan_option=None, 
                                              x_channel=None, 
                                              y_channel=None)
    
    assert output[-2] == f'RAFT Optimization Iteration {clickData_iteration_1["points"][0]["x"]}'
