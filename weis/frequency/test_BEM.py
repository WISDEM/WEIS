import os
from weis import weis_main
import unittest
import shutil


class TestFrequency(unittest.TestCase):

    @unittest.skipUnless("RUN_EXHAUSTIVE" in os.environ, "exhaustive on pull request only")
    def test_BEM(self):
        ## File management
        run_dir = os.path.dirname( os.path.realpath(__file__) )
        fname_wt_input = os.path.join(run_dir, "..", "..", "examples", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
        fname_modeling_options = os.path.join(run_dir, "..", "..", "examples", "02_generate_openfast_model_for_dlcs" , "iea15_semi_modeling.yaml")
        fname_analysis_options = os.path.join(run_dir, "..", "..", "examples", "02_generate_openfast_model_for_dlcs", "iea15_semi_analysis.yaml")

        # override the modeling options so that not creating a new yaml file
        modeling_override = {}
        modeling_override["OpenFAST"] = {}
        modeling_override["OpenFAST"]["flag"] = False
        modeling_override["RAFT"] = {}
        modeling_override["RAFT"]["flag"] = True
        modeling_override["RAFT"]["potential_model_override"] = 1
        #modeling_override["RAFT"]["potential_bem_members"] = ["main_column", "column1", "column2", "column3", "Y_pontoon_lower1", "Y_pontoon_lower2", "Y_pontoon_lower3"]
        modeling_override["RAFT"]["intersection_mesh"] = 0
        modeling_override["RAFT"]["characteristic_length_min"] = 3
        modeling_override["RAFT"]["characteristic_length_max"] = 7
        modeling_override["RAFT"]["plot_designs"] = True
        modeling_override["RAFT"]["save_designs"] = True

        analysis_override = {}
        analysis_override["general"] = {}
        analysis_override["general"]["folder_output"] = os.path.join(run_dir, 'test_BEM_output')
        
        # remove old openfast_runs folder for consistency
        if os.path.exists(analysis_override["general"]["folder_output"]):
            shutil.rmtree(analysis_override["general"]["folder_output"])
                                

        wt_opt, modeling_options, opt_options = weis_main(fname_wt_input,
                                                        fname_modeling_options,
                                                        fname_analysis_options,
                                                        modeling_override=modeling_override,
                                                        analysis_override=analysis_override,
                                                        test_run=True
                                                        )
    


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFrequency)
    result = unittest.TextTestRunner().run(suite)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)

