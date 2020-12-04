import unittest
import numpy as np
from weis.glue_code.runWEIS import run_weis
import os


class Test(unittest.TestCase):

    def test_ccblade(self):
        run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        fname_wt_input = run_dir + "../models/IEA-15-240-RWT_WISDEMieaontology4all.yaml"
        fname_analysis_options_ccblade = run_dir + "../models/modeling_options_ccblade.yaml"
        fname_opt_options = run_dir + "../models/analysis_options.yaml"
    
        # Run CCBlade
        wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_weis(
            fname_wt_input,
            fname_analysis_options_ccblade,
            fname_opt_options,
        )
        np.testing.assert_allclose(wt_opt_ccblade["ccblade.CP"], 0.472391)
        np.testing.assert_allclose(wt_opt_ccblade["ccblade.GenPwr"], 8932938.69112286)
        
    def test_openfast(self):
        run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        fname_wt_input = run_dir + "../models/IEA-15-240-RWT_WISDEMieaontology4all.yaml"
        fname_analysis_options_openfast = run_dir + "../models/modeling_options_openfast.yaml"
        fname_opt_options = run_dir + "../models/analysis_options.yaml"

        # Run OpenFAST
        wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_weis(
            fname_wt_input,
            fname_analysis_options_openfast,
            fname_opt_options,
        )
        np.testing.assert_allclose(wt_opt_openfast["aeroelastic.Cp_out"][0], 0.48263105453805283)
        np.testing.assert_allclose(wt_opt_openfast["aeroelastic.P_out"][0], 8974937.391731307)

if __name__ == '__main__':
    unittest.main()