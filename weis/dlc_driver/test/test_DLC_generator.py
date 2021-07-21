import unittest
from weis.dlc_driver.dlc_generator import DLCGenerator
import os
import weis.inputs as sch
import numpy as np

class TestIECWind(unittest.TestCase):

    def test_generator(self):

        # Wind turbine inputs that will eventually come in from somewhere
        cut_in = 4.
        cut_out = 25.
        rated = 10.
        wind_class = 'I'

        # Load modeling options file
        weis_dir                = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep
        fname_modeling_options = os.path.join(weis_dir , "examples", "05_IEA-3.4-130-RWT", "modeling_options.yaml")
        modeling_options = sch.load_modeling_yaml(fname_modeling_options)
        
        # Extract user defined list of cases
        DLCs = modeling_options['DLC_driver']['DLCs']
        
        # Initialize the generator
        dlc_generator = DLCGenerator(cut_in, cut_out, rated, wind_class)

        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)

        np.testing.assert_equal(dlc_generator.cases[11].URef, cut_out)
        np.testing.assert_equal(dlc_generator.n_ws_dlc11, 6)
        np.testing.assert_equal(dlc_generator.n_cases, 53)

if __name__ == "__main__":
    unittest.main()
