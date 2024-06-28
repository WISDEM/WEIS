import os
import unittest

from weis.ss_fitting.SS_FitTools import SSFit_Radiation, WAMIT_Out, FDI_Fitting, SSFit_Excitation


this_dir = os.path.dirname(__file__)


class Test_SS_Fitting(unittest.TestCase):

    # def test_radiation(self):

    #     model_dir = os.path.join(this_dir,'OC3_Spar')
    #     fdi = FDI_Fitting(
    #         HydroFile = os.path.join(model_dir,'spar')
    #     )

    #     fdi.fit()
    #     fdi.outputMats()
    #     fig_list = fdi.visualizeFits()
        
    #     if not os.path.exists(os.path.join(model_dir,'rad_fit')):
    #         os.makedirs(os.path.join(model_dir,'rad_fit'))

    #     for i_fig, fig in enumerate(fig_list):
    #         fig.savefig(os.path.join(model_dir,'rad_fit',f'rad_fit_{i_fig}.png'))


    def test_excitation(self):

        model_dir = os.path.join(this_dir,'OC3_Spar')
        exctn_fit = SSFit_Excitation(
            HydroFile = os.path.join(model_dir,'spar')
        )

        exctn_fit.writeMats()

        if not os.path.exists(os.path.join(model_dir,'exctn_fit')):
            os.makedirs(os.path.join(model_dir,'exctn_fit'))

        for i_fig, fig in enumerate(exctn_fit.TDF.fig_list):
            fig.savefig(os.path.join(model_dir,'exctn_fit',f'exctn_fit{i_fig}.png'))






if __name__ == "__main__":
    unittest.main()
