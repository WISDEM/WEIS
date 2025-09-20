'''
Update the DISCON.IN examples in the WEIS repository using the Tune_Case/ .yaml files

'''
import os
from rosco.toolbox.ofTools.fast_io.update_discons import update_discons


weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
iea15_dir = os.path.join(weis_dir,'examples/00_setup/OpenFAST_models/IEA-15-240-RWT/')

if __name__=="__main__":

    # {tune_yaml:discon}
    discon_map = {
        os.path.join(iea15_dir,'IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi_ROSCO.yaml'): os.path.join(iea15_dir,'IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi_DISCON.IN'),
        os.path.join(iea15_dir,'IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_ROSCO.yaml'): os.path.join(iea15_dir,'IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile_DISCON.IN'),
        # 'IEA15MW.yaml': 'IEA-15-240-RWT-UMaineSemi/DISCON-UMaineSemi.IN',
        # 'BAR.yaml': 'BAR_10/BAR_10_DISCON.IN'
    }


    # Make discons    
    update_discons(discon_map)
