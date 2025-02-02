import numpy as np

def calculate_channels(openfast_dict,fst_vt):
    # Add calculated channels to openfast_dict
    # Inputs: openfast_dict: dictionary of openfast timeseries for single simulation
    #         fst_vt: dictionary with fast variable tree
    #         someday add more to calculate strain, ect

    # Blade pitch rate
    for i_blade in range(fst_vt['ElastoDyn']['NumBl']):
        openfast_dict[f'dBldPitch{i_blade+1}'] = np.r_[0,np.diff(openfast_dict['BldPitch1'])] / fst_vt['Fst']['DT']

    # ADDED TO MAGNITUDE CHANNELS
    # Platform offset
    #openfast_dict['PtfmOffset'] = np.sqrt(openfast_dict['PtfmSurge']**2 + openfast_dict['PtfmSway']**2)
