import os
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd

def save_data(fname, prob, npz_file=True, mat_file=True, xls_file=True):
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Get all OpenMDAO inputs and outputs into a dictionary
    var_dict = prob.model.list_inputs(values=True, prom_name=False, units=True, out_stream=None)
    out_dict = prob.model.list_outputs(values=True, prom_name=False, units=True, out_stream=None)
    var_dict.extend( out_dict )

    # Pickle the full archive so that we can load it back in if we need
    with open(froot+'.pkl','wb') as f:
        pickle.dump(var_dict, f)

    # Reduce to variables we can save for matlab or python
    if npz_file or mat_file:
        array_dict = {}
        for k in range(len(var_dict)):
            unit_str = var_dict[k][1]['units']
            if unit_str is None or unit_str=='Unavailable':
                unit_str = ''
            elif len(unit_str) > 0:
                unit_str = '_' + unit_str

            iname = var_dict[k][0] + unit_str
            value = var_dict[k][1]['value']

            if type(value) in [type(np.array([])), type(0.0), type(0), np.float64, np.int64]:
                array_dict[iname] = value
            elif type(value) == type(True):
                array_dict[iname] = np.bool_(value)
            elif type(value) == type(''):
                array_dict[iname] = np.str_(value)
            elif type(value) == type([]):
                temp_val = np.empty(len(value), dtype=np.object)
                temp_val[:] = value[:]
                array_dict[iname] = temp_val
            #else:
            #    print(var_dict[k])

    # Save to numpy compatible
    if npz_file:
        kwargs = {key: array_dict[key] for key in array_dict.keys()}
        np.savez_compressed(froot+'.npz', **kwargs)

    # Save to matlab compatible
    if mat_file:
        sio.savemat(froot+'.mat', array_dict, long_field_names=True)

    if xls_file:
        data = {}
        data['variables'] = []
        data['units'] = []
        data['values'] = []
        for k in range(len(var_dict)):
            unit_str = var_dict[k][1]['units']
            if unit_str is None:  unit_str = ''

            data['variables'].append( var_dict[k][0] )
            data['units'].append( unit_str )
            data['values'].append( var_dict[k][1]['value'] )
        df = pd.DataFrame(data)
        df.to_excel(froot+'.xlsx')



def load_data(fname, prob):
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Load in the pickled data
    with open(froot+'.pkl','rb') as f:
        var_dict = pickle.load(f)

    # Store into Problem object
    for k in range(len(var_dict)):
        iname = var_dict[k][0]
        value = var_dict[k][1]['value']
        prob[iname] = value

    return prob
