import numpy as np
import openmdao.api as om


def check(A, B):

    if np.iscomplex(B).any():
        B_check = np.abs(B)
    else:
        B_check = B

    if (np.linalg.norm(np.atleast_1d(A) - np.atleast_1d(B_check), ord=1) / np.linalg.norm(np.atleast_1d(B_check))) < 1e-6:
        return True
    return False


def test_yamls(results1, results2):

    Pass = True

    for key in results1['properties']:
        p = check(results1['properties'][key], results2['properties'][key])
        if not p:
            print('test failed, results differ for properties_'+key)
            Pass = False

    for key in results1['response']:
        p = check(results1['response'][key], results2['response'][key])
        if not p:
            print('test failed, results differ for response_'+key)
            Pass = False

    return Pass


def test(omdao_prob, results):

    Pass = True

    outs = omdao_prob.model.list_outputs(values=False, out_stream=None)
    for i in range(len(outs)):
        if outs[i][0].startswith('properties_'):
            name = outs[i][0].split('properties_')[1] 
            p = check(omdao_prob.get_val('properties_'+name), results['properties'][name])
            if not p:
                print('test failed, results differ for properties_'+name)
                Pass = False
        elif outs[i][0].startswith('response_'):
            name = outs[i][0].split('response_')[1]
            p = check(omdao_prob.get_val('response_'+name), results['response'][name])
            if not p:
                print('test failed, results differ for response_'+name)
                Pass = False

    return Pass
