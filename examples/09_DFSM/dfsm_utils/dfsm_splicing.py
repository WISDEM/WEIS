import pickle
from copy import deepcopy
from numpy.linalg import eig

if __name__ == '__main__':


    dfsm_br = 'dfsm_fowt_1p6_br.pkl'
    dfsm_ar = 'dfsm_fowt_1p6_ar.pkl'

    with open(dfsm_br,'rb') as handle:
        dfsm_br = pickle.load(handle)

    with open(dfsm_ar,'rb') as handle:
        dfsm_ar = pickle.load(handle)

    splice_point = 14
    dfsm = deepcopy(dfsm_br)

    splice_ind = dfsm_br.W > splice_point

    dfsm.A_array[splice_ind,:,:] = dfsm_ar.A_array[splice_ind,:,:]
    dfsm.B_array[splice_ind,:,:] = dfsm_ar.B_array[splice_ind,:,:]
    dfsm.C_array[splice_ind,:,:] = dfsm_ar.C_array[splice_ind,:,:]
    dfsm.D_array[splice_ind,:,:] = dfsm_ar.D_array[splice_ind,:,:]


    with open('dfsm_fowt_1p6.pkl','wb') as handle:
        pickle.dump(dfsm,handle)

    eig_ar = eig(dfsm.A_array[6,:,:])
    eig_br = eig(dfsm_br.A_array[6,:,:])






    





