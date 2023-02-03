import numpy as np
from scipy.linalg import eig
import os
import pickle
import matplotlib.pyplot as plt

# get path to file
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# path to pickle file
pkl_path = mydir + os.sep + 'outputs/l2_op/CD_sweep/analysis/lin/ABCD_matrices.pkl'

# load pickle file
with open(pkl_path,'rb') as handle:
    ABCD_matrices = pickle.load(handle)
    
    
A = []

for ABCD in ABCD_matrices:
    A.append(ABCD['A'])


A = np.squeeze(np.array(A))

nmat = len(ABCD_matrices)


eig_values = np.zeros((nmat,2))

for i in range(nmat):
    eig_mat = eig(A[i,:,:])
    
    eig_mat = eig_mat[0][0]
    
    eig_values[i,0] = np.real(eig_mat)
    eig_values[i,1] = np.imag(eig_mat)
    


plt.plot(eig_values[:,0],eig_values[:,1],'.',markersize = 16)