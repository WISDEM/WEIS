import numpy as np
from weis.dtqpy.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod as multiprod
import time

if __name__ == '__main__':

    A1 = np.array([[lambda t: np.sin(t),0],[1,0]],dtype = object)
    A2 = np.array([[lambda t: np.exp(-t) +1,1],[0,0]],dtype = object)
    A3 = np.eye(2)

    t = np.arange(0,100000)
    matrices = np.array(['prod',A1,A2,A3],dtype = 'object')

    t1 = time.time()
    A = multiprod(matrices,[],t)
    t2 = time.time()

