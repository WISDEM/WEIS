import os
import numpy as np
from weis.aeroelasticse import FileTools
from pCrunch.io import load_FAST_out
import fnmatch
import matplotlib.pyplot as plt
import pickle

def get_reqoutputs(opdir):
    
    # get the required outputs from the yaml file
    outops_ = FileTools.load_yaml(opdir + os.sep + 'OutOps.yaml')
    
    # initialize
    req_outputs = []
    
    # loop through and get the required outputs
    for key in outops_:
        req_outputs.append(key)
    
    # add TTDspFA
    req_outputs.append('TTDspFA')
    
    
    return req_outputs
    
    

# get path to this file 
mydir = os.path.dirname(os.path.realpath(__file__))  

# get path to folder where the outbsimulations are stored
opdir = mydir + os.sep + 'outputs' + os.sep + 'l3_op'

# find the number of .outb available
nfiles = len(fnmatch.filter(os.listdir(opdir),'*.outb'))

# get the list of required outputs
req_outputs = get_reqoutputs(opdir)

# get the number of outputs
noutputs = len(req_outputs)

# initialize
OutOps_matrix = np.zeros((noutputs,nfiles))

# loop through and get outputs
for ifile in range(nfiles):


    if nfiles > 10:
        
        filename = opdir + os.sep +'lin_' + str("{:02d}".format(ifile)) + '.outb'
        
    else:
        
         filename = opdir + os.sep +'lin_' + str(ifile) + '.outb'
            
    
    # load simulations
    simulation_data = load_FAST_out(filename)[0]
    
    # go though the simulation data and extract output
    for nop,output in enumerate(req_outputs):
        
        # consider the simulation from 300 s
        nt300 = simulation_data['Time']>300
        
        # store
        OutOps_matrix[nop,ifile] = np.mean(simulation_data[output][-1])

        

# initialize
OutOps = {}

# create OutOps dictionary
for nop, output in enumerate(req_outputs):
    OutOps[output] = (OutOps_matrix[nop,:])
    
# save yaml
FileTools.save_yaml(mydir,'OutOps.yaml',OutOps)

compareflag = 0

if compareflag:
    linfile1 = mydir + os.sep +'outputs' + os.sep + 'l2_op' + os.sep + 'analysis' + os.sep + 'lin' + os.sep + 'ABCD_matrices.pkl' 
    # load ABCD matrices
    with open(linfile1, 'rb') as handle:
        ABCD_matrices = pickle.load(handle)[0]
        
    y_ops1 = ABCD_matrices['y_ops']
    
    linfile2 = mydir + os.sep +'outputs' + os.sep + 'l2_op' + os.sep + 'analysis2' + os.sep + 'lin' + os.sep + 'ABCD_matrices.pkl' 
    # load ABCD matrices
    with open(linfile1, 'rb') as handle:
        ABCD_matrices = pickle.load(handle)[0]
        
    y_ops2 = ABCD_matrices['y_ops']