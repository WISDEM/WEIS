#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:34:16 2021

@author: athulsun
"""

import os
import pickle 
import numpy as np 
import matplotlib.pyplot as plt

# get path to current file
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# get path to the dir where the linear models are stored
pkl_path = mydir + os.sep + "outputs"+os.sep + "EAB_sens" 

# path of the matrices
pkl_file = pkl_path  + os.sep +"ABCD_matrices.pkl" 

# load linear matrices
with open(pkl_file,'rb') as handle:
    ABCD_list = pickle.load(handle)
    
# get the number of cases
n_cases = len(ABCD_list)

# initialize new array
ABCD_new = []

# figures to plot
fig,ax = plt.subplots(1,1)
fig1,ax1 = plt.subplots(1,1)

nx,nu,nl = np.shape(ABCD_list[0]["B"])

nx_new = np.zeros((nx),dtype = 'bool')
nx_new[0:5] = True

# go through each set of linearized matrices and clip the ptfmpitch
for i in range(n_cases):
    # load matrices
    Aw = ABCD_list[i]["A"]
    Bw = ABCD_list[i]["B"]
    Cw = ABCD_list[i]["C"]
    Dw = ABCD_list[i]["D"]
    
    xw = ABCD_list[i]["x_ops"] 
    uw = ABCD_list[i]["u_ops"] 
    yw = ABCD_list[i]["y_ops"] 
    u_h = ABCD_list[i]["u_h"]
    
    DescCntrlInpt = ABCD_list[i]['DescCntrlInpt'] 
    DescOutput = ABCD_list[i]['DescOutput'] 
    DescStates = ABCD_list[i]['DescStates'] 
    
    # plot the unclipped PtfmPitch
    ax.plot(u_h,np.rad2deg(xw[0,:]))
    
    nl = len(u_h)
    ind_clip = np.rad2deg(xw[0,:]) > 6 
    
    ind_original = np.arange(nl)
    
    ind_fix = ind_original[ind_clip][0]-1
    
    
    for ind in range(nl):
        
        if ind_clip[ind]:
            
            Aw[:,:,ind] = Aw[:,:,ind_fix]
            Bw[:,:,ind] = Bw[:,:,ind_fix]
            Cw[:,:,ind] = Cw[:,:,ind_fix]
            Dw[:,:,ind] = Dw[:,:,ind_fix]
            
            xw[:,ind] = xw[:,ind_fix]
            
            
    ax1.plot(u_h,np.rad2deg(xw[0,:]))
    
    Aw = Aw[nx_new,:,:]; Aw = Aw[:,nx_new,:]
    Bw = Bw[nx_new,:,:]
    Cw = Cw[:,nx_new,:]
    
    xw = xw[nx_new,:]
    
    ABCD= {
            'A' : Aw,
            'B' : Bw,
            'C' : Cw,
            'D' : Dw,
            'x_ops':xw,
            'u_ops':uw,
            'y_ops':yw,
            'u_h':u_h,
            'DescCntrlInpt' : DescCntrlInpt,
            'DescStates' : DescStates,
            'DescOutput' : DescOutput,
            'sim_index':i,
        }
          
    ABCD_new.append(ABCD)
    
    
    
fname = pkl_path + os.sep + 'ABCD_CR.pkl'


with open(fname,'wb') as handle:
        pickle.dump([ABCD_new[2]],handle)
        


            
        
    
    
    
    