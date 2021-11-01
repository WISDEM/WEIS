# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
import pickle 
import numpy as np 
from scipy.interpolate import PchipInterpolator
from control import ss
import matplotlib.pyplot as plt

def ReverseIndices(FitInd,ValInd):
    
    nf = len(FitInd) 
    nv = len(ValInd)
    nl = nf + nv
    
    
    Fit_new = ValInd
    Val_new = []
    
    Fit_new = np.append(Fit_new,FitInd[-1])
    Fit_new = np.insert(Fit_new,0,FitInd[0])
    Val_new = FitInd[1:nf-1]
    
    return Fit_new,Val_new

def Get_indices(nl):
    if nl % 2:
        FitInd = np.arange(0,nl+1,2)
        ValInd = np.arange(1,nl-1,2)
    else:
        FitInd = np.arange(0,nl,2)
        FitInd = np.append(FitInd,nl-1)
        ValInd = np.arange(1,nl-1,2)
        
    return FitInd,ValInd

def Save_pickle(Aw,Bw,Cw,Dw,x_ops,u_ops,y_ops,u_h,DescStates,DescCntrlInpt,DescOutput,fname):
    
    ABCD= {
            'A' : Aw,
            'B' : Bw,
            'C' : Cw,
            'D' : Dw,
            'x_ops':x_ops,
            'u_ops':u_ops,
            'y_ops':y_ops,
            'u_h':u_h,
            'DescCntrlInpt' : DescCntrlInpt,
            'DescStates' : DescStates,
            'DescOutput' : DescOutput,
            'sim_index':0,
        }
    
    ABCD_list = [ABCD]
    
    with open(fname,'wb') as handle:
        pickle.dump(ABCD_list,handle)

def Hinf_error(Aw,Bw,Cw,Dw,u_h,DescCntrlInpt,DescOutput,FitInd,ValInd):
    
    nf = len(FitInd); nv = len(ValInd)
    nx,nu,nl = np.shape(Bw)
    ny,nu,nl = np.shape(Dw)
    
    A_fit = np.zeros((nf,nx,nx))
    B_fit = np.zeros((nf,nx,nu))
    C_fit = np.zeros((nf,ny,nx))
    D_fit = np.zeros((nf,ny,nu))
    
    w_fit = np.zeros((nf,))
    w_val = u_h[ValInd]
    
    for i in range(nf):
        
        k = FitInd[i]
        
        w_fit[i] = u_h[k]
        
        A_fit[i,:,:] = Aw[:,:,k]
        B_fit[i,:,:] = Bw[:,:,k]
        C_fit[i,:,:] = Cw[:,:,k]
        D_fit[i,:,:] = Dw[:,:,k]
        
        
    
    
    Out_ind = DescOutput.index('ED GenSpeed, (rpm)')
    In_ind = DescCntrlInpt.index('ED Extended input: collective blade-pitch command, rad')
    omega = np.logspace(-3,2,1000)
    
    N_interpolated = np.zeros((nv,))
    N_actual = np.zeros(nf,)
    
    A_pp = PchipInterpolator(w_fit,A_fit,axis = 0)
    Af_op = lambda w: A_pp(w)
    
    B_pp = PchipInterpolator(w_fit,B_fit,axis = 0)
    Bf_op = lambda w: B_pp(w)
    
    C_pp = PchipInterpolator(w_fit,C_fit,axis = 0)
    Cf_op = lambda w: C_pp(w)
    
    D_pp = PchipInterpolator(w_fit,D_fit,axis = 0)
    Df_op = lambda w: D_pp(w)
    
    for i in range(nv):
        
        wdx = ValInd[i]
        ws = w_val[i]
        
        A_act = Aw[:,:,wdx]
        B_act = Bw[:,:,wdx]
        C_act = Cw[:,:,wdx]
        D_act = Dw[:,:,wdx]
        
        A_int = Af_op(ws)
        B_int = Bf_op(ws) 
        C_int = Cf_op(ws)
        D_int = Df_op(ws)
        
        sys_act = ss(A_act,B_act[:,In_ind],C_act[Out_ind,:],D_act[Out_ind,In_ind])
        sys_int = ss(A_int,B_int[:,In_ind],C_int[Out_ind,:],D_int[Out_ind,In_ind])
        
        act_mag,act_phase,act_om = sys_act.freqresp(omega)
        int_mag,int_phase,int_om = sys_int.freqresp(omega)
        
        FR_act = np.squeeze(act_mag)*np.cos(act_phase) + 1j*np.squeeze(act_mag)*np.sin(act_phase)
        FR_int = np.squeeze(int_mag)*np.cos(int_phase) + 1j*np.squeeze(int_mag)*np.sin(int_phase)
        
        FR_diff = FR_act - FR_int 
        
        N_interpolated[i] = np.nanmax(np.abs(FR_diff))
        
    
    for i in range(nf):
        wdx = FitInd[i]
        ws = w_fit[i]
        
        A_act = Aw[:,:,wdx]
        B_act = Bw[:,:,wdx]
        C_act = Cw[:,:,wdx]
        D_act = Dw[:,:,wdx]
        
        A_int = Af_op(ws)
        B_int = Bf_op(ws) 
        C_int = Cf_op(ws)
        D_int = Df_op(ws)
        
        sys_act = ss(A_act,B_act[:,In_ind],C_act[Out_ind,:],D_act[Out_ind,In_ind])
        sys_int = ss(A_int,B_int[:,In_ind],C_int[Out_ind,:],D_int[Out_ind,In_ind])
        
        act_mag,act_phase,act_om = sys_act.freqresp(omega)
        int_mag,int_phase,int_om = sys_int.freqresp(omega)
        
        FR_act = np.squeeze(act_mag)*np.cos(act_phase) + 1j*np.squeeze(act_mag)*np.sin(act_phase)
        FR_int = np.squeeze(int_mag)*np.cos(int_phase) + 1j*np.squeeze(int_mag)*np.sin(int_phase)
        
        FR_diff = FR_act - FR_int 
        
        N_actual[i] = np.nanmax(np.abs(FR_diff))
        
    Wcombined = list(w_fit) + list(w_val)
    Hcombined = list(N_actual) + list(N_interpolated)
    
    Wcomb_ind = np.argsort(Wcombined)
    
    Wcombined = [Wcombined[idx] for idx in Wcomb_ind]
    Hcombined = [Hcombined[idx] for idx in Wcomb_ind]
    
    fig,ax = plt.subplots(1)
    ax.plot(Wcombined,Hcombined,'*-')
    
    return Hcombined
    

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

pkl_file = mydir + os.sep + "outputs" + os.sep +  "LinearTurbine_full.pkl" 

with open(pkl_file,'rb') as handle:
    ABCD_list = pickle.load(handle)
        
Aw = ABCD_list.A_ops
Bw = ABCD_list.B_ops
Cw = ABCD_list.C_ops 
Dw = ABCD_list.D_ops 

xw = ABCD_list.x_ops 
uw = ABCD_list.u_ops 
yw = ABCD_list.y_ops 

u_h = ABCD_list.u_h 

DescCntrlInpt = ABCD_list.DescCntrlInpt 
DescOutput = ABCD_list.DescOutput 
DescStates = ABCD_list.DescStates 

nl = len(u_h)

FitInd,ValInd = Get_indices(nl)

    
Hcombined1 = Hinf_error(Aw, Bw, Cw, Dw, u_h, DescCntrlInpt, DescOutput, FitInd, ValInd)

FitInd,ValInd = ReverseIndices(FitInd, ValInd)   

Hcombined2 = Hinf_error(Aw, Bw, Cw, Dw, u_h, DescCntrlInpt, DescOutput, FitInd, ValInd)    

Hcombined = np.array(Hcombined1) + np.array(Hcombined2)

fig,ax = plt.subplots(1)
ax.plot(u_h,Hcombined,'*-')

_ind = Hcombined < 6

Anew = Aw[:,:,_ind]
Bnew = Bw[:,:,_ind]
Cnew = Cw[:,:,_ind]
Dnew = Dw[:,:,_ind]

xw_new = xw[:,_ind]
uw_new = uw[:,_ind]
yw_new = yw[:,_ind]

u_new = u_h[_ind]

nl = len(u_new)


FitInd,ValInd = Get_indices(nl)

Hnew1 = Hinf_error(Anew, Bnew, Cnew, Dnew, u_new, DescCntrlInpt, DescOutput, FitInd, ValInd)

FitInd,ValInd = ReverseIndices(FitInd, ValInd)   

Hnew2 = Hinf_error(Anew, Bnew, Cnew, Dnew, u_new, DescCntrlInpt, DescOutput, FitInd, ValInd)  

Hnew = (np.array(Hnew1)+np.array(Hnew2))/2
fig,ax = plt.subplots(1)
ax.plot(u_new,Hnew,'*-')

Ind_fix = np.arange(23,27)

for i in range(nl):
    
    if i in Ind_fix :
        Anew[:,:,i] = Anew[:,:,23]
        Bnew[:,:,i] = Bnew[:,:,23]
        Cnew[:,:,i] = Cnew[:,:,23]
        Dnew[:,:,i] = Dnew[:,:,23]
        
        xw_new[:,i] = xw_new[:,23]
        uw_new[:,i] = uw_new[:,23]
        yw_new[:,i] = yw_new[:,23]
        

Hnew_fix1 = Hinf_error(Anew, Bnew, Cnew, Dnew, u_new, DescCntrlInpt, DescOutput, FitInd, ValInd)

FitInd,ValInd = ReverseIndices(FitInd, ValInd)   

Hnew_fix2 = Hinf_error(Anew, Bnew, Cnew, Dnew, u_new, DescCntrlInpt, DescOutput, FitInd, ValInd) 
Save_pickle(Anew, Bnew, Cnew, Dnew, xw_new, uw_new, yw_new, u_new, DescStates, DescCntrlInpt, DescOutput, 'LinModel_removed_fix.pkl')
#Fn,Vn = ReverseIndices(FitInd, ValInd)
    

    


