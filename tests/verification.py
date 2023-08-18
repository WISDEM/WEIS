# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:40:21 2021

@author: shousner
"""
import numpy as np
import yaml
import matplotlib.pyplot as plt

import raft

import os.path as osp
import hams.pyhams as ph





def plotAddedMass(fowtA, fowtB, A_wamit, w_wamit, A=[1,3,5,15,24]):
    if 1 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[0,0]]*len(fowtA.w)))
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[0,0,:], label='Surge-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[0,0,:]), label='Surge-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg]')
        plt.title('Surge Added Mass (A11) Comparison')
        plt.legend()
    if 3 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[2,2]]*len(fowtA.w)), label='Heave-Strip')
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[2,2,:], label='Heave-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[2,2,:]), label='Heave-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg]')
        plt.title('Heave Added Mass (A33) Comparison')
        plt.legend()
    if 5 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[4,4]]*len(fowtA.w)), label='Pitch-Strip')
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[4,4,:], label='Pitch-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[4,4,:]), label='Pitch-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg-m^2]')
        plt.title('Pitch Added Mass (A55) Comparison')
        plt.legend()
    if 6 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[5,5]]*len(fowtA.w)), label='Yaw-Strip')
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[5,5,:], label='Yaw-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[5,5,:]), label='Yaw-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg-m^2]')
        plt.title('Yaw Added Mass (A66) Comparison')
        plt.legend()
        
    if 15 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[0,4]]*len(fowtA.w)), label='Surge-Pitch-Strip')
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[0,4,:], label='Surge-Pitch-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[0,4,:]), label='Surge-Pitch-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg-m]')
        plt.title('Surge-Pitch Added Mass (A15) Comparison')
        plt.legend()
    if 24 in A:
        plt.figure()
        plt.plot(fowtA.w, np.array([fowtA.A_hydro_morison[1,3]]*len(fowtA.w)), label='Sway-Roll-Strip')
        plt.plot(fowtB.w_BEM, fowtB.A_BEM[1,3,:], label='Sway-Roll-HAMS')
        plt.plot(w_wamit, np.flip(A_wamit[1,3,:]), label='Sway-Roll-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Added Mass [kg-m]')
        plt.title('Sway-Roll Added Mass (A24) Comparison')
        plt.legend()

def plotDamping(fowtA, fowtB, B_wamit, w_wamit, B=[1,3,5,15,24]):
    if 1 in B:
        plt.figure()
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Surge-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[0,0,:], label='Surge-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[0,0,:]), label='Surge-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Damping [kg/s]')
        plt.title('Surge Damping (B11) Comparison')
        plt.legend()
    if 3 in B:
        plt.figure()
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Heave-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[2,2,:], label='Heave-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[2,2,:]), label='Heave-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Damping [kg/s]')
        plt.title('Heave Damping (B33) Comparison')
        plt.legend()
    if 5 in B:
        plt.figure()
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Pitch-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[4,4,:], label='Pitch-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[4,4,:]), label='Pitch-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Damping [kg-m^2/s]')
        plt.title('Pitch Damping (B55) Comparison')
        plt.legend()
    if 6 in B:
        plt.figure()
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Yaw-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[5,5,:], label='Yaw-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[5,5,:]), label='Yaw-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Damping [kg-m^2/s]')
        plt.title('Yaw Damping (B66) Comparison')
        plt.legend()
    
    if 15 in B or 24 in B:
        plt.figure()
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Surge-Pitch-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[0,4,:], label='Surge-Pitch-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[0,4,:]), label='Surge-Pitch-WAMIT')
        plt.plot(fowtA.w, np.array([0]*len(fowtA.w)), label='Sway-Roll-Strip')
        plt.plot(fowtB.w_BEM, fowtB.B_BEM[1,3,:], label='Sway-Roll-HAMS')
        plt.plot(w_wamit, np.flip(B_wamit[1,3,:]), label='Sway-Roll-WAMIT')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Damping [kg-m/s]')
        plt.title('Surge-Pitch, Sway-Roll Damping (B15/B24) Comparison')
        plt.legend()

def plotForcing(fowtA, fowtB, mod_wamit, phase_wamit, w_wamit, F=[1,3,5], mod=1, phase=1):
    
    # force units are N and Nm because the spectrum used was a unit amplitude spectrum
    
    if 1 in F:
        if mod==1:
            plt.figure()
            plt.plot(fowtA.w, abs(fowtA.F_hydro_iner[0,:]), label='Surge-Strip-no drag')
            #plt.plot(fowtA.w, abs(fowtA.F_hydro_iner[0,:]+fowtA.F_hydro_drag[0,:]), label='fowtA-strip')
            #plt.plot(fowtB.w, abs(fowtB.F_hydro_iner[0,:]), label='fowtB-strip')
            plt.plot(fowtB.w, abs(fowtB.F_BEM[0,:]), label='Surge-HAMS')
            #plt.plot(fowtA.w, abs(fowtA.F_BEM[0,:]), label='fowtA-BEM')
            plt.plot(w_wamit, np.flip(mod_wamit[0,:]), label='Surge-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Excitation Force [N]')
            plt.title('Surge Force Magnitude (F1) Comparison')
            plt.legend()
        if phase==1:
            plt.figure()
            plt.plot(fowtA.w, np.angle(fowtA.F_hydro_iner[0,:])*(180/np.pi), label='Surge-Strip-no drag')
            plt.plot(fowtB.w, np.angle(fowtB.X_BEM[0,:])*(180/np.pi), label='Surge-HAMS')
            plt.plot(w_wamit, np.flip(phase_wamit[0,:]), label='Surge-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Phase angle [deg]')
            plt.title('Surge Force Phase Angle (F1) Comparison')
            plt.legend()
        
    if 3 in F:
        if mod==1:
            plt.figure()
            plt.plot(fowtA.w, abs(fowtA.F_hydro_iner[2,:]), label='Heave-Strip-no drag')
            plt.plot(fowtB.w, abs(fowtB.F_BEM[2,:]), label='Heave-HAMS')
            plt.plot(w_wamit, np.flip(mod_wamit[2,:]), label='Heave-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Excitation Force [N]')
            plt.title('Heave Force Magnitude (F3) Comparison')
            plt.legend()
        if phase==1:
            plt.figure()
            plt.plot(fowtA.w, np.angle(fowtA.F_hydro_iner[2,:])*(180/np.pi), label='Heave-Strip-no drag')
            plt.plot(fowtB.w, np.angle(fowtB.X_BEM[2,:])*(180/np.pi), label='Heave-HAMS')
            plt.plot(w_wamit, np.flip(phase_wamit[2,:]), label='Heave-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Phase angle [deg]')
            plt.title('Heave Force Phase Angle (F3) Comparison')
            plt.legend()
        
    if 5 in F:
        if mod==1:
            plt.figure()
            plt.plot(fowtA.w, abs(fowtA.F_hydro_iner[4,:]), label='Pitch-Strip-no drag')
            plt.plot(fowtB.w, abs(fowtB.F_BEM[4,:]), label='Pitch-HAMS')
            plt.plot(w_wamit, np.flip(mod_wamit[4,:]), label='Pitch-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Excitation Moment [Nm]')
            plt.title('Pitch Force Magnitude (F5) Comparison')
            plt.legend()
        if phase==1:
            plt.figure()
            plt.plot(fowtA.w, np.angle(fowtA.F_hydro_iner[4,:])*(180/np.pi), label='Pitch-Strip-no drag')
            plt.plot(fowtB.w, np.angle(fowtB.X_BEM[4,:])*(180/np.pi), label='Pitch-HAMS')
            plt.plot(w_wamit, np.flip(phase_wamit[4,:]), label='Pitch-WAMIT')
            plt.xlabel('Frequency [rad/s]')
            plt.ylabel('Phase angle [deg]')
            plt.title('Pitch Force Phase Angle (F5) Comparison')
            plt.legend()
    
    
    
    
    ''' Plots to line up with OC3 report
    mag = abs(fowt.F_hydro_iner/fowt.zeta)
    
    plt.figure()
    plt.plot(fowtA.w, mag[0,:] ,'tab:pink')
    plt.plot(fowtA.w, mag[1,:], 'tab:cyan')
    plt.plot(fowtA.w, mag[2,:], 'lime')
    plt.plot(fowtB.w, fowtB.X_BEM[0,:])
    plt.plot(fowtB.w, fowtB.X_BEM[2,:])
    plt.plot(fowtB.w, fowtB.X_BEM[4,:])
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Exciting Force Magnitude [N/m]')
    plt.legend(['Surge','Sway','Heave'])
    plt.title('Wave excitation per unit amplitude - Translational')
    plt.grid()
    
    
    plt.figure()
    plt.plot(fowtA.w, mag[3,:] ,'tab:pink')
    plt.plot(fowtA.w, mag[4,:], 'tab:cyan')
    plt.plot(fowtA.w, mag[5,:], 'lime')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Exciting Moment Magnitude [Nm/m]')
    plt.legend(['Roll','Pitch','Yaw'])
    plt.title('Wave excitation per unit amplitude - Rotational')
    plt.grid()
    '''

    
    ''' Code to plot comparison between F_hydro_iner and F_hydro_iner+F_hydro_drag
    mag = abs(fowt.F_hydro_iner/fowt.zeta)
    mag2 = abs((fowt.F_hydro_iner + fowt.F_hydro_drag)/fowt.zeta)
    
    plt.figure()
    plt.plot(fowt.w, mag[0,:], label='F_hydro_iner')
    plt.plot(fowt.w, mag2[0,:], label='F_hydro_iner + F_hydro_drag')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Exciting Force Magnitude [N/m]')
    plt.title(f'Surge - dlsMax = {model.dlsMax}')
    plt.legend()
    '''

def getWamit1(rho=1025, file='spar.1'):
    runRAFTdir = osp.dirname(__file__)
    A_wamit, B_wamit, w_wamit = ph.read_wamit1B(osp.join(runRAFTdir,file), TFlag=1) # values are in reverse order
    A_wamit = A_wamit*rho
    B_wamit = B_wamit*rho
    return A_wamit, B_wamit, w_wamit
    
def getWamit3(heading=0, rho=1025, g=9.81, file='spar.3'):
    runRAFTdir = osp.dirname(__file__)
    mod, phase, real, imag, w, headings = ph.read_wamit3B(osp.join(runRAFTdir,file), TFlag=1)
    h_ind = list(headings).index(heading)
    mod_wamit = mod[h_ind,:,:]*rho*g        # in reverse order again
    phase_wamit = phase[h_ind,:,:] 
    w_wamit = w
    return mod_wamit, w_wamit, phase_wamit




def verify(name, a=1, b=1, f=1, mod=1, phase=1):
    if name=='OC3':
        filename='../designs/OC3spar.yaml'
        i = 0
        Adof = [1,3,5,15,24]
        Bdof = [1,3,5,15,24]
        Fdof = [1,3,5]
    elif name=='OC4':
        filename='../designs/OC4semi.yaml'
        i = 1
        Adof = [1,3,5,6,15,24]
        Bdof = [1,3,5,6,15,24]
        Fdof = [1,3,5]
    
    wamitfiles = [['spar.1', 'spar.3'],['marin_semi.1','marin_semi.3']]
    # get the values from the referenced wamit data to compare results to
    A_wamit, B_wamit, w_wamit = getWamit1(file=wamitfiles[i][0])
    mod_wamit, w_wamit, phase_wamit = getWamit3(file=wamitfiles[i][1])
    
    # read in the desired yaml file and set up initial variables
    with open(filename) as file:
        design = yaml.load(file, Loader=yaml.FullLoader)
    
    depth = float(design['mooring']['water_depth'])
    w = np.arange(0.05, 5, 0.05)
    
    # Create modelA where potModMaster is 1, so strip theory is used for all
    design['potModMaster'] = 1
    modelA = raft.Model(design, w=w, depth=depth)  # set up model
    modelA.setEnv()  # set basic wave and wind info
    modelA.calcSystemProps()          # get all the setup calculations done within the model
    
    # Create modelB where potModMaster is 2, so only BEM is used for all
    design['potModMaster'] = 2
    modelB = raft.Model(design, w=w, depth=depth)  # set up model
    modelB.setEnv()  # set basic wave and wind info
    modelB.calcSystemProps()          # get all the setup calculations done within the model
    
    fowtA = modelA.fowtList[0]
    fowtB = modelB.fowtList[0]
    
    if a==1:
        plotAddedMass(fowtA, fowtB, A_wamit, w_wamit, A=Adof)
    if b==1:
        plotDamping(fowtA, fowtB, B_wamit, w_wamit, B=Bdof) 
    if f==1:
        plotForcing(fowtA, fowtB, mod_wamit, phase_wamit, w_wamit, F=Fdof, mod=mod, phase=phase)


name = 'OC3'
#name = 'OC4'

verify(name)








