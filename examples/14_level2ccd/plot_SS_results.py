import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # get path to current directory
    mydir  = os.path.dirname(os.path.realpath(__file__))
    
    # get path to pickle file
    pkl_file = mydir + os.sep + 'DFSM_sensitivity_comparison_ED_KM_rated1000.pkl'
    
    # load pickle file
    with open(pkl_file,'rb') as handle:
        results = pickle.load(handle)
        
    # extract results
    nsamples_array = results['nsamples_array']
    TTrain_ED = results['TTrain_ED']
    TTrain_KM = results['TTrain_KM']
    T_kmeans = results['T_kmeans']
    
    fig,ax = plt.subplots(1)
    ax.plot(nsamples_array,TTrain_ED,'o-',label = 'ED')
    ax.plot(nsamples_array,TTrain_KM+T_kmeans,'o-',label = 'K-means')
    ax.set_xlabel('# of Training Points')
    ax.set_ylabel('Training Time [s]')
    ax.set_title('Comparison of Training Time')
    ax.set_xlim([nsamples_array[0],nsamples_array[-1]])
    ax.legend()
    
    MSE_dx1_ED_fit = results['MSE_dx1_ED_fit']
    MSE_dx1_KM_fit = results['MSE_dx1_KM_fit']
    MSE_dx2_ED_fit = results['MSE_dx2_ED_fit']
    MSE_dx2_KM_fit = results['MSE_dx2_KM_fit']
    
    MSE_dx1_ED_val = results['MSE_dx1_ED_val']
    MSE_dx1_KM_val = results['MSE_dx1_KM_val']
    MSE_dx2_ED_val = results['MSE_dx2_ED_val']
    MSE_dx2_KM_val = results['MSE_dx2_KM_val']
    
    SNE_fit_ED = results['SNE_fit_ED']
    SNE_fit_KM = results['SNE_fit_KM']
    
    SNE_val_ED = results['SNE_val_ED']
    SNE_val_KM = results['SNE_val_KM']
    
    #----------------------------------------------------------------
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    
    ax1.plot(nsamples_array,SNE_fit_ED,'o-',label = 'ED')
    ax1.plot(nsamples_array,SNE_fit_KM,'o-',label = 'K-means')
    ax1.set_xlim([nsamples_array[0],nsamples_array[-1]])
    
    #ax1.set_xlabel('# of training Points')
    ax1.set_ylabel('SNE')
    ax1.set_title('Fit-Trajectory')
    ax1.legend()
    
    ax2.plot(nsamples_array,SNE_val_ED,'o-',label = 'ED')
    ax2.plot(nsamples_array,SNE_val_KM,'o-',label = 'K-means')
    ax2.set_xlabel('# of training Points')
    ax2.set_ylabel('SNE')
    ax2.set_title('Validation-Trajectory')
    ax2.set_xlim([nsamples_array[0],nsamples_array[-1]])
    #ax2.legend()
    
    #---------------------------------------------------------
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    
    ax1.plot(nsamples_array,MSE_dx1_ED_fit,'o-',label = 'ED')
    ax1.plot(nsamples_array,MSE_dx1_KM_fit,'o-',label = 'K-means')
    
    #ax1.set_xlabel('# of training Points')
    ax1.set_ylabel('MSE')
    ax1.set_title('Fit')
    #ax1.legend()
    
    ax2.plot(nsamples_array,MSE_dx1_ED_val,'o-',label = 'ED')
    ax2.plot(nsamples_array,MSE_dx1_KM_val,'o-',label = 'K-means')
    ax2.set_xlabel('# of training Points')
    ax2.set_ylabel('MSE')
    ax2.set_title('Val')
    ax2.legend()
    
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace = 0.45)
    
    ax1.plot(nsamples_array,MSE_dx2_ED_fit,'o-',label = 'ED')
    ax1.plot(nsamples_array,MSE_dx2_KM_fit,'o-',label = 'K-means')
    
    #ax1.set_xlabel('# of training Points')
    ax1.set_ylabel('MSE')
    ax1.set_title('Fit')
    #ax1.legend()
    
    ax2.plot(nsamples_array,MSE_dx2_ED_val,'o-',label = 'ED')
    ax2.plot(nsamples_array,MSE_dx2_KM_val,'o-',label = 'K-means')
    ax2.set_xlabel('# of training Points')
    ax2.set_ylabel('MSE')
    ax2.set_title('Val')
    ax2.legend()
    
    
    
    
    
    
    # {'nsamples_array': nsamples_array,'TTrain_ED': TTrain_ED,
    #                 'TTrain_KM': TTrain_KM,'TEval_ED':TEval_ED,'TEval_KM':TEval_KM,
    #                 'T_kmeans':T_kmeans,'MSE_dx1_ED_fit':MSE_dx1_ED_fit,
    #                 'MSE_dx2_ED_fit':MSE_dx2_ED_fit, 'MSE_dx1_KM_fit': MSE_dx1_KM_fit,
    #                 'MSE_dx2_KM_fit':MSE_dx2_KM_fit,'MSE_dx1_ED_val':MSE_dx1_ED_val,
    #                 'MSE_dx2_ED_val':MSE_dx2_ED_val,'MSE_dx1_KM_val':MSE_dx1_KM_val,
    #                 'MSE_dx2_KM_val':MSE_dx2_KM_val}
    