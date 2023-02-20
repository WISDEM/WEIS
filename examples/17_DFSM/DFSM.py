import os
import numpy as np
from pCrunch.io import load_FAST_out
from scipy.interpolate import PchipInterpolator,Rbf,CubicSpline
import matplotlib.pyplot as plt
from scipy.signal import filtfilt,butter
from scipy.integrate import solve_ivp
from numpy.linalg import lstsq,qr,inv,norm
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk
from sklearn.gaussian_process.kernels import ExpSineSquared
import time as time1
from sklearn.cluster import KMeans

def calculate_MSE(dx1,dx2):
    
    # number of samples
    nt = len(dx1)
    
    # MSE
    MSE = np.sum((dx1-dx2)**2)/nt
    
    return MSE

def calculate_SNE(dx1,dx2):
    
    # number of samples
    nt = len(dx1)
    
    SNE = np.zeros((nt,))
    
    #SNE
    for i in range(nt):
        SNE[i] = norm(dx1[i,:]-dx2[i,:])
        
    SNE = np.sum(SNE)
    
    return SNE
        
class SimulationDetails:
    
    def __init__(self,simulation_name = '',simulation_file = '',reqd_channels = [],controls_ind = [],
                 states_ind = [],filter_flag = True,add_dx2 = False,tmin = 0, tmax = None):
        
        self.simulation_name = simulation_name
        self.simulation_file = simulation_file
        self.reqd_channels = reqd_channels
        self.controls_ind = controls_ind 
        self.states_ind = states_ind
        self.filter_flag = filter_flag
        self.add_dx2 = add_dx2
        self.tmin = tmin
        self.tmax = tmax
        
    def filter_control(self,t_f,time,signal):

        '''
        Function to filter the given signal
        '''
        
        dt = time[1]-time[0]
        nb = int(np.floor(t_f/dt))
        b = np.ones((nb,))/nb;a = 1
        #b,a = butter(3,0.1)
        signal = filtfilt(b,a,signal,axis = 0)
        
        return signal
        

    def extract_channels(self):
        
        '''
        Function to extract required channels from the .outb file
        '''
        
        # name of the simulation file
        simulation_file = self.simulation_file
        
        # required channels
        reqd_channels = self.reqd_channels
        
        # state and controls ind
        controls_ind = self.controls_ind 
        states_ind = self.states_ind
        
        # filter flag
        filter_flag = self.filter_flag
        
        # load simulation file
        outputs = load_FAST_out(simulation_file)[0]
        
        # get the number of channels
        noutputs = len(reqd_channels)
    
        # get the number of time points
        time = outputs['Time']
        time = time - np.min(time)
        
        if not(self.tmax == None):
            t_ind = np.logical_and(time >= self.tmin,time <= self.tmax)
            
            time = time[t_ind]
            
            time = time-np.min(time)
            
        else:
            
            t_ind = [True]*len(time)
                
        
        # number of simulation points
        nt = len(time)
        
        # extract t0 and tf
        t0 = time[0]; tf = time[-1]
    
        # initialize
        channels = np.zeros((nt, noutputs))
        ind = 0
    
        # loop through and extract channels
        for chan in reqd_channels:
            channels[:, ind] = outputs[chan][t_ind]
            ind += 1
    
        # controls
        controls = channels[:,controls_ind]
        
        if len(controls_ind)>1:
            controls[:,1] = controls[:,1]/1e3 # scale the value of gen torque
        
        # # states
        states = channels[:,states_ind]
        
        # fig,(ax1,ax2) = plt.subplots(2,1)
        # fig.subplots_adjust(hspace = 0.45)
        # fig.suptitle('x')
        
        # ax1.plot(time,states[:,0],label = 'Original')
        # ax1.set_ylabel('PtfmPitch')
        # ax1.set_xlim([t0,tf])
        
        # ax2.plot(time,states[:,1],label = 'Original')
        # ax2.set_ylabel('GenSpeed')
        # ax2.set_xlim([t0,tf])
        # ax2.set_xlabel('Time')
        
        
        # filter   
        if filter_flag:  
                               
            t_f = 2 # 3 for belowrated
            controls[:,0] = self.filter_control(t_f,time,controls[:,0])
            
            t_f = 2 # 1 for below rated
            states = self.filter_control(t_f,time,states)
            
            # t_f = 2 # 4 for below rated
            # states[:,1] = self.filter_control(t_f,time,states[:,1])
            
        # ax1.plot(time,states[:,0],label = 'Filtered')
        # ax2.plot(time,states[:,1],label = 'Filtered')      
        
        # ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
        
        # construct polynomial approximation
        states_pp = CubicSpline(time,states)
        dx_pp = states_pp.derivative
        
        # evaluate first time derivative
        dx_pp1 = dx_pp(nu = 1)
        
        # evaluate second time derivative
        dx_pp2 = dx_pp(nu = 2)
        
        # evaluate state derivatives
        state_derivatives = dx_pp1(time)
        
        # evaluate the second time derivatives
        state_derivatives2 = dx_pp2(time)
        
        # # plot
        # fig,(ax1,ax2) = plt.subplots(2,1)
        # fig.subplots_adjust(hspace = 0.45)
        # fig.suptitle('d2x')
        
        # ax1.plot(time,state_derivatives2[:,0],label = 'Original')
        # ax1.set_ylabel('d2PtfmPitch')
        # ax1.set_xlim([t0,tf])
        
        # ax2.plot(time,state_derivatives2[:,1],label = 'Original')
        # ax2.set_ylabel('d2GenSpeed')
        # ax2.set_xlim([t0,tf])
        # ax2.set_xlabel('Time')
        
        # filter state derivatives
        # if filter_flag:
            
        #     t_f = 2
            
        #     state_derivatives = self.filter_signal(t_f,time,state_derivatives)
            
        #     t_f = 2
            
        #     state_derivatives2[:,0] = self.filter_signal(t_f,time,state_derivatives2[:,0])
            
        #     t_f = 4
            
        #     state_derivatives2[:,1] = self.filter_signal(t_f,time,state_derivatives2[:,1])
            
        # ax1.plot(time,state_derivatives2[:,0],label = 'Filtered')
        # ax2.plot(time,state_derivatives2[:,1],label = 'Filtered')      
        
        # ax2.legend(ncol = 3,bbox_to_anchor=(0.5, 1.3),loc = 'upper center')
        
        # flag to add second time derivatives
        if self.add_dx2:
            states = np.hstack([states,state_derivatives])
            state_derivatives = np.hstack([state_derivatives,state_derivatives2])

        # names of controls and states
        controls_name = reqd_channels[:controls_ind[-1]+1]
        states_name = reqd_channels[states_ind[0]:]
        
        # state derivative names
        state_dx_name = ['d' + sname for sname in states_name]
        
        # second time derivative name
        if self.add_dx2:
            state_dx2_name = ['d2' + sname for sname in states_name]
            states_name = states_name + state_dx_name
            state_dx_name = state_dx_name + state_dx2_name
            
            
        # return values
        self.time = time
        self.controls = controls 
        self.states = states 
        self.state_derivatives = state_derivatives 
        self.state_derivatives2 = state_derivatives2
        self.t0 = t0; self.tf = tf
        self.state_names = states_name;self.control_names = controls_name; self.state_dx_names = state_dx_name
        
    def provide_values(self):
        
        return self.t0,self.tf,self.time,self.controls,self.states,self.state_derivatives,self.state_derivatives2
    
    def provide_inputs_outputs(self):
        
        return self.controls,self.states,self.state_derivatives,self.time

def odefun(t,x,u_fun,surrogate_model,DFSM_option,mu_inputs,std_inputs,mu_outputs,std_outputs):
    
    '''
    Function to evaluate the derivative function for simulation
    '''
    
    # control inputs
    u = u_fun(t)
    
    # arrange states and inputs for evaluating DFSM
    input_ = np.hstack([x,u])
    input_ = input_.reshape([1,len(input_)])
    
    # scale
    input_ = (input_-mu_inputs)/std_inputs
    
    # evaluate DFSM according to DFSM_option:
    if DFSM_option == 'linfit':
        
        dx = np.dot(input_,surrogate_model)
        
    elif DFSM_option in ['RBF_smt','KPLSK_smt','KRG_smt']:
        nop = len(surrogate_model)
        
        # initialize
        dx = np.zeros((nop,))
        
        # loop through and evaluate
        for iop in range(nop):
            dx[iop] = surrogate_model[iop].predict_values(input_)
            
    elif DFSM_option == 'GPR_sk':
        
        dx = surrogate_model.predict(input_,return_std = False)
    
    # reshape    
    dx = np.squeeze(dx)

    # scale
    dx = dx*std_outputs + mu_outputs

    return dx
    
    

    
def reduce_samples(nt,SimulationDetails):
    
    # extract values
    t0,tf,time,controls,states,state_derivatives,state_derivatives2 = SimulationDetails.provide_values()
    
    # create reduced time grid
    time_reduced = np.linspace(t0,tf,nt)
    
    # interpolate controls on reduced mesh
    controls_ = PchipInterpolator(time,controls)
    controls_reduced = controls_(time_reduced)
    
    # interpolate states on reduced mesh
    states_ = PchipInterpolator(time,states)
    states_reduced = states_(time_reduced)
    
    # interpolate state_derivatives and second state derivatives on reduced mesh
    state_derivatives_ = PchipInterpolator(time,state_derivatives)
    state_derivatives_reduced = state_derivatives_(time_reduced)
    
    # dx2
    state_derivatives2_ = PchipInterpolator(time,state_derivatives2)
    state_derivatives2_reduced = state_derivatives2_(time_reduced)
    
    
    return time_reduced,controls_reduced,states_reduced,state_derivatives_reduced,state_derivatives2_reduced

    
class DFSM:
    
    def __init__(self,DFSM_option = 'GPR_sk',inputs = [],outputs = [],scale_inputs_flag = True,scale_outputs_flag = False):
        
        self.DFSM_option = DFSM_option
        self.inputs = inputs
        self.outputs = outputs 
        self.scale_inputs_flag = scale_inputs_flag
        self.scale_outputs_flag = scale_outputs_flag
        
    def scale_values(self):
        
        scale_inputs_flag = self.scale_inputs_flag
        scale_outputs_flag = self.scale_outputs_flag
        
        # scale 
        scale_flag = (scale_inputs_flag or scale_outputs_flag)
        inputs = self.inputs 
        outputs = self.outputs
        
        # get the shape
        nsamples,ninputs = np.shape(inputs)
        nsamples,noutputs = np.shape(outputs)


        if scale_flag:
        
            # calculate the mean and std of inputs and outputs
            
            # inputs
            if scale_inputs_flag:
                mu_inputs = np.mean(inputs,axis = 0)
                std_inputs = np.std(inputs,axis = 0)
                
                # scale 
                inputs = (inputs-mu_inputs)/std_inputs
                
            else:
                
                mu_inputs = np.zeros((ninputs,))
                std_inputs = np.ones((ninputs,))
                    
            
            # outputs
            if scale_outputs_flag:
                mu_outputs = np.mean(outputs,axis = 0)
                std_outputs = np.std(outputs,axis = 0)
            
                # scale
                outputs = (outputs-mu_outputs)/std_outputs
                
            else:
                
                mu_outputs = np.zeros((noutputs,))
                std_outputs = np.ones((noutputs,))
            
        else:
             mu_inputs = np.zeros((ninputs,))
             std_inputs = np.ones((ninputs,))
             
             mu_outputs = np.zeros((noutputs,))
             std_outputs = np.ones((noutputs,))

            
        self.inputs = inputs
        self.outputs = outputs 
        self.mu_inputs = mu_inputs 
        self.mu_outputs = mu_outputs
        self.std_inputs = std_inputs
        self.std_outputs = std_outputs 
        self.ninputs = ninputs
        self.noutputs = noutputs
        
    def construct_surrogate(self):
        
        DFSM_option = self.DFSM_option
        inputs = self.inputs 
        outputs = self.outputs
        ninputs = self.ninputs
        noutputs = self.noutputs
        
        if DFSM_option == 'linfit':
            
            # start timer
            t1 = time1.time()
            
            sm = lstsq(inputs,outputs)
            sm = sm[0]
            
            # end timer
            t2 = time1.time()
            
            # training time
            t_train = t2-t1
            
        if DFSM_option == 'linfit + corr':
            
            # construct DFSM with linear model plus corrective function
            
            # start timer
            t1 = time1.time()
            
            # create linear model
            lm = lstsq(inputs,outputs)
            lm = lm[0]
            
            # 
            
        
        elif DFSM_option == 'GPR_sk':
            
            # start timer
            t1 = time1.time()
            
            kernel = 1*RBFsk(length_scale = [1]*ninputs,length_scale_bounds=(1e-2, 1e2))
            sm = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 4,random_state = 34534)
            sm.fit(inputs,outputs)
            
            # end timer
            t2 = time1.time()
            
            # training time
            t_train = t2-t1
            
            # print('')
            # print('traning time: ' + str(t_train))
            # print('')
         
        self.sm = sm
        self.t_train = t_train
        
        
    def evaluate_surrogate(self,inputs):
        
        # extract
        mu_inputs = self.mu_inputs; std_inputs = self.std_inputs
        mu_outputs = self.mu_outputs; std_outputs = self.std_outputs
        ninputs = self.ninputs; noutputs = self.noutputs
        sm = self.sm
        
        # scaler
        inputs = (inputs-mu_inputs)/std_inputs
        

        # initialize storage
        size = np.shape(inputs)
        s0 = size[0]
        
        if len(size) == 1:
            
            # reshape
            inputs = inputs.reshape([1,ninputs])
            
            # predict value
            dx_ = sm.predict(inputs,return_std = False)
            
            # rescale
            dx_value = dx_*std_outputs + mu_outputs
            
            
            
        else:
            
            # initialize
            dx_value = np.zeros([s0,noutputs])
            
            for i in range(nsamples):
                
                # extract sample
                isample = inputs[i,:]
                
                # predict dx value
                dx_ = sm.predict(isample.reshape([1,ninputs]),return_std = False)
                
                # rescale
                dx_value[i,:] = dx_*std_outputs + mu_outputs
                
        return dx_value
        
        
        
    def evaluate_surrogate_performance(self,SimulationDetails,nt = 5000,plot_sd_flag = False,title = '',return_flag = False):
        
        # extract
        mu_inputs = self.mu_inputs; std_inputs = self.std_inputs
        mu_outputs = self.mu_outputs; std_outputs = self.std_outputs
        ninputs = self.ninputs; noutputs = self.noutputs
        sm = self.sm
        DFSM_option = self.DFSM_option
         
        # reduce samples
        time_,controls_,states_,state_derivatives_,state_derivative2_ = reduce_samples(nt,SimulationDetails)
        #breakpoint()
        #controls_[:,1] = controls_[:,1]/1e3
        t0 = time_[0]; tf = time_[-1]
        
        # create inputs
        input_test = np.hstack([states_,controls_])
        input_test = (input_test-mu_inputs)/std_inputs
        
        # initialize
        dx_gpr = np.zeros((nt,noutputs))
        std_gpr = np.zeros([nt,noutputs])
        
        t1 = time1.time()
        for i in range(nt):
            ip_test = input_test[i,:]
            ip_test = ip_test.reshape([1,ninputs])
            
            dx_,std_ = sm.predict(ip_test,return_std = True)
            
            # reshape    
            dx_ = np.squeeze(dx_)
        
            # scale
            dx_gpr[i,:] = dx_*std_outputs + mu_outputs
            std_gpr[i,:] = std_
        t2 = time1.time()
        
        # evaluation time
        t_eval = t2-t1
        
        # print('')
        # print('evaluation time:'+ str(t_eval))
        # print('')
        
        # plot the results    
        fig,(ax1,ax2) = plt.subplots(2,1)
        fig.subplots_adjust(hspace = 0.45)
    
        
        ax1.plot(time_,dx_gpr[:,0],'r-',label = DFSM_option)
        ax1.plot(time_,state_derivatives_[:,0],'k',label = 'OF')
        if plot_sd_flag:
            ax1.fill_between(time_.ravel(),dx_gpr[:,0]-1.96*std_gpr[:,0],dx_gpr[:,0]+1.96*std_gpr[:,0],alpha = 0.7)
        ax1.set_xlim([t0,tf])
        ax1.set_title(title)
        ax1.set_ylabel('dPtfmPitch')
        #ax1.legend()
        
        ax2.plot(time_,dx_gpr[:,1],'r-',label = DFSM_option)
        ax2.plot(time_,state_derivatives_[:,1],'k',label = 'OF')
        if plot_sd_flag:
            ax2.fill_between(time_.ravel(),dx_gpr[:,1]-1.96*std_gpr[:,1],dx_gpr[:,1]+1.96*std_gpr[:,1],alpha = 0.7,label = '95% confidence')
        ax2.set_ylabel('dGenSpeed')
        ax2.set_xlim([t0,tf])
        ax2.legend(loc = 'lower right',ncol = 3)
        
        if return_flag:
            return state_derivatives_,dx_gpr,t_eval
        

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    datapath = mydir + os.sep + 'outputs'
    
    # .outb file and pickle file
    sim_file = datapath + os.sep + '1p1_samples_rated' + os.sep + 'lin_0.outb'
    
        # required channels
    reqd_channels = ['RtVAvgxh', 'GenTq',
                     'BldPitch1', 'PtfmPitch', 'GenSpeed']
    control_ind = [0,1,2]
    state_ind = [3,4]
    
    # create class instance
    training_samples = SimulationDetails(simulation_name = 'training',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind)
    
    # extract values
    training_samples.extract_channels()
    
    # input for training the DFSM
    controls,states,state_derivatives = training_samples.provide_inputs_outputs()
    inputs = np.hstack([states,controls])
    
    # number of dimensions
    nt,ninputs = np.shape(inputs)
    nt,noutputs = np.shape(state_derivatives)
    
    # load validation file
    sim_file = datapath + os.sep + '1p1_samples_rated'+ os.sep + 'lin_1.outb'
    
    # create class instance
    validation_samples = SimulationDetails(simulation_name = 'validation',
                                       simulation_file = sim_file,reqd_channels = reqd_channels,
                                       controls_ind = control_ind,states_ind = state_ind)
    
    # extract values
    validation_samples.extract_channels()
    
    
    # number of samples
    nsamples_array = [400] #[10,20,30,40,50,60,70,80,90,100,200,300,400,500]
    ntests = len(nsamples_array)
    
    # initialize storage arrays
    TTrain_ED = np.zeros((ntests,))
    TTrain_KM = np.zeros((ntests,))
    TEval_ED = np.zeros((ntests,))
    TEval_KM = np.zeros((ntests,))
    T_kmeans = np.zeros((ntests,))
    MSE_dx1_ED_fit = np.zeros((ntests,))
    MSE_dx2_ED_fit = np.zeros((ntests,))
    MSE_dx1_KM_fit = np.zeros((ntests,))
    MSE_dx2_KM_fit = np.zeros((ntests,))
    MSE_dx1_ED_val = np.zeros((ntests,))
    MSE_dx2_ED_val = np.zeros((ntests,))
    MSE_dx1_KM_val = np.zeros((ntests,))
    MSE_dx2_KM_val = np.zeros((ntests,))
    
    SNE_fit_ED = np.zeros((ntests,))
    SNE_fit_KM = np.zeros((ntests,))
    
    SNE_val_ED = np.zeros((ntests,))
    SNE_val_KM = np.zeros((ntests,))
    
    
    # loop through and evaluate 
    
    for i,nsamples in enumerate(nsamples_array):
        
        print('')
        print(nsamples)
        print('')
        
        # get training points from ED mesh
        time_ed,controls_ed,states_ed,state_derivatives_ed,state_derivatives2_ed = reduce_samples(nsamples,training_samples)
        inputs_ed = np.hstack([states_ed,controls_ed])
        
        # get the inputs using the kmeans clustering algorithm
        nclusters = nsamples
        
        # stater timer
        t1 = time1.time()
        
        # perform kmeans clustering algorithm
        kmeans = KMeans(n_clusters = nclusters,n_init = 10).fit(X = inputs)
        
        # extract centroid centers and centroid index
        inputs_km = kmeans.cluster_centers_
        labels_ = kmeans.labels_
        
        # inititalize output array
        outputs_km = np.zeros([nclusters,noutputs])
        
        # loop through and find the state derrivative values at the cluster centroids
        for icluster in range(nclusters):
            
            # index of elements of the original array present in the current cluster
            cluster_ind = (labels_ == icluster)
            
            # extract the sate derivative values at these indices
            outputs_ind = state_derivatives[cluster_ind,:]
            
            # evaluate the state derivative values at the centroid of this cluster
            outputs_km[icluster,:] = np.mean(outputs_ind,axis=0)
    
        
        # stop timer
        t2 = time1.time()
        
        # time to get input-output pairs using kmeans algorithm
        T_kmeans[i] = t2-t1
        
        # construct surrogate model with the equidistant points   
        DFSM_ed = DFSM(inputs = inputs_ed,outputs = state_derivatives_ed)
        DFSM_ed.scale_values()
        DFSM_ed.construct_surrogate()
        
        # store training time
        TTrain_ED[i] = DFSM_ed.t_train
        
        
        # construct surrogate with points identified from k-means centroids
        DFSM_km = DFSM(inputs = inputs_km,outputs = outputs_km)
        DFSM_km.scale_values()
        DFSM_km.construct_surrogate()
        
        # store training time
        TTrain_KM[i] = DFSM_km.t_train
        
        nt_eval = 500
        # evaluate the surrogate models using the fitting values
        dx_act_ed,dx_gpr_ed,t_eval_ed1 = DFSM_ed.evaluate_surrogate_performance(training_samples,nt = nt_eval,title = 'Fiting-ED-'+str(nsamples),return_flag = True)
        dx_act_km,dx_gpr_km,t_eval_km1 = DFSM_km.evaluate_surrogate_performance(training_samples,nt = nt_eval,title = 'Fiting-Kmeans-'+str(nsamples),return_flag = True)
        
        # calculate MSE for both the states
        MSE_dx1_ED_fit[i] = calculate_MSE(dx_act_ed[:,0],dx_gpr_ed[:,0])
        MSE_dx2_ED_fit[i] = calculate_MSE(dx_act_ed[:,1],dx_gpr_ed[:,1])
        
        MSE_dx1_KM_fit[i] = calculate_MSE(dx_act_km[:,0],dx_gpr_km[:,0])
        MSE_dx2_KM_fit[i] = calculate_MSE(dx_act_km[:,1],dx_gpr_km[:,1])
        
        # calculate SNE
        SNE_fit_ED[i] = calculate_SNE(dx_act_ed,dx_gpr_ed)
        SNE_fit_KM[i] = calculate_SNE(dx_act_km,dx_gpr_km)
        

        # evaluate the validation points with both surrogate models
        dx_val_ed,dx_gpr_ed,t_eval_ed2 = DFSM_ed.evaluate_surrogate_performance(validation_samples,nt = nt_eval,title = 'Validation-ED-'+str(nsamples),return_flag = True)
        dx_val_km,dx_gpr_km,t_eval_km2 = DFSM_km.evaluate_surrogate_performance(validation_samples,nt = nt_eval,title = 'Validation-Kmeans-'+str(nsamples),return_flag = True)
        
        # calculate MSE for both the states
        MSE_dx1_ED_val[i] = calculate_MSE(dx_val_ed[:,0],dx_gpr_ed[:,0])
        MSE_dx2_ED_val[i] = calculate_MSE(dx_val_ed[:,1],dx_gpr_ed[:,1])
        
        MSE_dx1_KM_val[i] = calculate_MSE(dx_val_km[:,0],dx_gpr_km[:,0])
        MSE_dx2_KM_val[i] = calculate_MSE(dx_val_km[:,1],dx_gpr_km[:,1])
        
        # calculate SNE
        SNE_val_ED[i] = calculate_SNE(dx_val_ed,dx_gpr_ed)
        SNE_val_KM[i] = calculate_SNE(dx_val_km,dx_gpr_km)
        
        
        TEval_ED[i] = np.mean([t_eval_ed1,t_eval_ed2])
        TEval_KM[i] = np.mean([t_eval_km1,t_eval_km2])
        
    # save results
    
    results_dict = {'nsamples_array': nsamples_array,'TTrain_ED': TTrain_ED,
                    'TTrain_KM': TTrain_KM,'TEval_ED':TEval_ED,'TEval_KM':TEval_KM,
                    'T_kmeans':T_kmeans,'MSE_dx1_ED_fit':MSE_dx1_ED_fit,
                    'MSE_dx2_ED_fit':MSE_dx2_ED_fit, 'MSE_dx1_KM_fit': MSE_dx1_KM_fit,
                    'MSE_dx2_KM_fit':MSE_dx2_KM_fit,'MSE_dx1_ED_val':MSE_dx1_ED_val,
                    'MSE_dx2_ED_val':MSE_dx2_ED_val,'MSE_dx1_KM_val':MSE_dx1_KM_val,
                    'MSE_dx2_KM_val':MSE_dx2_KM_val,'SNE_fit_ED':SNE_fit_ED,'SNE_fit_KM':SNE_fit_KM,
                    'SNE_val_ED':SNE_val_ED,'SNE_val_KM':SNE_val_KM}   
    
    pkl_name = mydir + os.sep + 'DFSM_sensitivity_comparison_ED_KM_rated' + str(nt_eval) +'.pkl'
    
    # with open(pkl_name,'wb') as handle:
    #     pickle.dump(results_dict,handle)
    # input_names = state_names+control_names
    
    # ninputs = len(input_names)
    
    # for i in range(ninputs):
    #     fig,ax = plt.subplots(1)
    #     ax.hist(inputs_ed[:,i],label = 'ED',alpha = 0.6)
    #     ax.hist(inputs_km[:,i],label = 'Kmeans',alpha = 0.6)
    #     ax.set_title(input_names[i])
    #     ax.legend()
    
        
    # plot_GPR = True
    # if plot_GPR:    
    # # load second simulation
     
    #     # .outb file and pickle file
    #     sim_file = datapath + os.sep + '1p1_samples_transition'+ os.sep + 'lin_1.outb'
        
    #     # load simulations
    #     outputs = load_FAST_out(sim_file)[0]
        
    #     # extract results
    #     time_val,t0,tf,controls_val,states_val,state_derivatives_val,state_derivatives2_val = extract_and_process_sim(outputs,reqd_channels,control_ind,state_ind)
        
        
    #     fig,ax = plt.subplots(1)
    #     ax.plot(time_of,controls_of[:,0],label = 'Fit')
    #     ax.plot(time_val,controls_val[:,0],label = 'Val')
    #     ax.set_xlabel('Time');ax.set_ylabel('Wind Speed')
    #     ax.legend()
           
    
        
    #     # reduce samples
    #     nt = 500
    #     time_, controls_, states_, state_derivatives_, state_derivative2_ = reduce_samples(
    #         nt, t0, tf, time_val, controls_val, states_val, state_derivatives_val, state_derivatives2_val)
    
    #     # create inputs
    #     input_test = np.hstack([states_, controls_])
    #     input_test = (input_test-mu_inputs)/std_inputs
    
    #     # initialize
    #     dx_gpr = np.zeros((nt, noutputs))
    #     std_gpr = np.zeros([nt,noutputs])
    #     for i in range(nt):
    #         ip_test = input_test[i, :]
    #         ip_test = ip_test.reshape([1, len(ip_test)])
    
    #         dx_ = sm.predict(ip_test, return_std=False)
    
    #         # reshape
    #         dx_ = np.squeeze(dx_)
    
    #         # scale
    #         dx_gpr[i, :] = dx_*std_outputs + mu_outputs
    
    #     # plot the results
    #     fig,(ax1,ax2) = plt.subplots(2,1)
    #     fig.subplots_adjust(hspace = 0.45)
    
        
    #     ax1.plot(time_,dx_gpr[:,0],'r-',label = DFSM_option)
    #     ax1.plot(time_,state_derivatives_[:,0],'k',label = 'OF')
    #     ax1.fill_between(time_.ravel(),dx_gpr[:,0]-1.96*std_gpr[:,0],dx_gpr[:,0]+1.96*std_gpr[:,0],alpha = 0.7)
    #     ax1.set_xlim([t0,tf])
    #     ax1.set_title('dPtfmPitch')
    #     #ax1.legend()
        
    #     ax2.plot(time_,dx_gpr[:,1],'r-',label = DFSM_option)
    #     ax2.plot(time_,state_derivatives_[:,1],'k',label = 'OF')
    #     ax2.fill_between(time_.ravel(),dx_gpr[:,1]-1.96*std_gpr[:,1],dx_gpr[:,1]+1.96*std_gpr[:,1],alpha = 0.7)
    #     ax2.set_title('dGenSpeed')
    #     ax2.set_xlim([t0,tf])
        
       
    #     #ax2.legend()
    #     fig,(ax1,ax2) = plt.subplots(2,1)
    #     fig.subplots_adjust(hspace = 0.45)
        
    #     ax1.plot(controls[:,0],state_derivatives[:,0],'.',label = 'fit')
    #     ax1.plot(controls_[:,0],state_derivatives_[:,0],'.',label = 'val')
    #     ax1.set_title('windspeed vs dPtfmPitch')
    #     ax1.legend()
        
    #     ax2.plot(controls[:,0],state_derivatives[:,1],'.',label = 'fit')
    #     ax2.plot(controls_[:,0],state_derivatives_[:,1],'.',label = 'val')
    #     ax2.set_title('windspeed vs dGenSpeed')
    #     ax2.legend() 
    
    # breakpoint()
    # functon for controls
    # u_pp = PchipInterpolator(time_of,controls_of)
    # u_fun = lambda t: u_pp(t)
    
    # #tf = tf
    # # time span
    # tspan = [t0,tf]
    
    # # initial states
    # x0 = states_of[0,:]
    
    # # solve
    # sol = solve_ivp(odefun,tspan,x0,method='RK45',args = (u_fun,sm,DFSM_option,mu_inputs,std_inputs,mu_outputs,std_outputs))
    
    # # extract solution
    # T = sol['t']
    # X = sol['y']; X = X.T
    
    # '''
    # Plot to visualize the predictive capabilities of DFSM
    # '''
    
    # # plot simulation results and compare to original
    # fig,(ax1,ax2,ax3) = plt.subplots(3,1)
    # fig.subplots_adjust(hspace = 0.65)
    
    # ax1.plot(time_of,controls_of[:,0])
    # ax1.set_title('Wind Speed [m/s]')
    # ax1.set_xlim([t0,tf])
    
    # ax2.plot(time_of,states_of[:,0],'k',label = 'OpenFAST')
    # ax2.plot(T,X[:,0],'r',label = DFSM_option)
    # ax2.set_title('PtfmPitch [deg]')
    # ax2.set_xlim([t0,tf])
    # #ax2.legend()
    
    # ax3.plot(time_of,states_of[:,1],'k',label = 'OpenFAST')
    # ax3.plot(T,X[:,1],'r',label = DFSM_option)
    # ax3.set_title('GenSpeed [rpm]')
    # ax3.set_xlim([t0,tf])
    # ax3.legend(loc = 'lower right',ncol = 2)
    
        # # plot states and controls
    # fig,ax = plt.subplots(3,2)
    # fig.subplots_adjust(hspace = 0.65)
    
    # ax[0,0].plot(time,controls[:,0])
    # ax[0,0].set_title('RtVAvghx [m/s]')
    # ax[0,0].set_xlim([t0,tf])
    
    # ax[1,0].plot(time,controls[:,1])
    # ax[1,0].set_title('GenTq [kWm]')
    # ax[1,0].set_xlim([t0,tf])
    
    # ax[2,0].plot(time,controls[:,2])
    # ax[2,0].set_title('BldPitch [deg]')
    # ax[2,0].set_xlim([t0,tf])
    
    # ax[0,1].plot(time,states[:,0])
    # ax[0,1].set_title('PtfmPitch [deg]')
    # ax[0,1].set_xlim([t0,tf])
    
    # ax[1,1].plot(time,states[:,1])
    # ax[1,1].set_title('GenSpeed [rad/s]')
    # ax[1,1].set_xlim([t0,tf])
    
    
    
    
    # # plot state derivatives
    # fig,ax = plt.subplots(2,2)
    # fig.subplots_adjust(hspace = 0.65)
    
    # ax[0,0].plot(time,states[:,0])
    # ax[0,0].set_title('PtfmPitch [deg]')
    # ax[0,0].set_xlim([t0,tf])
    
    # ax[1,0].plot(time,state_derivatives[:,0])
    # ax[1,0].set_title('dPtfmPitch')
    # ax[1,0].set_xlim([t0,tf])
    
    # ax[0,1].plot(time,states[:,1])
    # ax[0,1].set_title('GenSpeed [rpm]')
    # ax[0,1].set_xlim([t0,tf])
    
    # ax[1,1].plot(time,state_derivatives[:,1])
    # ax[1,1].set_title('dGenSpeed')
    # ax[1,1].set_xlim([t0,tf])
    
    
    # # plot state space
    # fig,ax = plt.subplots(1)
    # ax.plot(states[:,0],states[:,1],'.')
    # ax.set_ylabel('GenSpeed')
    # ax.set_xlabel('PtfmPitch')
    
    # # correlation against wind
    
    # # sort
    # wind = np.sort(controls[:,0])
    # ind = np.argsort(controls[:,0])
    
    # # sorted values
    # bladepitch = controls[:,2][ind]
    # gentq = controls[:,1][ind]
    # ptfmpitch = states[:,0][ind]
    # genspeed = states[:,1][ind]
    
    # '''
    # Plot to visualize how the states and controls vary with wind speed
    # '''
    # fig,ax = plt.subplots(2,2)
    # fig.subplots_adjust(hspace = 0.65)
    
    # ax[0,0].plot(wind,gentq,'.')
    # ax[0,0].set_title('windspeed vs gentq')
    # #ax[0,0].set_xlim([t0,tf])
    
    # ax[1,0].plot(wind,ptfmpitch,'.')
    # ax[1,0].set_title('windspeed vs ptfmpitch')
    # #ax[1,0].set_xlim([t0,tf])
    
    # ax[0,1].plot(wind,bladepitch,'.')
    # ax[0,1].set_title('windspeed vs bldpitch')
    # #ax[0,1].set_xlim([t0,tf])
    
    # ax[1,1].plot(wind,genspeed,'.')
    # ax[1,1].set_title('windspeed vs genspeed')
    # #ax[1,1].set_xlim([t0,tf])
    
    

    
    # rearrange_flag = False
    
    # if rearrange_flag:
    #     states = states[ind,:]
    #     controls = controls[ind,:]
    #     state_derivatives = state_derivatives[ind,:]
    
    