import numpy as np
from sklearn.cluster import KMeans

def dict2array(data):
    
    
    # initialize storage array
    input_list = []
    state_dx_list = []
    output_list = []
    
    # loop through and extract data from each dictionary
    for isim in data:
        
        # extract
        controls = isim['controls']
        states = isim['states']
        state_derivatives = isim['state_derivatives']
        outputs = isim['outputs']
        
        # number of outputs
        n_outputs = isim['n_outputs']
        
        # stack controls and states
        model_inputs = np.hstack([controls,states])
        
        # add to list
        input_list.append(model_inputs)
        state_dx_list.append(state_derivatives)
        
        # if there are outputs, then append
        if n_outputs > 0:
            output_list.append(outputs)
    
    # vertically stack input and outputs
    model_inputs = np.vstack(input_list)
    state_derivatives = np.vstack(state_dx_list)
    
    if n_outputs > 0:
        outputs = np.vstack(output_list)
    else:
        outputs = []
        

    return model_inputs,state_derivatives,outputs

def sample_data(data,sampling_type,n_samples,grouping = 'together'):
    
    # extract data from dict
    model_inputs,state_derivatives,outputs = dict2array(data)
    
   # get the number of state derivatives and outputs
    n_deriv = data[0]['n_deriv']
    n_outputs = data[0]['n_outputs']
    n_model_inputs = data[0]['n_model_inputs']
    
    
    if sampling_type == 'KM':
        
        if grouping == 'separate':
        
            # initialize storage array
            dx_sampled = np.zeros((n_samples,n_deriv))
            
            # if there are outputs, then initialize
            if n_outputs > 0:
                outputs_sampled = np.zeros((n_samples,n_outputs))
            else: 
                outputs_sampled = []
                
            # perform kmeans clustering algorithm
            kmeans = KMeans(n_clusters = n_samples,n_init = 10).fit(X = model_inputs)
        
            # extract centroid centers and centroid index
            inputs_sampled = kmeans.cluster_centers_
            labels_ = kmeans.labels_
            
            
            # loop through and find the state derrivative values at the cluster centroids
            for icluster in range(n_samples):
                
                # index of elements of the original array present in the current cluster
                cluster_ind = (labels_ == icluster)
                
                # extract the sate derivative values at these indices
                dx_ind = state_derivatives[cluster_ind,:]
                
                # evaluate the state derivative values at the centroid of this cluster
                dx_sampled[icluster,:] = np.mean(dx_ind,axis=0)
                
                # if there are outputs, then extract
                if n_outputs > 0:
                    outputs_ind = outputs[cluster_ind,:]
                    
                    outputs_sampled[icluster,:] = np.mean(outputs_ind, axis = 0)
                    
        elif grouping == 'together':
            

            if n_outputs > 0:
                model_inputs_ = np.hstack([model_inputs,state_derivatives,outputs])
            else:
                model_inputs_ = np.hstack([model_inputs,state_derivatives])
            
            # scale the inputs/outputs
            inputs_max = np.max(abs(model_inputs_),0)
            model_inputs_ = model_inputs_/inputs_max
            
            
            # perform kmeans clustering algorithm
            kmeans = KMeans(n_clusters = n_samples,n_init = 10).fit(X = model_inputs_)
        
            # extract centroid centers and centroid index
            inputs_sampled_ = kmeans.cluster_centers_
            
            inputs_sampled_ = inputs_sampled_*inputs_max
            
            inputs_sampled = inputs_sampled_[:,0:n_model_inputs]
            dx_sampled = inputs_sampled_[:,n_model_inputs:n_model_inputs+n_deriv]
            
            if n_outputs >0:
                outputs_sampled = inputs_sampled_[:,n_model_inputs+n_deriv:]
            else:
                outputs_sampled = []
            
                 
    return inputs_sampled,dx_sampled,outputs_sampled,model_inputs,state_derivatives,outputs    
        
        
        
        
    
    

