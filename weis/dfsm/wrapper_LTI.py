import numpy as np 

class wrapper_LTI():
    
    def __init__(self,inputs = None,outputs = None,nstates = None,ncontrols = None):
        self.inputs = inputs
        self.outputs = outputs
        
        self.nstates = nstates
        self.ncontrols = ncontrols 
        

        
    def linear_model(self,x):
        
        nstates = self.nstates 
        ncontrols = self.ncontrols
        
        # reshape the parameters
        x = np.reshape(x,[int(nstates/2),nstates+ncontrols],order = 'F')
        
        # extract elements corresponding to A and B matrices
        B_par = x[:,:ncontrols]
        A_par = x[:,ncontrols:ncontrols+nstates]
        
        # construct A and B matrices
        A = np.hstack([np.zeros((int(nstates/2),int(nstates/2))),np.eye(int(nstates/2))])
        A = np.vstack([A,A_par])
        
        B = np.vstack([np.zeros((int(nstates/2),ncontrols)),B_par])
        
        
        return A,B
        
    def objective_function(self,xdict):
        
        # extract inputs
        inputs = self.inputs 
        outputs = self.outputs
            
        # extract parameters
        x = xdict['xvars']
        
        # reshape
        A,B = self.linear_model(x)
        
        Aeig = np.linalg.eig(A)
        Aeig = Aeig[0]
        
        LM = np.hstack([B,A]).T
        
        # prediction
        dx_predicted = np.dot(inputs,LM)
        
        # evaluate error
        error = outputs - dx_predicted
        
        # number of samples
        N = len(error)
        
        # calculate loss
        V = 1/N*(np.trace(np.dot(error.T,error)))
        
        # initialize
        funcs = {}
        
        # objective
        funcs['obj'] = V
        #print(Aeig.real)
        
        # constraints
        funcs['con'] = Aeig.real
        
        fail = False
        
        return funcs,fail

