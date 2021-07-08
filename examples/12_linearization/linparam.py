import numpy as np

from eva import eigA
import copy


def openFASTstateSpace(eval_parameters, *args, **kwargs):
    """
    Get linearized state space model from OpenFAST for a given set of parameters
    """
    # Generate OpenFAST inputs
    # TODO
    # Call OpenFAST
    # TODO
    # Open "lin" files
    # TODO
    # Run MBC
    # TODO
    # Return time invariant state space:
    A = np.zeros((4,4))
    B = np.zeros((4,2))
    C = np.zeros((3,5))
    D = np.zeros((3,2))

    A[:2,2:] = np.eye(2)
    A[2,0]= -eval_parameters[list(eval_parameters.keys())[0]]
    A[3,1]= -eval_parameters[list(eval_parameters.keys())[1]]
    A[2,2]= -eval_parameters[list(eval_parameters.keys())[0]]
    A[3,3]= -eval_parameters[list(eval_parameters.keys())[1]]
    return A,B,C,D


def stateSpaceSlopes(ABCDp, ABCDm, deltaP):
    """ 
    Compute slopes of linear state space model (A,B,C,D matrices), e.g.:
        DeltaA = (Ap-Am)/deltaP

    INPUTS:
    - ABCDp: array contaning A,B,C,D evaluated for a "positive" perturbation
    - ABCDm: array contaning A,B,C,D evaluated for a "negative" perturbation

    """
    return [ (Ap-Am)/deltaP for Ap,Am in zip(ABCDp,ABCDm)]

def parametricStateSpace(parameter_ranges, *args, **kwargs):
    """ 
    Obtain the linearized state space model for a set of reference parameters and get the 
    individual slopes for changes in each of the parameters.

    INPUTS:
     - parameter_ranges: dictionary with the parameter name as key, and as value a dictionary of the form:
               {'min':pmin , 'max':pmax, 'ref':pref}
    OUTPUTS:
      - ABCD0: state space at reference point 
      - dABCD: state space slopes
    """

    # First evaluate at reference
    eval_parameters_ref={}
    for name,param in parameter_ranges.items():
        eval_parameters_ref[name] = param['ref']

    ABCD0 = openFASTstateSpace(eval_parameters_ref, *args, **kwargs)

    # For each parameter, evalute at min and max, and compute slope
    dABCD={}
    for name,p_range in parameter_ranges.items():
        eval_parameters=eval_parameters_ref.copy()
        deltaP = p_range['max'] - p_range['min']
        # Eval at min
        eval_parameters[name] = p_range['min']
        ABCDm = openFASTstateSpace(eval_parameters, *args, **kwargs)
        # Eval at max
        eval_parameters[name] = p_range['max']
        ABCDp = openFASTstateSpace(eval_parameters, *args, **kwargs)
        # Compute slope
        dABCD[name] = stateSpaceSlopes(ABCDp, ABCDm, deltaP)

    return ABCD0, dABCD

def evalParametericStateSpace(parameters, parameter_ranges, ABCD0, dABCD):
    """ 
    INPUTS:
      - parameters:
    """
    A,B,C,D = copy.deepcopy(ABCD0)
    for name,p in parameters.items():
        deltaP = p - parameter_ranges[name]['ref'] 
        A += dABCD[name][0] * deltaP
        B += dABCD[name][1] * deltaP
        C += dABCD[name][2] * deltaP
        D += dABCD[name][3] * deltaP
    return A,B,C,D


if __name__ == '__main__':


    # --- Initial step
    # Define parameters to be changed in optimization
    parameter_ranges = {
        'tower root diameter':{'min':5, 'max':10, 'ref':6},
        'tower top diameter': {'min':2, 'max':10, 'ref':4}
    }

    # Evaluate state space at ref point, and all statespace slopes
    SS0, dSS = parametricStateSpace(parameter_ranges)

    # --- Within WEIS Optimization loop
    # Optimizer requesting state space at a given point
    parameters={
        'tower root diameter':8,
        'tower top diameter': 4
    }
    # Method "2A"
    SSl = evalParametericStateSpace(parameters, parameter_ranges, SS0, dSS)
    print('State space evaluated using linearized param')
    print(SSl[0])
    # Method "3"
    SSd = openFASTstateSpace(parameters)
    print('State space evaluated directly')
    print(SSl[0])


    # --- Comparison of methods
    # What we need to do now, compare the linear parametrization with brute force
    # Perform a 2D sweep on the parameter space
    vp1 = np.linspace(5.,8.,3)
    vp2 = np.linspace(2.,3.,4)

    nFreq=1
    dF = np.zeros((len(vp1),len(vp2), nFreq)) # Relative error in frequencies
    dZ = np.zeros((len(vp1),len(vp2), nFreq)) # Relative error in damping

    for i1, p1 in enumerate(vp1):
        for i2, p2 in enumerate(vp2):
            parameters={
                'tower root diameter':p1,
                'tower top diameter': p2,
            }
            # Direct call
            ABCDd = openFASTstateSpace(parameters)
            # Linear parametrization call
            ABCDl = evalParametericStateSpace(parameters, parameter_ranges, SS0, dSS)
            # Compare eigenvalues
            Ad = ABCDd[0]
            Al = ABCDl[0]
            freqd,dampd,_,_, = eigA(Ad)
            freql,dampl,_,_, = eigA(Al)
            # Unfortunately, if things are unphysical we won't get the same number of frequencies
            if len(freqd) and len(freql)>0:
                dF[i1,i2,:] = (freqd[:nFreq]-freql[:nFreq])/freqd[:nFreq]*100
                dZ[i1,i2,:] = (dampd[:nFreq]-dampl[:nFreq])/dampd[:nFreq]*100
            else:
                dF[i1,i2,:] = np.nan
                dZ[i1,i2,:] = np.nan

    print('relative error in first frequency')
    print(dF[:,:,0])

