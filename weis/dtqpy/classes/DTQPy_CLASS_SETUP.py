# -*- coding: utf-8 -*-
"""
DTQPy_CLASS_SETUP
Class to carry the information about different problem variables

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
import types

class setup:
    def __init__(self,A = np.empty((0,0)),B = np.empty((0,0)),G = np.empty((0,0)),d = np.empty((0,0)),
                 Lagrange = [], Mayer = [],Lquadratic = [],Llinear = [],Lconstant = [],Mquadratic = [],
                 Mlinear = [],Mconstant = [], 
                 UB = [], LB = [], Y = [], Z = [],Scaling = [],
                 t0 = 0,tf = None,auxdata = None,ScaleObjective = False ):
        
        self.A = A; self.B = B; self.G = G; self.d = d;
        
        self.Lagrange = Lagrange; self.Mayer = Mayer
        self.Lquadratic = Lquadratic;self.Llinear = Llinear;self.Lconstant = Lconstant;
        self.Mquadratic = Mquadratic;self.Mlinear = Mlinear;self.Mconstant = Mconstant;
        self.t0 = t0; self.tf = tf;
        self.UB = UB; self.LB = LB
        self.Y = Y; self.Z = Z;self.Scaling = Scaling
        self.auxdata = auxdata;self.ScaleObjective = ScaleObjective
        
        
    def ListClass(self,Class_Name,Class_Type):
        
        """
        Function to evaluate if a class is present in a list
        """
        
        Cname = "self." + Class_Name
        
        # evaluate string
        CnameE = eval(Cname)
        
        # if its a list do nothing
        if isinstance(CnameE,list):
            pass
        
        # else Put the class in a list
        elif isinstance(CnameE,eval(Class_Type)):
            CnameE = [CnameE]
            
            
        return CnameE
    
        
    def Check_Struct(self):
        
        """
        Goes through the attributes of class setup and put them in a list

        """
        # Lagrange
        ClassList = self.ListClass("Lagrange","LQ_objective")
        self.Lagrange = ClassList
        
        # Mayer
        ClassList = self.ListClass("Mayer","LQ_objective")
        self.Mayer = ClassList
        
        # UB
        ClassList = self.ListClass("UB","Simple_Bounds")
        self.UB = ClassList
        
        # LB
        ClassList = self.ListClass("LB","Simple_Bounds")
        self.LB = ClassList
        
        # Y
        ClassList = self.ListClass("Y","Simple_Linear_Constraints")
        self.Y = ClassList
        
        # Z
        ClassList = self.ListClass("Z","Simple_Linear_Constraints")
        self.Z = ClassList
        
        # Scaling
        ClassList = self.ListClass("Scaling","Scaling")
        self.Scaling = ClassList
  
            
    def Check_Matrix_shape(self):
        
        """
        Function to check if the linear dynamic matrices (A,B,G) are in the 
        right shape.

        """
        # A matrix
        if isinstance(self.A,np.ndarray):
            
            # get shape
            s = np.shape(self.A)
            
            # Change 1d array to 2d
            if len(s) == 1:
                self.A = np.reshape(self.A,(s[0],1),order = 'F')
                
        else:
            # float or int into a np array of size [1,1]
            self.A = np.ones((1,1))*self.A  
        
        # B matrix
        if isinstance(self.B,np.ndarray):
            
            # get shape
            s = np.shape(self.B)
            
            # Change 1d array to 2d
            if len(s) == 1:
                self.B = np.reshape(self.B,(s[0],1),order = 'F')
        else:
            # float or int into a np array of size [1,1]
            self.B = np.ones((1,1))*self.B  
            
        # C matrix        
        if isinstance(self.G,np.ndarray):
            
            # get shape
            s = np.shape(self.G)
            
            # change 1d to 2d array
            if len(s) == 1:
                self.G = np.reshape(self.G,(s[0],1),order = 'F')
        else:
            # float or int into a np array of size [1,1]
            self.G = np.ones((1,1))*self.G  
            
        
        
#------------------------------------------------------------------------------       
        
class LQ_objective:
    
    
    def __init__(self,left=None,right=None,matrix=np.empty((0,0))):
        '''
        class LQ_objective
        
        Stores the values of Lagrange and Mayer objective terms
        It has three atributes
         1. left
         2. right
         3. matrix
         
        left and right correspond to the different optimization variables that are present in the problem
        left and right can take values between [0,1,2,3,4,5], with each number corresponding to a specific
        optimization variable
        
        0. No variables
        1. Controls "u(t)"
        2. States "x(t)"
        3. Plant parameters "pl"
        4. Initial state value "x(t0)"
        5. Final state value time "x(tf)"
        
        Eg. the term u'*R*u can be encoded as
        left = 1
        right = 1
        matrix = R
        
        x'Qu can be enocded as
        left = 2
        right = 1
        matrix  = Q
        
        Ky can be encoded as
        left = 0
        right = 2
        matrix  = K
        
        
        Mayer term can only contain time invariant quantities like 
        plant parameters, initial and final state values

        '''
        self.right = right
        self.left = left
        self.matrix = matrix
        

    def Check_shape(self):
        """
        Function to check if the lagrange matrices are in the 
        right shape ie, n-dimensional np array.

        """
        if isinstance(self.matrix,(np.ndarray)):
            
            # get shape
            s = np.shape(self.matrix)
            
            # change 1d to 2d array
            if len(s) ==1:
                self.matrix = np.reshape(self.matrix,(s[0],1),order = 'F')
                
        else:
            # float or int into a np array of size [1,1]
            self.matrix = np.ones((1,1))*self.matrix
            
            
#------------------------------------------------------------------------------ 

class Simple_Bounds:
    
    """
    class Simple_Bounds
    
    Class to store the simple bound constriants for the problem
    It has two attributes:
    1. right
    2. matrix
    
    The field 'right' corresponds to the variable type.
    It can take the following values [1,2,3,4,5]
    
    1. Controls "u(t)"
    2. States "x(t)"
    3. Plant parameters "pl"
    4. Initial state value "x(t0)"
    5. Final state value time "x(tf)"
    
    The field 'matrix' corresponds the matrix
    
    For example:
        0 < u < 5 can be encoded as
        
        UB = Simple_Bounds(right = 1, matrix = 5)
        LB = Simple_Bounds(right = 1, matrix = 0)
        
    Simple_Bounds can also be used to specify initial and final state values.
    For example to set x(0) = 2:
        UB = Simple_Bounds(right = 4, matrix = 2)
        LB = Simple_Bounds(right = 4, matrix = 2)
        
    Additionally, Simple_Bounds class is also used in class Simple_Linear_Constraints
    to store linear matrix
    
    """
    
    
    def __init__(self,right=None,matrix = np.empty((0,0))):
        self.right = right
        self.matrix = matrix
  
        
    def Check_shape(self):
        #breakpoint()
        if isinstance(self.matrix,(np.ndarray)):
            
            s = np.shape(self.matrix)
            
            if len(s) == 1:
                self.matrix = np.reshape(self.matrix,(s[0],1),order = 'F')
                
        elif type(self.matrix) is types.LambdaType:
            temp = np.empty((1,1),dtype = 'O')
            temp[0,0] = self.matrix
            self.matrix = temp
        else:
            self.matrix = np.ones((1,1))*self.matrix

#-----------------------------------------------------------------------------
    
class Simple_Linear_Constraints:
    
    """
    class Simple_Linear_Constraints
    Kx + Nu =< b
    
    Class to store simple linear constraints with the optimization variables
    
    This class takes two fields:
        1. linear
        2. b
        
    1. linear is an instance of class Simple_Bounds with fields right and matrix
    2. b is the lhs
    
    For example 5x + 3u < 1 can be encoded as:
        L1 = Simple_Bounds(right = 2, matrix = 5)
        L2 = Simple_Bounds(right = 1, matrix = 3)
        
        Z = Simple_Linear_Constraints(linear = [L1,L2], b = 1)

    """
    
    def __init__(self,linear=[],b = []):
        
        self.linear = linear
        self.b = b
        
    def Check_shape(self):
        
        # check if the linear field is a list
        if isinstance(self.linear,list):
            pass
        elif isinstance(self.linear,Simple_Bounds):
            self.linear = [self.linear]
            
        # check shape
        if isinstance(self.b,(np.ndarray)):
            
            # get shape
            s = np.shape(self.b)
            
            # change 1d array to 2 d
            if len(s) == 1:
                self.b = np.reshape(self.b,(s[0],1),order = 'F')
        
        # check if it is a anaonymous functions, and put it in a np array
        elif type(self.b) is types.LambdaType:
            temp = np.empty((1,1),dtype = 'O')
            temp[0,0] = self.b
            self.b = temp
                
        else:
            self.b = np.ones((1,1))*self.b
            
#------------------------------------------------------------------------------
            
class Scaling:
    
    """
    class Scaling
    Class to store scaling matrix variables
    
    Scaling has three fields:
        1. right
        2. matrix
        3. constant
        
    Scaling can be used to scale a specific optimization variables 
    x = sM*x + sC
    
    For example to scale the controls by 3 
    u_new = 3*u 
    s = Scaling(right = 1, matrix = 3, constant = 0)
    """
    
    def __init__(self,right = None,matrix = np.empty((0,0)),constant = None):
        self.right = right 
        self.matrix = matrix 
        self.constant = constant 
        
    def Check_Fields(self,internal):
        right = self.right
        nu = internal.nu; ny = internal.ny; npl = internal.npl
        
        if right == 1:
            n = nu
        elif right == 2:
            n = ny
        elif right == 3:
            n = npl
            
        if type(self.matrix) == int or type(self.matrix) == float:
            self.matrix = np.ones((1,1))*self.matrix
            
        elif isinstance(self.matrix,(np.ndarray)):
            
            s = np.shape(self.matrix)
            
            if len(s) == 1:
                self.matrix = np.reshape(self.matrix,(s[0],1),order = 'F')
                
        elif type(self.matrix) is types.LambdaType:
            temp = np.empty((1,1),dtype = 'O')
            temp[0,0] = self.matrix
            self.matrix = temp
        else:
            self.matrix = np.ones((1,n))
            
        
        if type(self.constant) == int or type(self.constant) == float:
            self.constant = np.ones((1,1))*self.constant
            
        elif isinstance(self.constant,(np.ndarray)):
            
            s = np.shape(self.constant)
            
            if len(s) == 1:
                self.constant = np.reshape(self.constant,(s[0],1),order = 'F')
                
        elif type(self.constant) is types.LambdaType:
            temp = np.empty((1,1),dtype = 'O')
            temp[0,0] = self.constant 
            self.constant = temp
        else:
             self.constant = np.zeros((1,n))

#-----------------------------------------------------------------------------
        
         
class auxdata():
    """
    class to store auxillary data associated with a problem
    """
    pass