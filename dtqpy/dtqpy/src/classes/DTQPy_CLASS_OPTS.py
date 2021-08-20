# -*- coding: utf-8 -*-
"""
DTQPy_CLASS_OPTS
Create the options class

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np

class options:
    def __init__(self):
        self.general = self.general()
        self.dt = self.dt()
        self.method = self.method()
        self.solver = self.solver()
        
     # general options   
    class general:
        def __init__(self,plotflag = None,saveflag = None,displevel = None):
            self.plotflag = plotflag;
            self.saveflag = saveflag
            self.displevel = displevel
            
     # options specific to direct transcription methods        
    class dt:
        def __init__(self,defects = 'TR',quadrature = 'CTR',mesh = 'ED',nt = 100,meshr = None):
            self.defects = defects # method to transcribe the dynamics
            self.quadrature = quadrature # method to transcribe quadrature
            self.mesh = mesh # mesh
            self.nt = nt # number of time points in a given time horizon
            self.meshr = self.meshr() # mesh refinement
            
        class meshr:
            def __init__(self,method = 'None'):
                self.method = 'None'
                
    class method:
        def __init__(self,reordervariables = False,scalematrixrows = False, form = None, olqflag = True):
            self.reordervariables = reordervariables
            self.scalematrixrows = scalematrixrows 
            self.form = form 
            self.olqflag = olqflag 
            
    # solver specific options        
    class solver:
        def __init__(self,function = 'osqp',tolerence = 1e-3,maxiters = 50000,display = True):
            self.function = function # function to solve the QP problem
            self.tolerence = tolerence # solver tolerence
            self.maxiters = maxiters # maximum iterations
            self.display = display # display options