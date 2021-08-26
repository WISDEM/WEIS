# -*- coding: utf-8 -*-
"""
DTQPy_DEFECTS_TR
Create defect constraint using trapezoidal rule

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np
from numpy.matlib import repmat
from scipy.sparse import csc_matrix
from matplotlib.pyplot import spy
from dtqpy.src.utilities.DTQPy_tmultiprod import DTQPy_tmultiprod

def Index_Columns(a,b,c):
            J = np.arange(a*c+1,a*(b+c)+1)
            J = np.reshape(J,(a,b),order = 'F')
            J = np.delete(J,a-1,axis=0)
            J = np.reshape(J,((a-1)*b,1),order = 'F')
            return J-1 # python indexing

def DTQPy_DEFECTS_TR(A,B,G,d,internal,opts):
        
        # Extract variables
        nu = internal.nu; ny = internal.ny; npl = internal.npl; nd = internal.nd; nx = internal.nx;
        auxdata = internal.auxdata; nt = internal.nt; t = internal.t; h = internal.h
        
        
        # evaluate time varying matrices
        
        At = DTQPy_tmultiprod(A,auxdata,t)
        Bt = DTQPy_tmultiprod(B,auxdata,t)
        Gt = DTQPy_tmultiprod(G,auxdata,t)
        dt = DTQPy_tmultiprod(d,auxdata,t)
        
        Aflag = At.any()
        Bflag = Bt.any()
        Gflag = Gt.any()
        dflag = dt.any()
        
        #breakpoint()
        # initalize identity matrix
        K = np.kron(np.eye(ny),np.ones((nt-1,1)))
        
       
        # initialize storage arrays
        Isav = np.array([])
        Jsav = np.array([])
        Vsav = np.array([])
        
        # Generate column indices
        Jy = Index_Columns(nt, ny, nu)
        Jys = np.vstack([Jy,Jy+1])
        Ty = Jy-nt*nu 
        Hy = repmat(0.5*h,ny,1)
        
        # Column indices for controls
        if nu>0:
            Ju = Index_Columns(nt,nu,0)
            Jus = np.vstack([Ju,Ju+1])
            Tu = Ju
            Hu = repmat(0.5*h,nu,1)
        
        # Column indices for plant parameters
        if npl>0:
            Jp = np.kron(nt*(nu+ny)*(np.arange(1,npl+1)),np.ones((nt-1,1)))
            Tp = Index_Columns(nt,npl,0)
            Hp = repmat(0.5*h,npl,1)
        
        # Column indices for disturbances
        if nd>0:
            Hd = np.squeeze(0.5*h)
            Td = np.arange(0,nt-1)
        
        # Defect constraint of row continuous constraints
        for i in range(ny):
            
            # current defect constraint row indices
            DefectIndices = np.arange(i*(nt-1),(i+1)*(nt-1))
            DefectIndices = DefectIndices[None].T
            
            # controls
                
            if nu>0 and Bflag:
               # extract controls
               Bv = np.reshape(Bt[:,i,:],(nt*nu,1),order = 'F')
               
               # check for nonzero entires
               if Bv.any():
                   
                   # defect constraint row value
                   Iu = repmat(DefectIndices,nu,1)
                
                   # theta values
                   V3 = -Hu*(np.take(Bv,Tu))
                   V4 = -Hu*(np.take(Bv,Tu+1))
                
                   # combine
                   Is = np.vstack([Iu,Iu]);Js = Jus; Vs = np.vstack([V3,V4])
                   
                   # remove zeros
                   NZeroIndex = np.nonzero(Vs)
                   Is = Is[NZeroIndex[0]]; Js = Js[NZeroIndex[0]]; Vs = Vs[NZeroIndex[0]]
                   
                   # combine
                   Isav = np.append(Isav,Is);Jsav = np.append(Jsav,Js);Vsav = np.append(Vsav,Vs)
                   
               # states    
               if ny>0:
                  
                 # extract A matrix if nonempty
                 if Aflag:
                     Av = np.reshape(At[:,i,:],(nt*ny,1),order = 'F')
                 else:
                     Av = np.array([])
                 
                 
                 Ki = K[:,i]; Ki = Ki[None].T; KiFlag = Ki.any()
                 
                 if  KiFlag or Av.any():
                     
                     # defect row constraints
                     Iy = repmat(DefectIndices,ny,1)
                     
                     # theta values
                     if Av.any():
                        
                         V1 = -Ki - Hy*(np.take(Av,Ty))
                         V2 = Ki -Hy*(np.take(Av,Ty+1))
                     else:
                         V1 = -Ki
                         V2 = Ki
                  
                     # combine
                     Is = np.vstack([Iy,Iy]); Js = Jys; Vs = np.vstack([V1,V2])
                     
                     # remove zeros
                     NZeroIndex = np.nonzero(Vs)
                     Is = Is[NZeroIndex[0]]; Js = Js[NZeroIndex[0]]; Vs = Vs[NZeroIndex[0]]
                     
                     # combine
                     Isav = np.append(Isav,Is);Jsav = np.append(Jsav,Js);Vsav = np.append(Vsav,Vs)
                     
               if npl>0 and Gflag:
                   
                   # extract G matrix
                   Gv = np.reshape(Gt[:,i,:],(nt*npl,1),order = 'F')
                   
                   if any.Gv():
                       
                       # defect constraint (row) locations
                       Is = repmat(DefectIndices,np,1)
                       
                       # theta values
                       Vs = -Hp*(np.take(Gv,Tp)+np.take(Gv,Tp+1))
                       
                       # combine
                       Js = Jp
                       
                       # remove zeros
                       NZeroIndex = np.nonzeros(Vs)
                       Is = Is[NZeroIndex[0]]; Js = Js[NZeroIndex[0]]; Vs = Vs[NZeroIndex[0]]
                       
                       # combine
                       Isav = np.append(Isav,Is);Jsav = np.append(Jsav,Js);Vsav = np.append(Vsav,Vs)
                   
              
                  
         # Generate output sparse matrix           
        Aeq = csc_matrix((Vsav,(Isav,Jsav)),shape = (ny*(nt-1),nx)) 
        

        #  Disturbances
        if nd >0 and dflag:
            
            # initialize storage arrays
            Isav = np.array([]); Vsav = np.array([])
            
            for i in range(ny):
                
                # extract matrices
                dv = np.reshape(dt[:,i,:],(nt*nd,1),order = 'F')
                
                # check for nonzero entries
                if dv.any():
                    
                    # defect constraint (row) locations
                    Is = np.reshape(np.arange((i)*(nt-1),(i+1)*(nt-1)),(nt-1,1),order = 'F')
                    #breakpoint()
                    # nu values
                    Vs = Hd*(np.take(dv,Td)+np.take(dv,Td+1))
                    #breakpoint()
                    # remove zeros
                    NZeroIndex = np.nonzero(Vs)
                    Is = Is[NZeroIndex[0]]; Vs = Vs[NZeroIndex[0]]
                 
                    # combine
                    Isav = np.append(Isav,Is);Vsav = np.append(Vsav,Vs)
            #breakpoint()
            # output sparse matrix
            Ilen = len(Isav)
            Jsav = np.zeros((Ilen,))
            beq = csc_matrix((Vsav,(Isav,Jsav)),shape = (ny*(nt-1),1))
            
        else:
            # output sparse matrix
            beq = csc_matrix((ny*(nt-1),1))
                
        
        return Aeq,beq
                
                















         