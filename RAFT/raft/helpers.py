import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- misc classes -----------------------------------


class Env:
    '''This could be a simple environmental parameters class <<< needed most immediately to pass rho and g info.'''
    def __init__(self):
        self.rho = 1025.0
        self.g = 9.81
        self.Hs = 1.0
        self.Tp = 10.0
        self.spectrum = "unit"
        self.V = 10.0
        self.beta = 0.0




# ------------------------ misc helper functions -------------------------------
# GB: If it helps, there are np.rad2deg and np.deg2rad functions
def rad2deg(rad):
    return rad*57.29577951308232
def deg2rad(deg):
    return deg*0.017453292519943295
    
def rpm2radps(rpm):
    return rpm*0.1047
def radps2rpm(radps):
    return radps/0.1047


def FrustumVCV(dA, dB, H, rtn=0):
    '''returns the volume and center of volume of a frustum, which can be a cylinder (box), cone (pyramid), or anything in between
    Source: https://mathworld.wolfram.com/PyramidalFrustum.html '''

    if np.sum(dA)==0 and np.sum(dB)==0:
        V = 0
        hc = 0
    else:
        if np.isscalar(dA) and np.isscalar(dB): # if the inputs are scalar, meaning that it's just a diameter
            A1 = (np.pi/4)*dA**2
            A2 = (np.pi/4)*dB**2
            Amid = (np.pi/4)*dA*dB
        elif len(dA)==2 and len(dB)==2: # if the inputs are of length 2, meaning if it's two side lengths per node
            A1 = dA[0]*dA[1]
            A2 = dB[0]*dB[1]
            Amid = np.sqrt(A1*A2)
        else:
            raise ValueError('Input types not accepted')

        V = (A1 + A2 + Amid) * H/3
        hc = ((A1 + 2*Amid + 3*A2)/(A1 + Amid + A2)) * H/4

    if rtn==0:
        return V, hc
    elif rtn==1:
        return V
    elif rtn==2:
        return hc


def getKinematics(r, Xi, ws):
    '''Get node complex displacement, velocity, and acceleration complex 
    amplitudes based on platform motion's and relative position from 
    platform reference point PRP).
    
    Parameters
    ----------
    r : array
        X, y, z coordinates of point of interest relative to PRP [m].
    Xi : complex 2D array
        Complex amplitudes of 6 degree of freedom as a function of frequency
        (size 6 by nw).
    ws : array
        Frequency vector of length nw [rad/s].
    
    Returns
    -------
    dr, v, a : complex 2D array
        Each is a 3 by nw array of the complex amplitudes of the point's
        displacements, velocities, and accelerations, respectively.
    '''

    nw = len(ws)

    dr = np.zeros([3,nw], dtype=complex) # node displacement complex amplitudes
    v  = np.zeros([3,nw], dtype=complex) # velocity
    a  = np.zeros([3,nw], dtype=complex) # acceleration


    for i in range(nw):
        dr[:,i] = Xi[:3,i] + SmallRotate(r, Xi[3:,i])
        v[ :,i] = 1j*ws[i]*dr[:,i]
        a[ :,i] = 1j*ws[i]*v[ :,i]


    return dr, v, a # node dispalcement, velocity, acceleration (Each  [3 x nw])


## Get wave velocity and acceleration complex amplitudes based on wave spectrum at a given location
def getWaveKin(zeta0, beta, w, k, h, r, nw, rho=1025.0, g=9.81):

    # inputs: wave elevation fft, wave freqs, wave numbers, depth, point position

    zeta = np.zeros(nw , dtype=complex ) # local wave elevation
    u  = np.zeros([3,nw], dtype=complex) # local wave kinematics velocity
    ud = np.zeros([3,nw], dtype=complex) # local wave kinematics acceleration
    pDyn = np.zeros(nw , dtype=complex ) # local dynamic pressure

    for i in range(nw):

        # .............................. wave elevation ...............................
        zeta[i] = zeta0[i]* np.exp( -1j*(k[i]*(np.cos(beta)*r[0] + np.sin(beta)*r[1])))  # shift each zetaC to account for location


        # ...................... wave velocities and accelerations ............................
        z = r[2]

        # only process wave kinematics if this node is submerged
        if z < 0:

            # Calculate SINH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/COSH( k*h )
            # given the wave number, k, water depth, h, and elevation z, as inputs.
            if (    k[i]   == 0.0  ):                   # When .TRUE., the shallow water formulation is ill-conditioned; thus, the known value of unity is returned.
                SINHNumOvrSIHNDen = 1.0
                breakpoint()
                COSHNumOvrSIHNDen = 99999.0
                COSHNumOvrCOSHDen = 99999.0   # <<< check
            elif ( k[i]*h >  89.4 ):                # When .TRUE., the shallow water formulation will trigger a floating point overflow error; however, with h > 14.23*wavelength (since k = 2*Pi/wavelength) we can use the numerically-stable deep water formulation instead.
                SINHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrCOSHDen = np.exp( k[i]*z ) + np.exp(-k[i]*(z + 2.0*h))
            else:                                    # 0 < k*h <= 89.4; use the shallow water formulation.
                SINHNumOvrSIHNDen = np.sinh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrSIHNDen = np.cosh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrCOSHDen = np.real( np.cosh(k[i]*(z+h)) )/np.cosh(k[i]*h)   # <<< check

            # Fourier transform of wave velocities
            u[0,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.cos(beta)
            u[1,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.sin(beta)
            u[2,i] = 1j*w[i]* zeta[i]*SINHNumOvrSIHNDen

            # Fourier transform of wave accelerations
            ud[:,i] = 1j*w[i]*u[:,i]

            # Fourier transform of dynamic pressure
            pDyn[i] = rho*g* zeta[i] * COSHNumOvrCOSHDen


    return u, ud, pDyn



# calculate wave number based on wave frequency in rad/s and depth
def waveNumber(omega, h, e=0.001):

    g = 9.81
    # omega - angular frequency of waves

    k = 0                           #initialize  k outside of loop
                                    #error tolerance
    k1 = omega*omega/g                  #deep water approx of wave number
    k2 = omega*omega/(np.tanh(k1*h)*g)      #general formula for wave number
    while np.abs(k2 - k1)/k1 > e:           #repeate until converges
        k1 = k2
        k2 = omega*omega/(np.tanh(k1*h)*g)

    k = k2

    return k


# translate point at location r based on three small angles in th
def SmallRotate(r, th):

    rt = np.zeros(3, dtype=complex)    # translated point
    
    rt[0] =              -th[2]*r[1] + th[1]*r[2]
    rt[1] =  th[2]*r[0]              - th[0]*r[2]
    rt[2] = -th[1]*r[0] + th[0]*r[1]
    '''
    rt[0] =              th[2]*r[1] - th[1]*r[2]
    rt[1] = th[2]*r[0]              - th[0]*r[2]
    rt[2] = th[1]*r[0] - th[0]*r[1]
    '''
    return rt


# given a size-3 vector, vec, return the matrix from the multiplication vec * vec.transpose
def VecVecTrans(vec):

    vvt = np.zeros([3,3])

    for i in range(3):
        for j in range(3):
            vvt[i,j]= vec[i]*vec[j]

    return vvt


def intrp(x, xA, xB, yA, yB):  
    '''Do simple interpolation between two points.'''
    return yA + (x-xA)*(yB-yA)/(xB-xA)

# produce alternator matrix
def getH(r):
    '''Produces the alternator matrix for a size-3 vector, r. Multiplying this array by a vector 
    is equivalent to the cross produce of r and that vector.
    '''
    
    H = np.array([[ 0   , r[2],-r[1]],
                  [-r[2], 0   , r[0]],
                  [ r[1],-r[0], 0   ]])

    return H

def rotationMatrix(x3,x2,x1):
    '''Calculates a rotation matrix based on order-z,y,x instrinsic (tait-bryan?) angles, meaning
    they are about the ROTATED axes. (rotation about z-axis would be (0,0,theta) )
    
    Parameters
    ----------
    x3, x2, x1: floats
        The angles that the rotated axes are from the nonrotated axes. Normally roll,pitch,yaw respectively. [rad]

    Returns
    -------
    R : matrix
        The rotation matrix
    '''
    # initialize the sines and cosines
    s1 = np.sin(x1) 
    c1 = np.cos(x1)
    s2 = np.sin(x2) 
    c2 = np.cos(x2)
    s3 = np.sin(x3) 
    c3 = np.cos(x3)
    
    # create the rotation matrix
    R = np.array([[ c1*c2,  c1*s2*s3-c3*s1,  s1*s3+c1*c3*s2],
                  [ c2*s1,  c1*c3+s1*s2*s3,  c3*s1*s2-c1*s3],
                  [   -s2,           c2*s3,           c2*c3]])
    
    return R    

def translateForce3to6DOF(Fin, r):
    '''Takes in a position vector and a force vector (applied at the positon), and calculates
    the resulting 6-DOF force and moment vector.

    :param array r: x,y,z coordinates at which force is acting [m]
    :param array Fin: x,y,z components of force [N]
    :return: the resulting force and moment vector
    :rtype: array
    '''
    Fout = np.zeros(6, dtype=Fin.dtype) # initialize output vector as same dtype as input vector (to support both real and complex inputs)

    Fout[:3] = Fin

    Fout[3:] = np.cross(r, Fin)

    return Fout


def transformForce(f_in, offset=[], orientation=[]):
    '''Transform a size-3 or size-6 force from one reference frame to another
    
    Parameters
    ----------
    f_in : size 3 or 6 array
        the input force vector or force and moment vector
    offset : size-3 array
        the x,y,z coordinates at which f_in is acting, relative to the reference frame at which the force and moment should be returned
    orientation : size-3 array
        The orientation of f_in relative to the reference frame of the results. If size 3: x,y,z Euler angles 
        describing the rotations around each axis (applied in order z, y, x). If 3-by-3, the rotation matrix.
    '''
    
    # input size checks
    if not len(f_in) in [3,6]:
        raise ValueError("f_in input must be size 3 or 6")
    if not len(offset) in [0,3]:
        raise ValueError("offset input if provided must be size 3")
    
    # prep output
    if len(f_in) == 6:
        f = np.array(f_in) 
    elif len(f_in) == 3:
        f = np.hstack([f_in, [0,0,0]]) 
    
    # prep rotation matrix
    if len(orientation) > 0:
        rot = np.array(orientation)
        if rot.shape == (3,):
            rotMat = rotationMatrix(*rot)
        elif rot.shape == (3,3):
            rotMat = rot
        else:
            raise ValueError("orientation input if provided must be size 3 or 3-by-3")
        
    # rotation
    f_in2 = np.array(f_in)
    if len(orientation) > 0:
        f[:3] = np.matmul(rotMat, f_in2[:3])
        if len(f_in) == 6:
            f[3:] = np.matmul(rotMat, f_in2[3:])
        
    # translation
    if len(offset) > 0:
        f[3:] += np.cross(offset, f[:3])  # add moment created by offsetting forces
    
    return f


# translate mass matrix to make 6DOF mass-inertia matrix
def translateMatrix3to6DOF(Min, r):
    '''Transforms a 3x3 matrix to be about a translated reference point, resulting in a 6x6 matrix.'''

    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    # note that the J term and I terms are zero in this case because the input is just a mass matrix (assumed to be about CG)

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik

    Mout = np.zeros([6,6]) #, dtype=complex)

    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min

    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[:3,3:] = np.matmul(Min, H)
    Mout[3:,:3] = Mout[:3,3:].T

    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]

    Mout[3:,3:] = np.matmul(np.matmul(H,Min), H.T)

    return Mout


def translateMatrix6to6DOF(Min, r):
    '''Transforms a 6x6 matrix to be about a translated reference point.
    r is a vector that goes from where you want the reference point to be
    to where the reference point currently is'''

    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik

    Mout = np.zeros([6,6]) #, dtype=complex)

    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min[:3,:3]

    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[:3,3:] = np.matmul(Min[:3,:3], H) + Min[:3,3:]
    Mout[3:,:3] = Mout[:3,3:].T

    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]
    Mout[3:,3:] = np.matmul(np.matmul(H,Min[:3,:3]), H.T) + np.matmul(Min[3:,:3], H) + np.matmul(H.T, Min[:3,3:]) + Min[3:,3:]

    return Mout


   
def rotateMatrix6(Min, rotMat):
    '''apply a rotation to a 6-by-6 mass/inertia tensor (see Sadeghi and Incecik 2005 for theory)
    the process for each of the following is to 
    1. copy out the relevant 3x3 matrix section,
    2. rotate it, and
    3. paste it into the output 6x6 matrix
    
    PARAMETERS
    ----------
    Min : array(6,6) 
        inputted matrix to be rotated
    rotMat : array(3,3)  
        rotation matrix (DCM)
    '''    
    outMat = np.zeros_like(Min)  # rotated matrix

    shape = Min.shape
    if not (shape[0]==6 and shape[1]==6):
        raise Exception('The input matrix must be 6x6 (with an optional third dimension).')

    if len(shape) == 2:
        outMat[:3,:3] = rotateMatrix3(Min[:3,:3], rotMat)    # mass matrix
        outMat[:3,3:] = rotateMatrix3(Min[:3,3:], rotMat)    # product of inertia matrix
        outMat[3:,:3] = outMat[:3,3:].T
        outMat[3:,3:] = rotateMatrix3(Min[3:,3:], rotMat)    # moment of inertia matrix
    
    elif len(shape) == 3:
        for i in range(shape[2]):  # process each 6x6 slice
            outMat[:3,:3, i] = rotateMatrix3(Min[:3,:3, i], rotMat)
            outMat[:3,3:, i] = rotateMatrix3(Min[:3,3:, i], rotMat)
            outMat[3:,:3, i] = outMat[:3,3:, i].T
            outMat[3:,3:, i] = rotateMatrix3(Min[3:,3:, i], rotMat)
    else:
        raise Exception('Input matrix must be two- or three-dimensional.')

    return outMat


def rotateMatrix3(Min, rotMat):
    '''apply a rotation to a 3-by-3 mass matrix or any other second order tensor   
    overall operation is [m'] = [a]*[m]*[a]^T
    
    PARAMETERS
    ----------
    Min : array(3,3) 
        inputted matrix to be rotated
    rotMat : array(3,3)  
        rotation matrix (DCM)
    '''
    return np.matmul( np.matmul(rotMat, Min), rotMat.T )


def RotFrm2Vect( A, B):
    '''Rodriguez rotation function, which returns the rotation matrix 
    that transforms vector A into Vector B.
    '''
    
    
    v = np.cross(A,B)
    
    if np.sum(v**2)==0:  # if something goes wrong (or A==B), no transformation
        return np.eye(3)
        
    ssc = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])
         
    R =  np.eye(3,3) + ssc + np.matmul(ssc,ssc)*(1-np.dot(A,B))/(np.linalg.norm(v)*np.linalg.norm(v))            

    return R

def getRMS(xi):
    '''Calculates standard deviation or RMS of inputted (complex) response amplitude vector.
    If a matrix is provided, the first dimension is considered to be multiple cases and the
    second dimension is considered to be frequencies. Results are summed across cases for 
    each frequency. The calculation is the same regardless.'''
    
    return np.sqrt( np.sum( np.abs(xi)**2 ) )


def getPSD(xi, dw):
    '''Calculates power spectral density from inputted (complex) response amplitude vector. Units of [unit]^2/(rad/s).
    If a matrix is provided, the first dimension is considered to be multiple cases and the
    second dimension is considered to be frequencies. Results are summed across cases for 
    each frequency.'''
    
    if len(xi.shape) == 1:
        psd = 0.5*np.abs(xi)**2/dw
    elif len(xi.shape) == 2:
        psd = np.sum(0.5*np.abs(xi)**2/dw, axis=0)  # sum squares across excitation sources for each frequency
    else:
        raise Exception("getPSD must be passed an array with 1 or 2 dimensions.")
    
    return psd


def JONSWAP(ws, Hs, Tp, Gamma=None):
    '''Returns the JONSWAP wave spectrum for the given frequencies and parameters.

    Parameters
    ----------
    ws : float | array
        wave frequencies to compute spectrum at (scalar or 1-D list/array) [rad/s]
    Hs : float
        significant wave height of spectrum [m]
    Tp : float
        peak spectral period [s]
    Gamma : float
        wave peak shape parameter []. The default value of 1.0 gives the Pierson-Moskowitz spectrum.

    Returns
    -------
    S : array
        wave power spectral density array corresponding to frequencies in ws [m^2/Hz]


    This function calculates and returns the one-sided power spectral density spectrum
    at the frequency/frequencies (ws) based on the provided significant wave height,
    peak period, and (optionally) shape parameter gamma.

    This formula for the JONSWAP spectrum is adapted from FAST v7 and based
    on what's documented in IEC 61400-3.
    '''

    # If peak shape parameter gamma is not specified, use the recommendation 
    # from IEC 61400-3 as a function of Hs and Tp. For PM spectrum, use 1.
    if not Gamma:
        TpOvrSqrtHs = Tp/np.sqrt(Hs)
        if TpOvrSqrtHs <= 3.6:
            Gamma = 5.0
        elif TpOvrSqrtHs >= 5.0:
            Gamma = 1.0
        else:
            Gamma = np.exp( 5.75 - 1.15*TpOvrSqrtHs )
    
    # handle both scalar and array inputs
    if isinstance(ws, (list, tuple, np.ndarray)):
        ws = np.array(ws)
    else:
        ws = np.array([ws])

    # initialize output array
    S = np.zeros(len(ws))


    # the calculations
    f        = 0.5/np.pi * ws                         # wave frequencies in Hz
    fpOvrf4  = pow((Tp*f), -4.0)                      # a common term, (fp/f)^4 = (Tp*f)^(-4)
    C        = 1.0 - ( 0.287*np.log(Gamma) )          # normalizing factor
    Sigma = 0.07*(f <= 1.0/Tp) + 0.09*(f > 1.0/Tp)    # scaling factor

    Alpha = np.exp( -0.5*((f*Tp - 1.0)/Sigma)**2 )

    return  0.5/np.pi *C* 0.3125*Hs*Hs*fpOvrf4/f *np.exp( -1.25*fpOvrf4 )* Gamma**Alpha


def printMat(mat):
    '''Print a matrix'''
    for i in range(mat.shape[0]):
        print( "  ".join(["{:+10.3e}"]*mat.shape[1]).format( *mat[i,:] ))

def printVec(vec):
    '''Print a vector'''
    print( "  ".join(["{:+10.3e}"]*len(vec)).format( *vec ))


def getFromDict(dict, key, shape=0, dtype=float, default=None, index=None):
    '''
    Function to streamline getting values from design dictionary from YAML file, including error checking.

    Parameters
    ----------
    dict : dict
        the dictionary
    key : string
        the key in the dictionary
    shape : list, optional
        The desired shape of the output. If not provided, assuming scalar output. If -1, any input shape is used.
    dtype : type
        Must be a python type than can serve as a function to format the input value to the right type.
    default : number or list, optional
        The default value to fill in if the item isn't in the dictionary. 
        Otherwise will raise error if the key doesn't exist. It may be a list
        (to be tiled shape times if shape > 1) but may not be a numpy array.
    '''
    # in future could support nested keys   if type(key)==list: ...

    if key in dict:
        val = dict[key]                                      # get the value from the dictionary
        if shape==0:                                         # scalar input expected
            if np.isscalar(val):
                return dtype(val)
            else:
                raise ValueError(f"Value for key '{key}' is expected to be a scalar but instead is: {val}")
        elif shape==-1:                                      # any input shape accepted
            if np.isscalar(val):
                return dtype(val)
            else:
                return np.array(val, dtype=dtype)
        else:
            if np.isscalar(val):                             # if a scalar value is provided and we need to produce an array (of any shape)
                return np.tile(dtype(val), shape)

            elif np.isscalar(shape):                         # if expecting a 1D array (or if wanting the result to have the same length as the input)
                if len(val) == shape:                        # throw an error if the input is not the same length as the shape, meaning the user is missing data
                    if index == None:
                        return np.array([dtype(v) for v in val])    # if no index is provided, do normally and return the array input
                    else:
                        keyshape = np.array(val).shape              # otherwise, use the index to create the output arrays desired
                        if len(keyshape) == 1:                      # if the input is 1D, shape=n, and index!=None, then tile the indexed value of length shape
                            if index in range(keyshape[0]):
                                return np.tile(val[index], shape)
                            else:
                                raise ValueError(f"Value for index '{index}' is not within the size of {val} (len={keyshape[0]})")
                        else:                                               # if the input is 2D, len(val)=shape, and index!=None
                            if index in range(keyshape[1]):
                                return np.array([v[index] for v in val])    # then pull the indexed value out of each row of 2D input
                            else:
                                raise ValueError(f"Value for index '{index}' is not within the size of {val} (len={keyshape[0]})")
                else:
                    raise ValueError(f"Value for key '{key}' is not the expected size of {shape} and is instead: {val}")

            else:                                            # must be expecting a multi-D array
                vala = np.array(val, dtype=dtype)            # make array

                if list(vala.shape) == shape:                      # if provided with the right shape
                    return vala
                elif len(shape) > 2:
                    raise ValueError("Function getFromDict isn't set up for shapes larger than 2 dimensions")
                elif vala.ndim==1 and len(vala)==shape[1]:   # if we expect an MxN array, and an array of size N is provided, tile it M times
                    return np.tile(vala, [shape[0], 1] )
                else:
                    raise ValueError(f"Value for key '{key}' is not a compatible size for target size of {shape} and is instead: {val}")

    else:
        if default == None:
            raise ValueError(f"Key '{key}' not found in input file...")
        else:
            if shape==0 or shape==-1:
                return default
            else:
                if np.isscalar(default):
                    return np.tile(default, shape)
                else:
                    return np.tile(default, [shape, 1])

def convertIEAturbineYAML2RAFT(fname_turbine):
    '''
    This loads data from a standard turbine YAML file and saves to corresponding info in RAFT format YAML.
    '''
    
    import wisdem.inputs as sch    # used for loading turbine YAML and using WISDEM validation process
    from wisdem.commonse.utilities import arc_length
    
    # dictionary to be filled in with turbine data
    d = dict(blade={}, airfoils=[], env={})
    
    # Load wind turbine geometry yaml
    print("Loading turbine YAML file: "+fname_turbine)
    
    run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
    fname_input_wt = os.path.join(run_dir, fname_turbine)
    wt_init = sch.load_geometry_yaml(fname_input_wt)
        
    print(f"'{wt_init['name']}'")
    
    # Conversion of the yaml inputs into CCBlade inputs
    Rhub = 0.5 * wt_init["components"]["hub"]["diameter"] # [m] - hub radius
    d['precone'] = np.rad2deg(wt_init["components"]["hub"]["cone_angle"]) # [deg] - rotor precone angle
    d['shaft_tilt'] = np.rad2deg(wt_init["components"]["nacelle"]["drivetrain"]["uptilt"]) # [deg] -  nacelle uptilt angle
    d['overhang'] = wt_init["components"]["nacelle"]["drivetrain"]["overhang"] # [m] - distance from tower centerline to blade root location?
    d['nBlades'] = wt_init["assembly"]["number_of_blades"] # [-] - number of blades

    # Set discretization parameters
    n_span = 30 # [-] - number of blade stations along span
    grid = np.linspace(0., 1., n_span) # equally spaced grid along blade span, root=0 tip=1

    # Blade quantities
    blade = wt_init["components"]["blade"]["outer_shape_bem"]
    rotor_diameter = wt_init["assembly"]["rotor_diameter"]
    blade_ref_axis = np.zeros((n_span, 3))
    blade_ref_axis[:, 0] = np.interp(grid, blade["reference_axis"]["x"]["grid"], blade["reference_axis"]["x"]["values"])
    blade_ref_axis[:, 1] = np.interp(grid, blade["reference_axis"]["y"]["grid"], blade["reference_axis"]["y"]["values"])
    blade_ref_axis[:, 2] = np.interp(grid, blade["reference_axis"]["z"]["grid"], blade["reference_axis"]["z"]["values"])
    if rotor_diameter != 0.0:
        blade_ref_axis[:, 2] = (blade_ref_axis[:, 2] * rotor_diameter / ((arc_length(blade_ref_axis)[-1] + Rhub) * 2.0))
    d['blade']['r'          ] = blade_ref_axis[1:-1, 2] + Rhub # [m] - radial position along straight blade pitch axis
    d['blade']['Rtip'       ] = blade_ref_axis[-1, 2] + Rhub
    d['blade']['chord'      ] = np.interp(grid[1:-1], blade["chord"]["grid"], blade["chord"]["values"]) # [m] - blade chord distributed along r
    d['blade']['theta'      ] = np.rad2deg(np.interp(grid[1:-1], blade["twist"]["grid"], blade["twist"]["values"])) # [deg] - blade twist distributed along r
    d['blade']['precurve'   ] = blade_ref_axis[1:-1, 0] # [m] - blade prebend distributed along r, usually negative for upwind rotors
    d['blade']['precurveTip'] = blade_ref_axis[-1, 0] # [m] - prebend at blade tip
    d['blade']['presweep'   ] = blade_ref_axis[1:-1, 1] # [m] - blade presweep distributed along r, usually positive
    d['blade']['presweepTip'] = blade_ref_axis[-1, 1] # [m] - presweep at blade tip

    # Hub height
    if wt_init["assembly"]["hub_height"] != 0.0:
        d['Zhub'] = wt_init["assembly"]["hub_height"]
    else:
        d['Zhub'] = wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"][-1] + wt_init["components"]["nacelle"]["drivetrain"]["distance_tt_hub"]

    d['Rhub'] = Rhub


    # Atmospheric boundary layer data
    d['env']['rho'     ] = wt_init['environment']["air_density"] # [kg/m3] - density of air
    d['env']['mu'      ] = wt_init['environment']["air_dyn_viscosity"] # [kg/(ms)] - dynamic viscosity of air
    d['env']['shearExp'] = wt_init['environment']["shear_exp"] # [-] - shear exponent


    # Airfoil data
    n_af = len(wt_init["airfoils"])
    
    d['blade']['airfoils'] = {'grid': blade["airfoil_position"]["grid"], 'labels': blade["airfoil_position"]["labels"]}
    
   
    for i in range(n_af):
        af_name = wt_init["airfoils"][i]["name"]
        
        afdict = {'name': af_name, 'relative_thickness': wt_init["airfoils"][i]["relative_thickness"], 
                  'key': ['alpha', 'c_l', 'c_d', 'c_m'], 'data': [] }

        if len( wt_init["airfoils"][i]["polars"] ) > 1:
            print(f"Warning for airfoil {af_name}, RAFT only uses one polar entry (the first one).")
        
        n = len(wt_init["airfoils"][i]["polars"][0]["c_l"]["grid"])
        
        for j in range(n):
        
            # make sure each of cl, cd, and cm have the same AOA values (we don't handle the weird alternative)
            if (    wt_init["airfoils"][i]["polars"][0]["c_l"]["grid"][j] == wt_init["airfoils"][i]["polars"][0]["c_d"]["grid"][j]
                and wt_init["airfoils"][i]["polars"][0]["c_l"]["grid"][j]== wt_init["airfoils"][i]["polars"][0]["c_m"]["grid"][j] ):
                
                afdict['data'].append( [ rad2deg(wt_init["airfoils"][i]["polars"][0]["c_l"]["grid"][j]),
                                                 wt_init["airfoils"][i]["polars"][0]["c_l"]["values"][j],
                                                 wt_init["airfoils"][i]["polars"][0]["c_d"]["values"][j],
                                                 wt_init["airfoils"][i]["polars"][0]["c_m"]["values"][j] ] )
            
            else:            
                raise ValueError(f"AOA values for airfoil {af_name} are not consistent between Cl, Cd, and Cm at index {j}.")
    
        d['airfoils'].append(afdict)
        
    # write YAML file manually for better formatting
    with open("test.yaml", 'w') as outfile:
        #yaml.dump(d, outfile)
        
        outfile.write(f"# RAFT-style YAML inputs for turbine \n\n")
        outfile.write(f"turbine:\n")
        outfile.write(f"    nBlades     : {d['nBlades']}     # number of blades\n")
        outfile.write(f"    Zhub        : {d['Zhub']}        # hub height [m]\n")
        outfile.write(f"    Rhub        : {d['Rhub']}        # hub radius [m]\n")
        outfile.write(f"    precone     : {d['precone']}     # [rad]\n")
        outfile.write(f"    shaft_tilt  : {d['shaft_tilt']}  # [rad]\n")
        outfile.write(f"    overhang    : {d['overhang']}  # [m]\n\n")
        outfile.write(f"    env: \n")
        outfile.write(f"        rho     : {d['env']['rho']}   # air density [kg/m^3]\n")
        outfile.write(f"        mu      : {d['env']['mu']}   # air dynamic viscosity\n")
        outfile.write(f"        shearExp: {d['env']['shearExp']}  # shear exponent\n\n")
        
        outfile.write(f"    blade: \n")
        outfile.write(f"        precurveTip : {d['blade']['precurveTip']}  # \n")
        outfile.write(f"        presweepTip : {d['blade']['presweepTip']}  # \n")
        outfile.write(f"        Rtip        : {d['blade']['Rtip']}         # rotor radius\n\n")
        
        outfile.write(f"        geometry: #  r     chord    theta  precurve  presweep\n")
        outfile.write(f"          #         (m)     (m)     (deg)     (m)      (m)\n")
        for i in range(len(d['blade']['r'])):
            outfile.write("          - [{:10.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f} ]\n".format(
                                      d['blade']['r'       ][i],
                                      d['blade']['chord'   ][i],
                                      d['blade']['theta'   ][i],
                                      d['blade']['precurve'][i],
                                      d['blade']['presweep'][i] ) )
        
        outfile.write(f"\n")
        outfile.write(f"        airfoils: # location   name \n")
        outfile.write(f"          #         (0-1)     (string)     name must match name in airfoils list\n")
        for i in range(len(d['blade']['airfoils']['grid'])):
            outfile.write("          - [  {:10.5f}, {:30} ]\n".format(d['blade']['airfoils']['grid'][i], d['blade']['airfoils']['labels'][i]) )
        
        
        
        outfile.write(f"\n    airfoils: \n")
        
        for iaf in range(n_af):
            outfile.write(f"      - name               : {d['airfoils'][iaf]['name']}  # \n")
            outfile.write(f"        relative_thickness : {d['airfoils'][iaf]['relative_thickness']}  # \n")
            outfile.write(f"        data:  #  alpha      c_l         c_d         c_m   \n")
            outfile.write(f"          #       (deg)      (-)         (-)         (-)\n")
            for j in range(len(d['airfoils'][iaf]['data'])):
                outfile.write("          - [{:10.2f}, {:10.5f}, {:10.5f}, {:10.5f} ] \n".format( *d['airfoils'][iaf]['data'][j] ))
        
        print("DIBE")
    
    return d
    
    
# ----- additional helper functions from Joep van der Spek -----                                 
    
    
def getUniqueCaseHeadings(keys, values):
    '''
    Function to obtain unique heading values for calculation in BEM.

    Parameters
    ----------
    keys : dict
        dict with keys of design data
    values : dict
        dict of all design data
    '''
    caseHeadings = []
    data = [dict(zip(keys, value)) for value in values]
    wave_headings = [float(data_head['wave_heading']) for data_head in data]
    wave_headings += [float(data_head['wave_heading2']) for data_head in data]
    for wh in wave_headings:
        if wh not in caseHeadings:
            caseHeadings.append(wh)
    # print(caseHeadings)
    maxHeading = max(caseHeadings)
    minHeading = min(caseHeadings)
    if len(caseHeadings) == 2:
        headingStep = maxHeading - minHeading
        numberOfHeadings = 2
    elif len(caseHeadings) > 2:
        headingStep = np.min(np.abs(np.diff(np.sort(caseHeadings))))
        numberOfHeadings = int((maxHeading - minHeading) / headingStep + 1)
        # this is different from only two headings, as it requires headings in between
    else:
        headingStep = 0
        numberOfHeadings = 1
    return caseHeadings, headingStep, numberOfHeadings # only have unique values in evaluated list, otherwise possibly issues with np.diff


def getSigmaXPSD(TBFA, TBSS, frequencies, angles=np.linspace(0,2*np.pi,50), d = 10, thickness= 0.083):
    """Function to retrieve Axial stress (sigma_x) around tower base circumference using tower base bending"""
    # angles = np.linspace(0,2*np.pi,dAngles) # Array with angles to calculate for anywhere around tower.

    angleMeshFA, TBFAMesh = np.meshgrid(angles, TBFA)
    angleMeshSS, TBSSMesh = np.meshgrid(angles, TBSS)
    # print('TBFA',TBFA)
    # print('TBFSS', TBSS)
    Izz = np.pi/8*thickness*d**3 # Bending moment of inertia, assume thin walled

    sigmaX = ((TBFAMesh*np.cos(angleMeshFA)-TBSSMesh*np.sin(angleMeshSS))*d/2)/Izz # Return?
    # print(sigmaX)
    # sigmaX = psdTBFAMesh*(np.cos(angleMeshFA)*d/2/Izz)**2+psdTBSSMesh*(np.sin(angleMeshSS)*d/2/Izz)**2
    ANGLESMesh, FREQMesh = np.meshgrid(angles, frequencies)
    return getPSD(sigmaX/10**6, frequencies[1]-frequencies[0]), ANGLESMesh, FREQMesh


def parametricAnalysisBuilder(design, changeType, startValueSensitivityStudy, parametricAnalysis= False):

    if parametricAnalysis:
        misalignmentAngle = getFromDict(design['parametricAnalysis'], 'misalignmentAngle', default=0)
        numMisalignAngles = getFromDict(design['parametricAnalysis'], 'numMisalign', dtype=int, default=0)
        if misalignmentAngle is not None and numMisalignAngles is not None and changeType == 'misalignment':
            design['cases']['data'][0][13] = startValueSensitivityStudy
            for misalign in range(numMisalignAngles):
                add_design = design['cases']['data'][0].copy()
                add_design[13] += misalignmentAngle * (misalign + 1)
                design['cases']['data'].append(add_design)

        windMisalignmentAngle = getFromDict(design['parametricAnalysis'], 'windMisalignmentAngle', default=0)
        numWindMisalignAngles = getFromDict(design['parametricAnalysis'], 'numWindMisalign', dtype=int, default=0)
        if windMisalignmentAngle is not None and numWindMisalignAngles is not None and changeType == 'windMisalignment':
            design['cases']['data'][0][1] = startValueSensitivityStudy
            for angle in range(numWindMisalignAngles):
                add_design = design['cases']['data'][0].copy()
                add_design[1] += windMisalignmentAngle * (angle + 1)
                design['cases']['data'].append(add_design)

        rotationAngle = getFromDict(design['parametricAnalysis'], 'rotationAngle', default=0)
        numRotations = getFromDict(design['parametricAnalysis'], 'numRotations', dtype=int, default=0)
        if rotationAngle is not None and numRotations is not None and changeType == 'floaterRotation':
            for misalign in range(numRotations):
                add_design = design['cases']['data'][0].copy()
                add_design[1] += rotationAngle * (misalign + 1)
                add_design[8] += rotationAngle * (misalign + 1)
                add_design[13] += rotationAngle * (misalign + 1)
                design['cases']['data'].append(add_design)

        windSpeedIncrement = getFromDict(design['parametricAnalysis'], 'windSpeedIncrement', default=0)
        numWSIncrements = getFromDict(design['parametricAnalysis'], 'numWSIncrements', dtype=int, default=0)
        if windSpeedIncrement is not None and numWSIncrements is not None and changeType == 'windSpeed':
            design['cases']['data'][0][0] = startValueSensitivityStudy
            for numIncr in range(numWSIncrements):
                add_design = design['cases']['data'][0].copy()
                add_design[0] += windSpeedIncrement * (numIncr + 1)
                design['cases']['data'].append(add_design)

        waveHeightIncrement1 = getFromDict(design['parametricAnalysis'], 'waveHeightIncrement1', default=0)
        numWHincrements1 = getFromDict(design['parametricAnalysis'], 'numWHIncrements1', dtype=int, default=0)
        if waveHeightIncrement1 is not None and numWHincrements1 is not None and changeType == 'waveHeight1':
            design['cases']['data'][0][7] = startValueSensitivityStudy
            for numIncr in range(numWHincrements1):
                add_design = design['cases']['data'][0].copy()
                add_design[7] += waveHeightIncrement1 * (numIncr + 1)
                design['cases']['data'].append(add_design)

        waveHeightIncrement2 = getFromDict(design['parametricAnalysis'], 'waveHeightIncrement2', default=0)
        numWHincrements2 = getFromDict(design['parametricAnalysis'], 'numWHIncrements2', dtype=int, default=0)
        if waveHeightIncrement2 is not None and numWHincrements2 is not None and changeType == 'waveHeight2':
            design['cases']['data'][0][12] = startValueSensitivityStudy
            for numIncr in range(numWHincrements2):
                add_design = design['cases']['data'][0].copy()
                add_design[12] += waveHeightIncrement2 * (numIncr + 1)
                design['cases']['data'].append(add_design)

        wavePeriodIncrement1 = getFromDict(design['parametricAnalysis'], 'wavePeriodIncrement1', default=0)
        numWPincrements1 = getFromDict(design['parametricAnalysis'], 'numWPIncrements1', dtype=int, default=0)
        if wavePeriodIncrement1 is not None and numWPincrements1 is not None and changeType == 'wavePeriod1':
            design['cases']['data'][0][6] = startValueSensitivityStudy
            for numIncr in range(numWPincrements1):
                add_design = design['cases']['data'][0].copy()
                add_design[6] += wavePeriodIncrement1 * (numIncr + 1)
                design['cases']['data'].append(add_design)

        wavePeriodIncrement2 = getFromDict(design['parametricAnalysis'], 'wavePeriodIncrement2', default=0)
        numWPincrements2 = getFromDict(design['parametricAnalysis'], 'numWPIncrements2', dtype=int, default=0)
        if wavePeriodIncrement2 is not None and numWPincrements2 is not None and changeType == 'wavePeriod2':
            design['cases']['data'][0][11] = startValueSensitivityStudy
            for numIncr in range(numWPincrements2):
                add_design = design['cases']['data'][0].copy()
                add_design[11] += wavePeriodIncrement2 * (numIncr + 1)
                design['cases']['data'].append(add_design)

        return design
    else:
        return design


def retrieveAxisParAnalysis(iCase, cases, changeType, variableXaxis, parametricAnalysisDict):

    if changeType == 'misalignment':
        variableXaxis.append(cases['wave_heading2'])
        string_x_axis = 'Misalignment second wave system [deg]'
        title_string = f'Misalignment WS 2 $= {variableXaxis[-1]:10.2f}$ [deg]'
    elif changeType == 'misalignment1':
        variableXaxis.append(cases['wave_heading1'])
        string_x_axis = 'Misalignment first wave system [deg]'
        title_string = f'Misalignment WS 1 $= {variableXaxis[-1]:10.2f}$ [deg]'
    elif changeType == 'windMisalignment':
        variableXaxis.append(cases['wind_heading'])
        string_x_axis = 'Wind heading [deg]'
        title_string = f'Misalignment Wind $= {variableXaxis[-1]:10.2f}$ [deg]'

    elif changeType == 'floaterRotation':
        rotationAngle = getFromDict(parametricAnalysisDict, 'rotationAngle')
        variableXaxis.append(iCase * rotationAngle)  # not robust, now only works for single base case being rotated
        string_x_axis = 'Floater rotation [deg]'
        title_string = f'Floater Rotation $= {variableXaxis[-1]:10.2f}$ [deg]'
    elif changeType == 'windSpeed':
        variableXaxis.append(cases['wind_speed'])
        string_x_axis = 'Average Wind Speed [m/s]'
        title_string = f'$U_{{ave}} = {variableXaxis[-1]:10.2f}$ [m/s]'
    elif changeType == 'waveHeight1':
        variableXaxis.append(cases['wave_height'])
        string_x_axis = 'Wave Height system 1 [m]'
        title_string = f'WS1 $H_{{s}}={variableXaxis[-1]:10.2f}$ [m]'
    elif changeType == 'waveHeight2':
        variableXaxis.append(cases['wave_height2'])
        string_x_axis = 'Wave Height system 2 [m]'
        title_string = f'WS2 $H_{{s}}={variableXaxis[-1]:10.2f}$ [m]'
    elif changeType == 'wavePeriod1':
        variableXaxis.append(cases['wave_period'])
        string_x_axis = 'Wave Period system 1 [s]'
        title_string = f'WS1 $T_{{p}}={variableXaxis[-1]:10.2f}$ [s]'
    elif changeType == 'wavePeriod2':
        variableXaxis.append(cases['wave_period2'])
        string_x_axis = 'Wave Period system 2 [s]'
        title_string = f'WS2 $T{{p}}={variableXaxis[-1]:10.2f}$ [s]'
    else:
        variableXaxis.append(iCase)
        string_x_axis = 'Case number'
        title_string = f'Base Case {iCase+1}'

    return variableXaxis, string_x_axis, title_string


def plotFloaterRotation_min_max_moments(equivalent_moment_list, floater_rotations):

    max_moment = []
    min_moment = []

    for index, equivalent_moment_list_in_list in enumerate(equivalent_moment_list):
        max_moment.append([np.amax(equivalent_moment_list_in_list)])
        min_moment.append([np.amax(equivalent_moment_list_in_list)])

    fig, ax = plt.subplots(figsize=get_figsize(400))
    ax.plot(floater_rotations, max_moment, label='Max eq stress')
    ax.plot(floater_rotations, min_moment, label='Min eq stress')
    ax.set_xlabel('Floater Rotation angle [deg]')
    ax.set_ylabel('Equivalent Moment [MNm]')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Equivalent Moment for changing different floater rotations')
    ax.legend()
    ax.grid()


def get_figsize(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """

    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


def printCaseToTable(cases, keys):
    parameters = ['U$_{ave}$ [m/s]', 'Turbulence Intensity [-]', 'Misalignment Angle [deg]', 'Yaw error [deg]'
                  ,'T$_p$ system 1 [s]', 'H$_s$ system 1 [m]', '$\gamma$ system 1 [-]', 'Misalignment angle system 1 [deg]'
                  ,'T$_p$ system 2 [s]', 'H$_s$ system 2 [m]', '$\gamma$ system 2 [-]', 'Misalignment angle system 2 [deg]']
    for iCase, item in enumerate(cases):
        case = dict(zip(keys, cases[iCase]))
        parameters[0] += f" & {case['wind_speed']:.2f}"
        parameters[1] += f" & {case['turbulence']:.2f}"
        parameters[2] += f" & {case['wind_heading']:.2f}"
        parameters[3] += f" & {case['yaw_misalign']:.2f}"
        parameters[4] += f" & {case['wave_period']:.2f}"
        parameters[5] += f" & {case['wave_height']:.2f}"
        parameters[6] += f" & {case['gamma_ws1']:.2f}"
        parameters[7] += f" & {case['wave_heading']:.2f}"
        parameters[8] += f" & {case['wave_period2']:.2f}"
        parameters[9] += f" & {case['wave_height2']:.2f}"
        parameters[10] += f" & {case['gamma_ws2']:.2f}"
        parameters[11] += f" & {case['wave_heading2']:.2f}"

    for p, temp in enumerate(parameters):
        parameters[p] += f' \\\ \hline'
        print(parameters[p])


def adjustMooring(ms, design):
    '''
    ms: moorpy system 
    design: RAFT input dictionary 
    
    Returns updated design dictionary to match moorpy mooring system (hackish method - will not work for every mooring system!)
    '''
    
    design['mooring']['water_depth'] = ms.depth
    for i, linetype in enumerate(ms.lineTypes):
        #design['mooring']['line_types'][i]['name'] = linetype
        design['mooring']['line_types'][i]['diameter'] = ms.lineTypes[linetype]['input_d']
        design['mooring']['line_types'][i]['mass_density'] = ms.lineTypes[linetype]['m']
        design['mooring']['line_types'][i]['stiffness'] = ms.lineTypes[linetype]['EA']
   
    nLines = len(ms.lineList)
    for i in range(nLines):
        design['mooring']['points'][i]['location'] = list(ms.pointList[i*2].r)
        design['mooring']['points'][i+3]['location'] = list(ms.pointList[i*2+1].r)
    
        design['mooring']['lines'][i]['length'] = ms.lineList[i].L
    
    return(design)


if __name__ == '__main__':
    
    
    d = convertIEAturbineYAML2RAFT('../designs/rotors/IEA-15-240-RWT.yaml')
