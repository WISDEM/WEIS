
import numpy as np
from ctypes import (
	CDLL,
	POINTER,
	pointer,
	c_bool,
	c_char,
	c_char_p,
	c_int,
	c_float,
	c_double,
	c_longdouble,
	create_string_buffer,
	byref,
	get_errno,
	sizeof
)
import MoorDyn_Types

# Load the MoorDyn shared library
moordynlib = CDLL('/Users/rmudafor/Development/openfast/build/modules/moordyn/libmoordynlib.dylib')

# Initialize the data for MD_Init
error_message = create_string_buffer(1025)
error_status = pointer(c_int(0))

init_input_type = MoorDyn_Types.MD_InitInputType_t()
input_type = MoorDyn_Types.MD_InputType_t()
parameter_type = MoorDyn_Types.MD_ParameterType_t()
continuous_state_type = MoorDyn_Types.MD_ContinuousStateType_t()
discrete_state_type = MoorDyn_Types.MD_DiscreteStateType_t()
constraint_state_type = MoorDyn_Types.MD_ConstraintStateType_t()
other_state_type = MoorDyn_Types.MD_OtherStateType_t()
output_type = MoorDyn_Types.MD_OutputType_t()
misc_var_type = MoorDyn_Types.MD_MiscVarType_t()
init_output_type = MoorDyn_Types.MD_InitOutputType_t()

time_step = c_double(0.01)
# utimes = c_float * 2
# utimes(0.0, 0.1)

init_input_type.g = c_float(9.81)
init_input_type.rhoW = c_float(1025)
init_input_type.WtrDepth = c_float(10.0)
init_input_type.PtfmInit = (6 * c_float)(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
init_input_type.FileName = b"MoorDyn.dat"
init_input_type.RootName = b"MoorDyn.MD"

# Configure the MoorDyn functions
moordynlib.MD_Init_C.argtypes = [
	POINTER(MoorDyn_Types.MD_InitInputType_t),
	POINTER(MoorDyn_Types.MD_InputType_t),
	POINTER(MoorDyn_Types.MD_ParameterType_t),
	POINTER(MoorDyn_Types.MD_ContinuousStateType_t),
	POINTER(MoorDyn_Types.MD_DiscreteStateType_t),
	POINTER(MoorDyn_Types.MD_ConstraintStateType_t),
	POINTER(MoorDyn_Types.MD_OtherStateType_t),
	POINTER(MoorDyn_Types.MD_OutputType_t),
	POINTER(MoorDyn_Types.MD_MiscVarType_t),
	POINTER(c_double),
    POINTER(MoorDyn_Types.MD_InitOutputType_t),
	# INTEGER(IntKi),                 	:: ErrStat
	# CHARACTER(*),                   	:: ErrMsg
]
moordynlib.MD_Init_C.restype = c_int

moordynlib.MD_UpdateStates_C.argtypes = [
	c_double,
	c_int,
	POINTER(MoorDyn_Types.MD_InputType_t),
	# POINTER(c_float),
	POINTER(MoorDyn_Types.MD_ParameterType_t),
	POINTER(MoorDyn_Types.MD_ContinuousStateType_t),
	POINTER(MoorDyn_Types.MD_DiscreteStateType_t),
	POINTER(MoorDyn_Types.MD_ConstraintStateType_t),
	POINTER(MoorDyn_Types.MD_OtherStateType_t),
	POINTER(MoorDyn_Types.MD_MiscVarType_t),
	# INTEGER(IntKi),                 	:: ErrStat
	# CHARACTER(*),                   	:: ErrMsg
]
moordynlib.MD_UpdateStates_C.restype = c_int

# moordynlib.MD_End_C.argtypes = [
# 	POINTER(MoorDyn_Types.MD_InputType_t),
# 	POINTER(MoorDyn_Types.MD_ParameterType_t),
# 	POINTER(MoorDyn_Types.MD_ContinuousStateType_t),
# 	POINTER(MoorDyn_Types.MD_DiscreteStateType_t),
# 	POINTER(MoorDyn_Types.MD_ConstraintStateType_t),
# 	POINTER(MoorDyn_Types.MD_OtherStateType_t),
# 	POINTER(MoorDyn_Types.MD_OutputType_t),
# 	POINTER(MoorDyn_Types.MD_MiscVarType_t),
# 	# INTEGER(IntKi),                 	:: ErrStat
# 	# CHARACTER(*),                   	:: ErrMsg
# ]
# moordynlib.MD_End_C.restype = c_int




# Run the simulation
result = moordynlib.MD_Init_C(
	byref(init_input_type),
	byref(input_type),
	byref(parameter_type),
	byref(continuous_state_type),
	byref(discrete_state_type),
	byref(constraint_state_type),
	byref(other_state_type),
	byref(output_type),
	byref(misc_var_type),
	byref(time_step),
	byref(init_output_type)
)

for t in [0]: #, 0.1]:
	moordynlib.MD_UpdateStates_C(
		byref(c_double(t)),
		byref(c_int(1)),
		byref(c_int(1)),
		byref(input_type),
		# utimes,
		byref(parameter_type),
		byref(continuous_state_type),
		byref(discrete_state_type),
		byref(constraint_state_type),
		byref(other_state_type),
		byref(misc_var_type),
	)

# moordynlib.MD_End_C(
# 	byref(input_type),
# 	byref(parameter_type),
# 	byref(continuous_state_type),
# 	byref(discrete_state_type),
# 	byref(constraint_state_type),
# 	byref(other_state_type),
# 	byref(output_type),
# 	byref(misc_var_type),
# )











# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm

# from subprocess import Popen
# from shutil import copyfile

# from ctypes import *

# #make data type
# #doublepp = np.ctypeslib.ndpointer(dtype=np.uintp) 

# nCD = 6 # number of coupling degrees of freedom

# # make MoorDyn function prototypes, parameter lists; load it and setup functions

# MDInitProto = CFUNCTYPE(c_int, c_double*nCD, c_double*nCD)  # remember, first entry is return type, rest are args
# MDCalcProto = CFUNCTYPE(c_int, c_double*nCD, c_double*nCD, c_double*nCD, c_double*1, c_double*1)
# MDClosProto = CFUNCTYPE(c_int)

# MDInitParams = (1, "x"), (1, "xd")
# MDCalcParams = (1, "x"), (1, "xd"), (2, "f"), (1, "t"), (1, "dtC")  # don't call this with the output parameter, it will be returned instead
# MDFairParams = (1, "l"), (2, "f")

# MDdll = WinDLL("MoorDyn2.dll")   # load MoorDyn DLL

# #MDdll.MoorDynInit.argtypes  = (POINTER(c_double), POINTER(c_double))  # remember, first entry is return type, rest are args
# #MDdll.MoorDynStep.argtypes  = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double))
# #MDdll.MoorDynClose.argtypes = ()
# #
# #MDdll.MoorDynInit.restypes = c_int
# #MDdll.MoorDynStep.restypes = c_int
# #MDdll.MoorDynClose.restypes= None

# MDInit = MDInitProto(("MoorDynInit", MDdll), MDInitParams)
# MDCalc = MDCalcProto(("MoorDynStep", MDdll), MDCalcParams)
# MDClose= MDClosProto(("MoorDynClose", MDdll))




# # ------------------ run MoorDyn ---------------------
	
# x1  = (c_double*6)()			# initialize some arrays for communicating with MoorDyn
# xd1 = (c_double*6)()


# t  = (c_double*1)()	 	# pointer to t
# dt = (c_double*1)()		# pointer to dt

# x  = (c_double*6)()
# xd = (c_double*6)()
# f  = (c_double*6)()	 		# fairlead force (first 3) or platform force (all 6)


# # parameters
# dtC = 0.02   # coupling time step size
# a1 = 3  # amplitude
# w1 = 2*np.pi/20 # frequency
# df = 0   # phase difference
# depth = 2  # depth below water of circle center

# ts = np.arange(0,90,dtC)


# # initialize MoorDyn

# phi = -70*np.pi/180

# x1[0] = 0
# x1[1] = 0
# x1[2] = -depth
# x1[3] = -np.cos(phi)
# x1[4] = 0 
# x1[5] = -np.sin(phi)

# #for i in range(6):
# #	xd[i]=0

# MDInit(x1,xd1)

# t[0] = 0
# dt[0] = 0.2

# xold = np.zeros(6)

# print("STARTING")


# for i in range(len(ts)):
# 	t[0] = ts[i]
# 	dt[0] = dtC
	
# 	xa = (1.0 - np.exp(-0.1*ts[i]) ) * a1*np.sin(w1*ts[i])              
# 	za = (1.0 - np.exp(-0.1*ts[i]) ) * a1*np.cos(w1*ts[i]) - depth
	
# 	x[0] = xa
# 	x[1] = 0.0
# 	x[2] = za
# 	x[3] = -np.cos(phi)
# 	x[4] = 0 
# 	x[5] = -np.sin(phi)
		
# 	# calculate velocities with finite difference
# 	if i==0:
# 		for k in range(6):
# 			xd[  k] = 0.0
# 			xold[k] = 0.0
# 	else:
# 		for k in range(6):
# 			xd  [k] =(x[k] - xold[k])/dtC
# 			xold[k] = x[k]
				
# 	status = MDCalc(x, xd, t, dt)		
	
# #	if status<0:
# #		print("MoorDynStep returned error")
# #		break
		
		
# MDClose()
