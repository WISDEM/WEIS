
# make all classes and functions in MoorPy.py available under the main package
#from moorpy.MoorPy import *     

# new core MoorPy imports to eventually replace the above 
from moorpy.line import Line    # this will make mp.Line available from doing "import moorpy as mp"
from moorpy.point import Point
from moorpy.body import Body
from moorpy.lineType import LineType
from moorpy.system import System

from moorpy.helpers import *
from moorpy.Catenary import catenary
