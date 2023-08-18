#from .raft_model import Model

import raft.raft_model as model

from importlib import reload
Model = reload(model).Model
