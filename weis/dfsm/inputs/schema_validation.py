import os
from wisdem.inputs import validate_with_defaults

schema_dir = os.path.dirname(os.path.abspath(__file__))

def load_dfsm_yaml(finput):
    rosco_schema = os.path.join(schema_dir,'dfsm_schema.yaml')
    return validate_with_defaults(finput, rosco_schema)


if __name__=='__main__':

    this_dir = os.path.dirname(os.path.abspath(__file__))
    fname = this_dir + os.sep + 'example_schema.yaml'
    #new_input = load_dfsm_yaml(fname)
    
    print('here')

