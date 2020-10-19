import jsonschema as json

try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

import os
fdefaults_geom  = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'geometry_defaults.yaml')
fschema_geom    = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'geometry_schema.yaml')
fdefaults_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modeling_defaults.yaml')
fschema_model   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'modeling_schema.yaml')
fdefaults_opt   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysis_defaults.yaml')
fschema_opt     = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysis_schema.yaml')
    
                
    
#---------------------
def load_yaml(fname_input):
    with open(fname_input, 'r') as f:
        input_yaml = ry.load(f, Loader=ry.Loader)
    return input_yaml
#---------------------
def write_yaml(instance, foutput):
    # Write yaml with updated values
    yaml=ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    with open(foutput, 'w') as f:
        yaml.dump(instance, f)
    
#---------------------
# This is for when the defaults are in another file
def nested_get(indict, keylist):
    rv = indict
    for k in keylist:
        rv = rv[k]
    return rv

def nested_set(indict, keylist, val):
    rv = indict
    for k in keylist:
        if k == keylist[-1]:
            rv[k] = val
        else:
            rv = rv[k]

def integrate_defaults(instance, defaults, yaml_schema):
    # Prep iterative validator
    #json.validate(self.wt_init, yaml_schema)
    validator = json.Draft7Validator(yaml_schema)
    errors = validator.iter_errors(instance)

    # Loop over errors
    for e in errors:
        if e.validator == 'required':
            for k in e.validator_value:
                if not k in e.instance.keys():
                    mypath = e.absolute_path.copy()
                    mypath.append(k)
                    v = nested_get(defaults, mypath)
                    if isinstance(v, dict) or isinstance(v, list) or v in ['name','material']:
                        # Too complicated to just copy over default, so give it back to the user
                        raise(e)
                    else:
                        print('WARNING: Missing value,',list(mypath),', so setting to:',v)
                        nested_set(instance, mypath, v)
        else:
            raise(e)
    return instance
#---------------------
# See: https://python-jsonschema.readthedocs.io/en/stable/faq/#why-doesn-t-my-schema-s-default-property-set-the-default-on-my-instance
def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if 'default' in subschema:
                instance.setdefault(property, subschema['default'])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return json.validators.extend(validator_class, {"properties" : set_defaults})
DefaultValidatingDraft7Validator = extend_with_default(json.Draft7Validator)

def validate_without_defaults(finput, fschema):
    yaml_schema = load_yaml(fschema) if type(fschema) == type('') else fschema
    myobj = load_yaml(finput) if type(finput) == type('') else finput
    json.Draft7Validator(yaml_schema).validate(myobj)
    return myobj
    
def validate_with_defaults(finput, fschema):
    yaml_schema = load_yaml(fschema) if type(fschema) == type('') else fschema
    myobj = load_yaml(finput) if type(finput) == type('') else finput
    DefaultValidatingDraft7Validator(yaml_schema).validate(myobj)
    return myobj
#---------------------
def load_default_geometry_yaml():
    return load_yaml(fdefaults_geom)

def load_geometry_yaml(finput):
    return validate_with_defaults(finput, fschema_geom)

def write_geometry_yaml(instance, foutput):
    validate_without_defaults(instance, fschema_geom)
    sfx_str = '.yaml'
    if foutput[-5:] == sfx_str:
        sfx_str = ''
    write_yaml(instance, foutput+sfx_str)
    
def load_modeling_yaml(finput):
    return validate_with_defaults(finput, fschema_model)

def write_modeling_yaml(instance, foutput):
    validate_without_defaults(instance, fschema_model)
    sfx_str = '.yaml'
    if foutput[-5:] == sfx_str:
        sfx_str = ''
    write_yaml(instance, foutput+sfx_str)
    
def load_analysis_yaml(finput):
    return validate_with_defaults(finput, fschema_opt)

def write_analysis_yaml(instance, foutput):
    validate_without_defaults(instance, fschema_opt)
    sfx_str = '.yaml'
    if foutput[-5:] == sfx_str:
        sfx_str = ''
    write_yaml(instance, foutput+sfx_str)

if __name__ == '__main__':
    yaml_schema = load_yaml(fschema_opt)
    myobj = load_yaml(fdefaults_opt)
    DefaultValidatingDraft7Validator(yaml_schema).validate(myobj)
    #validator.validate( myobj )
    print([k for k in myobj.keys()])
    print(myobj['general'])
    
    obj = {}
    schema = {'properties': {'foo': {'default': 'bar'}}}
    # Note jsonschem.validate(obj, schema, cls=DefaultValidatingDraft7Validator)
    # will not work because the metaschema contains `default` directives.
    DefaultValidatingDraft7Validator(schema).validate(obj)
    print(obj)
    

    
