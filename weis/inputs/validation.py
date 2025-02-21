import os
import jsonmerge
import wisdem.inputs
from wisdem.inputs.validation import load_yaml, write_yaml, _validate, simple_types, DefaultValidatingDraft7Validator
import rosco.toolbox.inputs
from openfast_io.FileTools import remove_numpy

froot_wisdem           = os.path.dirname(wisdem.inputs.__file__)
fschema_geom_wisdem    = os.path.join(froot_wisdem, 'geometry_schema.yaml')
fschema_model_wisdem   = os.path.join(froot_wisdem, 'modeling_schema.yaml')
fschema_opt_wisdem     = os.path.join(froot_wisdem, 'analysis_schema.yaml')

froot_rosco            = os.path.dirname(rosco.toolbox.inputs.__file__)
fschema_model_rosco    = os.path.join(froot_rosco, 'toolbox_schema.yaml')

froot           = os.path.dirname(os.path.realpath(__file__))
fdefaults_geom  = os.path.join(froot, 'geometry_defaults.yaml')
fschema_geom    = os.path.join(froot, 'geometry_schema.yaml')
fschema_model   = os.path.join(froot, 'modeling_schema.yaml')
fschema_opt     = os.path.join(froot, 'analysis_schema.yaml')

fschema_openfast       = os.path.join(froot, 'openfast_schema.yaml')
#---------------------
def load_default_geometry_yaml():
    return load_yaml(fdefaults_geom)

def get_geometry_schema():
    wisdem_schema = load_yaml(fschema_geom_wisdem)
    weis_schema   = load_yaml(fschema_geom)
    merged_schema = jsonmerge.merge(wisdem_schema, weis_schema)
    return merged_schema

def load_geometry_yaml(finput):
    merged_schema = get_geometry_schema()
    return _validate(finput, merged_schema, defaults=True)

def write_geometry_yaml(instance, foutput):
    merged_schema = get_geometry_schema()
    _validate(instance, merged_schema, defaults=False)
    sfx_str = '.yaml'
    if foutput[-5:] == sfx_str:
        sfx_str = ''
    write_yaml(instance, foutput+sfx_str)

def get_modeling_schema():
    wisdem_schema = load_yaml(fschema_model_wisdem)
    rosco_schema  = load_yaml(fschema_model_rosco)
    openfast_schema = load_yaml(fschema_openfast)
    weis_schema   = load_yaml(fschema_model)

    # Merge ROSCO options and update modeling
    merged_rosco_schema = jsonmerge.merge(rosco_schema['properties']['controller_params'], weis_schema['properties']['ROSCO'])
    merged_rosco_schema['properties']['linmodel_tuning'] = rosco_schema['properties']['linmodel_tuning']
    weis_schema['properties']['ROSCO'].update(merged_rosco_schema)
    
    # Update WEIS schema with WISDEM schema
    weis_schema['properties']['WISDEM'].update( wisdem_schema['properties']['WISDEM'] )

    # Update WEIS schema with OpenFAST schema
    weis_schema['properties']['OpenFAST'].update( openfast_schema['properties']['OpenFAST'] )


    return weis_schema

def load_modeling_yaml(finput):
    weis_schema = get_modeling_schema()
    return _validate(finput, weis_schema, defaults=True)

def write_modeling_yaml(instance, foutput):
    weis_schema = get_modeling_schema()
    sfx_str = ".yaml"
    if foutput[-5:] == sfx_str:
        foutput = foutput[-5:]
    elif foutput[-4:] == ".yml":
        foutput = foutput[-4:]
    sfx_str = "-modeling.yaml"
    instance2 = simple_types(instance)
    instance2 = remove_numpy(instance2)
    _validate(instance2, weis_schema, defaults=False)
    write_yaml(instance2, foutput+sfx_str)

def get_analysis_schema():
    wisdem_schema = load_yaml(fschema_opt_wisdem)
    weis_schema   = load_yaml(fschema_opt)
    merged_schema = jsonmerge.merge(wisdem_schema, weis_schema)
    return merged_schema

def load_analysis_yaml(finput):
    merged_schema = get_analysis_schema()
    return _validate(finput, merged_schema, defaults=True)

def write_analysis_yaml(instance, foutput):
    merged_schema = get_analysis_schema()
    _validate(instance, merged_schema, defaults=False)
    sfx_str = ".yaml"
    if foutput[-5:] == sfx_str:
        foutput = foutput[-5:]
    elif foutput[-4:] == ".yml":
        foutput = foutput[-4:]
    sfx_str = "-analysis.yaml"
    write_yaml(instance, foutput+sfx_str)

def re_validate_modeling(modeling_dict):
    fschema = get_modeling_schema()
    yaml_schema = load_yaml(fschema) if isinstance(fschema, type("")) else fschema
    DefaultValidatingDraft7Validator(yaml_schema).validate(modeling_dict)

def re_validate_analysis(analysis_dict):
    fschema = get_analysis_schema()
    yaml_schema = load_yaml(fschema) if isinstance(fschema, type("")) else fschema
    DefaultValidatingDraft7Validator(yaml_schema).validate(analysis_dict)
