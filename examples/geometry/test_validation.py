import weis.yaml.validation as val
import os

run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep

spar    = val.load_yaml( os.path.join(run_dir,'nrel5mw-spar_oc3.yaml') )
semi    = val.load_yaml( os.path.join(run_dir,'nrel5mw-semi_oc4.yaml') )

yaml_schema = val.load_yaml(val.fschema_geom)

val.DefaultValidatingDraft7Validator(yaml_schema).validate(spar)
val.DefaultValidatingDraft7Validator(yaml_schema).validate(semi)
