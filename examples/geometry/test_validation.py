import weis.yaml.validation as val
import os
import unittest

run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep

yaml_schema = val.load_yaml(val.fschema_geom)


class TestValidation(unittest.TestCase):
    
    def test_NREL5MW_Spar(self):
        spar    = val.load_yaml( os.path.join(run_dir,'nrel5mw-spar_oc3.yaml') )
        a = val.DefaultValidatingDraft7Validator(yaml_schema).validate(spar)
        self.assertIsNone(a)
        
    def test_NREL5MW_Semi(self):
        semi    = val.load_yaml( os.path.join(run_dir,'nrel5mw-semi_oc4.yaml') )
        a = val.DefaultValidatingDraft7Validator(yaml_schema).validate(semi)
        self.assertIsNone(a)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestValidation))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
