from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, DFSM_Turbine, Level3_Turbine


class LFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb,warmstart_file = None,scaling_dict = None,multi_fid_dict = None):

        super(LFTurbine, self).__init__(desvars_init,warmstart_file)
        self.n_count = 0
        self.warmstart_file = warmstart_file
        self.scaling_dict = scaling_dict
        self.multi_fid_dict = multi_fid_dict
        self.lf_turb = DFSM_Turbine(mf_turb)

    def compute(self, desvars):
        scaling_dict = self.scaling_dict
        outputs = self.lf_turb.compute(desvars,scaling_dict)
        self.n_count+=1

        if not(self.multi_fid_dict == None):

            obj1 = self.multi_fid_dict['obj1']
            obj2 = self.multi_fid_dict['obj2']
            wt_obj1 = self.multi_fid_dict['w1']
            wt_obj2 = self.multi_fid_dict['w2']

            o1 = outputs[obj1]
            o2 = outputs[obj2]

            outputs['wt_objectives'] = wt_obj1*o1 + wt_obj2*o2
        
        return outputs
        
        
class HFTurbine(BaseModel):
    
    def __init__(self, desvars_init, mf_turb,warmstart_file = None,scaling_dict = None,multi_fid_dict = None):

        super(HFTurbine, self).__init__(desvars_init,warmstart_file)
        self.l3_turb = Level3_Turbine(mf_turb)
        self.warmstart_file = warmstart_file
        self.multi_fid_dict = multi_fid_dict
        self.scaling_dict = scaling_dict
        self.n_count = 0

    def compute(self, desvars):
        scaling_dict = self.scaling_dict
        
        outputs = self.l3_turb.compute(desvars,scaling_dict)
        self.n_count+=1

        if not(self.multi_fid_dict == None):

            obj1 = self.multi_fid_dict['obj1']
            obj2 = self.multi_fid_dict['obj2']
            wt_obj1 = self.multi_fid_dict['w1']
            wt_obj2 = self.multi_fid_dict['w2']

            o1 = outputs[obj1]
            o2 = outputs[obj2]

            outputs['wt_objectives'] = wt_obj1*o1 + wt_obj2*o2
        
        return outputs
        
        

