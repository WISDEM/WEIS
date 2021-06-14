import numpy as np
import os
import weis.inputs as sch

class DLCInstance(object):
    
    def __init__(self):
        # Set default DLC with empty properties
        self.wind_speed = 0.0
        self.wind_heading = 0.0
        self.yaw_misalign = 0.0
        self.turbsim_seed = 0
        self.turbine_status = '' # Could make this True/False?
        self.wave_spectrum = ''
        self.turbulent_wind = '' # Is this NTM/ETM/None?
        self.label = '' # For 1.1/Custom

    def to_dict(self):
        out = {}
        keys = ['wind_speed','wind_heading','yaw_misalign','turbsim_seed','turbine_status',
                'wave_spectrum','turbulent_wind','label']
        for k in keys:
            out[k] = getattr(self, k)
        return out


        
class DLCGenerator(object):

    def __init__(self, cut_in=4.0, cut_out=25.0, rated=10.0):
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.rated = rated
        self.cases = []
        self.rng = np.random.default_rng()

    def to_dict(self):
        return [m.to_dict() for m in self.cases]
    
    def generate(self, label, options):
        known_dlcs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 6.1, 6.2, 6.3]
        
        found = False
        for ilab in known_dlcs:
            func_name = 'generate_'+str(ilab).replace('.','p')
            
            if label in [ilab, str(ilab)]: # Match either 1.1 or '1.1'
                found = True
                getattr(self, func_name)(options) # calls self.generate_1p1(options)
                break
            
        if not found:
            raise ValueError(f'DLC {label} is not currently supported')

        
    def generate_custom(self, options):
        pass

    def generate_1p1(self, options):

        self.NTM(options)

    
    def NTM(self, options):
        wind_speeds = np.arange(self.cut_in, self.cut_out+1.0, options['ws_bin_size'])
        if wind_speeds[-1] != self.cut_out:
            wind_speeds = np.append(wind_speeds, self.cut_out)
            
        seeds = self.rng.integers(2147483648, size=options['n_seeds'])

        for ws in wind_speeds:
            for seed in seeds:
                idlc = DLCInstance()
                idlc.wind_speed = ws
                idlc.turbsim_seed = seed
                idlc.turbulent_wind = 'NTM'
                idlc.turbine_status = 'operating'
                idlc.label = '1.1'
                self.cases.append(idlc)


if __name__ == "__main__":
    
    # Wind turbine inputs that will eventually come in from somewhere
    cut_in = 4.
    cut_out = 25.
    rated = 10.

    # Load modeling options file
    weis_dir                = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
    fname_modeling_options = os.path.join(weis_dir , "examples", "05_IEA-3.4-130-RWT", "modeling_options.yaml")
    modeling_options = sch.load_modeling_yaml(fname_modeling_options)
    
    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']
    
    # Initialize the generator
    dlc_generator = DLCGenerator(cut_in, cut_out, rated)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

    print(dlc_generator.cases[43].wind_speed)
                
