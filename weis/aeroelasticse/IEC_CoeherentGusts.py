import numpy as np
import os
from weis.aeroelasticse import turbsim_file

class IEC_CoherentGusts():

    def __init__(self):

        self.Vert_Slope = 0 # Vertical slope of the wind inflow (deg)
        self.T0 = 0.
        self.TF = 630.
        self.TStart = 30 # Time to start transient conditions (s)
        self.dt = 0.05 # Transient wind time step (s)
        self.D = 90. # Rotor diameter (m)
        self.HH = 100. # Hub height (m)        

    def execute(self, dir, base_name, dlc):

        if self.HH > 60:
            self.Lambda_1 = 42
        else:
            self.Lambda_1 = 0.7*self.HH

        wind_file_name = os.path.join(dir, base_name + '_' + dlc.IEC_WindType + '_U%1.6f'%dlc.URef +  '_D%s'%dlc.direction_pn + '_S%s'%dlc.shear_hv + '.wnd')

        if dlc.IEC_WindType == 'EOG':
            self.EOG(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'EDC':
            self.EDC(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'ECD':
            self.ECD(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'EWS':
            self.EWS(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'Ramp':
            self.Ramp(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'Step':
            self.Step(dlc, wind_file_name)
        elif dlc.IEC_WindType == 'Custom':
            wind_file_name = dlc.wind_file
        else:
            raise Exception('The gust ' + dlc.IEC_WindType + ' is not supported by WEIS')

        return wind_file_name

    def EOG(self, dlc, wind_file_name):
        # Extreme operating gust: 6.3.2.2

        T = 10.5
        t = np.linspace(0., T, num=int(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = dlc.URef*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = dlc.URef*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = dlc.sigma1
        V_e1 = dlc.V_e1

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        V_gust = min([ 1.35*(V_e1 - V_hub), 3.3*(sigma_1/(1+0.1*(self.D/self.Lambda_1))) ])

        V_gust_t = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V_gust_t[i] = 0. - 0.37*V_gust*np.sin(3*np.pi*ti/T)*(1-np.cos(2*np.pi*ti/T))
            else:
                V_gust_t[i] = 0.
        
        data = np.column_stack((t, V, V_dir, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust_t, upflow))
        # Header
        hd = '! EOG gust, wind speed = ' + str(dlc.URef)
        self.write_wnd(wind_file_name, data, hd)

    def EDC(self, dlc, wind_file_name):
        # Extreme direction change: 6.3.2.4
        
        T = 6.
        t = np.linspace(0., T, num=int(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = dlc.URef*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = dlc.URef*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = dlc.sigma1

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        # Transient
        Theta_e = 4.*np.arctan(sigma_1/(V_hub*(1.+0.01*(self.D/self.Lambda_1))))*180./np.pi
        if Theta_e > 180.:
            Theta_e = 180.

        if dlc.direction_pn == 'p':
            k=1.
        elif dlc.direction_pn == 'n':
            k=-1.
        else:
            raise Exception('The EDC gust can only have positive (p) or negative (n) direction, whereas the script receives '+ dlc.direction_pn)

        Theta = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                Theta[i] = k * 0.5*Theta_e*(1-np.cos(np.pi*ti/T))
            else:
                Theta[i] = k * Theta_e


        data = np.column_stack((t, V, Theta, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
        # Header
        hd = '! EDC gust, wind speed = ' + str(dlc.URef) + ', direction ' + dlc.direction_pn
        self.write_wnd(wind_file_name, data, hd)
    
    def ECD(self, dlc, wind_file_name):
        # Extreme coherent gust with direction change: 6.3.2.5
        
        T = 10.
        t = np.linspace(0., T, num=int(T/self.dt+1))

        # constants from standard
        alpha = 0.2
        V_cg = 15 #m/s

        # Flow angle adjustments
        V_hub = dlc.URef*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = dlc.URef*np.sin(self.Vert_Slope*np.pi/180)

        # Contant variables
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        # Transient
        if V_hub < 4:
            Theta_cg = 180
        else:
            Theta_cg = 720/V_hub

        if dlc.direction_pn == 'p':
            k=1.
        elif dlc.direction_pn == 'n':
            k=-1.
        else:
            raise Exception('The ECD gust can only have positive (p) or negative (n) direction, whereas the script receives '+ dlc.direction_pn)
        Theta = np.zeros_like(t)
        V = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V[i] = V_hub + 0.5*V_cg*(1-np.cos(np.pi*ti/T))
                Theta[i] = k * 0.5*Theta_cg*(1-np.cos(np.pi*ti/T))
            else:
                V[i] = V_hub+V_cg
                Theta[i] = k * Theta_cg
      
        data = np.column_stack((t, V, Theta, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
        # Header
        hd = '! ECD gust, wind speed = ' + str(dlc.URef) + ', direction ' + dlc.direction_pn
        self.write_wnd(wind_file_name, data, hd)

    def EWS(self, dlc, wind_file_name):
        T = 12
        t = np.linspace(0., T, num=int(T/self.dt+1))

        # constants from standard
        alpha = 0.2
        Beta = 6.4

        # Flow angle adjustments
        V_hub = dlc.URef*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = dlc.URef*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = dlc.sigma1

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_vert = np.zeros_like(t)+alpha
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        # Transient
        shear_lin = np.zeros_like(t)

        if dlc.direction_pn == 'p':
            k_dir=1.
        elif dlc.direction_pn == 'n':
            k_dir=-1.
        else:
            raise Exception('The EWS gust can only have positive (p) or negative (n) direction, whereas the script receives '+ dlc.direction_pn)

        if dlc.shear_hv == 'v':
            k_v=1.
            k_h=0.
        elif dlc.shear_hv == 'h':
            k_v=0.
            k_h=1.
        else:
            raise Exception('The EWS gust can only have vertical (v) or horizontal (h) shear, whereas the script receives '+ dlc.shear_hv)

        for i, ti in enumerate(t):
            shear_lin[i] = k_dir * (2.5+0.2*Beta*sigma_1*(self.D/self.Lambda_1)**(1/4))*(1-np.cos(2*np.pi*ti/T))/V_hub

        data = np.column_stack((t, V, V_dir, V_vert, k_h * shear_lin, shear_vert, k_v * shear_lin, V_gust, upflow))

        # Header
        hd = '! EWS gust, wind speed = ' + str(dlc.URef) + ', direction ' + dlc.direction_pn + ', shear ' + dlc.shear_hv
        self.write_wnd(wind_file_name, data, hd)

    def Ramp(self, dlc, wind_file_name):
        # Ramp

        T = dlc.total_time - dlc.transient_time
        t = np.linspace(0., dlc.ramp_duration, 2)

        # Contant variables
        V = np.linspace(dlc.wind_speed,dlc.wind_speed+dlc.ramp_speeddelta,2)
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        
        data = np.column_stack((t, V, V_dir, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
        # Header
        hd = f'! Ramp wind starting from wind speed = {dlc.URef} to wind speed = {dlc.wind_speed+dlc.ramp_speeddelta}\n'
        self.write_wnd(wind_file_name, data, hd)

    def Step(self, dlc, wind_file_name):
        # Step

        T = dlc.total_time - dlc.transient_time
        t = np.linspace(0., self.dt, 2)

        # Contant variables
        V = np.linspace(dlc.wind_speed,dlc.wind_speed+dlc.step_speeddelta,2)
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        
        data = np.column_stack((t, V, V_dir, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
        # Header
        hd = f'! Step wind starting from wind speed = {dlc.URef} to wind speed = {dlc.wind_speed+dlc.step_speeddelta}\n'
        self.write_wnd(wind_file_name, data, hd)

    def write_wnd(self, fname, data, hd):

        # Move transient event to user defined time
        data[:,0] += self.TStart
        data = np.vstack((data[0,:], data, data[-1,:]))
        data[0,0] = self.T0
        data[-1,0] = self.TF

        # Headers
        hd1 = ['Time', 'Wind', 'Wind', 'Vertical', 'Horiz.', 'Pwr. Law', 'Lin. Vert.', 'Gust', 'Upflow']
        hd2 = ['',     'Speed', 'Dir', 'Speed',    'Shear',  'Vert. Shr', 'Shear',     'Speed', 'Angle']
        hd3 = ['(s)', '(m/s)', '(deg)', '(m/s)', '(-)', '(-)', '(-)', '(m/s)', '(deg)']

        fid = open(fname, 'w')

        fid.write('! Wind file generated by WEIS - IEC 61400-1 3rd Edition\n')
        for ln in hd:
            fid.write(ln)
        fid.write('! ---------------------------------------------------------------\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd1]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd2]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd3]) + '\n')
        for row in data:
            fid.write('  '+''.join([('%.6f' % val).center(12) for val in row]) + '\n')

        fid.close()

    def write_bts(self, bts_fname, wnd_fname, new_fname = None):

        data = np.loadtxt(wnd_fname, skiprows=6)

        ## TODO: more advanced function to convert wnd to bts
        ts = self.wnd2bts(data, 0, self.HH)
        if new_fname: self.add_turbulence(bts_fname, ts['u'], ts['v'], ts['w'], data[:,0], self.HH, new_fname)
        else: return self.add_turbulence(bts_fname, ts['u'], ts['v'], ts['w'], data[:,0], self.HH, new_fname)

    def wnd2bts(self, data, y, z):
        """
        Converts wnd-style data to bts-style data
        0: Time
        1: Wind speed
        2: Wind dir
        3: Vertical speed
        4: Horizontal shear
        5: Pwr law vert shear
        6: Linear vertical shear
        7: Gust speed
        8: Upflow angle
        """
        ts = {}
        u = np.cos(data[:,2]*np.pi/180) * np.cos(data[:,8]*np.pi/180) * data[:,1] + \
            np.sin(data[:,8]*np.pi/180) * data[:,3]
        v = -np.sin(data[:,2]*np.pi/180)
        w = -np.cos(data[:,2]*np.pi/180) * np.sin(data[:,8]*np.pi/180) * data[:,1] + \
            np.cos(data[:,8]*np.pi/180) * data[:,3]
        ts['u'] = u
        ts['v'] = v
        ts['w'] = w

        return ts

    def add_turbulence(fname, u, v = None, w = None, time = None, HH = None, new_fname = None):
        """
        Creates a new BTS file using the turbulence of an existing BTS file,
        combined with time-varying u (and possibly v and w) signals.
        Superimposed the turbulence of BTS file filename on top of the provided signals.
        Velocity signals can be 1D arrays (time), or 3D arrays that match the dimensions 
        of the turbsim file (Nt, Ny, Nz).
        
        Inputs:
            fname (str):            path to BTS file to grab turbulence from.
            u (array):              time-varying velocity in x-direction.
            v (array, optional):    time-varying velocity in x-direction (assumed 0).
            u (array, optional):    time-varying velocity in x-direction (assumed 0).
            time (array, optional): time array, must be same size as u (and v and w).
                                        If undefined, velocity inputs must match BTS file size.
            HH (float, opt):        user-defined height to take average WS at.
                                        Assumed to be middle of grid.
            new_fname (str, opt):   filename for the new BTS file. If undefined, the function
                                        returns the TurbSimFile object instead.
        """

        if v is None: v = np.zeros_like(u)
        if w is None: w = np.zeros_like(u)

        ## Load BTS file
        ts = turbsim_file.TurbSimFile(fname)
        ts_shape = ts['u'].shape
        if HH: 
            ## Find z location closest to reference height
            ref_idx = np.argmin(abs(ts['z']-HH))
            ## If ref_height not defined, assume it to be in the middle
        elif ts_shape[3] % 2 == 1: 
            ref_idx = int(ts_shape[3]/2-0.5)
        else: 
            ref_idx = [int(ts_shape[3]/2-1), int(ts_shape[3]/2)]

        ## Interpolate velocity signal if size is different from BTS file
        if ts_shape[1] != np.shape(u):
            u = np.interp(ts['t'], time, u)
            v = np.interp(ts['t'], time, v)
            w = np.interp(ts['t'], time, w)

        ## Calculate mean in x-direction at reference height
        ## Note: it is assumed that  v and w are always mean 0
        tsu_mean = np.mean(ts['u'][0,:,:,ref_idx])

        ## Scaling factor for scaling turbulence
        scaling = u / tsu_mean

        ## Copy old turbsimfile object into new turbsimfile object
        ts_new = ts
        
        ## Change velocity profile as defined
        ts_new['u'][0,:,:,:] = (ts['u'][0,:,:,:] - tsu_mean) + \
                    np.tile(u[:, np.newaxis, np.newaxis],(1, ts_shape[2], ts_shape[3]))
        ts_new['u'][1,:,:,:] = ts['u'][1,:,:,:] + \
                    np.tile(v[:, np.newaxis, np.newaxis],(1, ts_shape[2], ts_shape[3]))
        ts_new['u'][2,:,:,:] = ts['u'][2,:,:,:] + \
                    np.tile(w[:, np.newaxis, np.newaxis],(1, ts_shape[2], ts_shape[3]))

        if new_fname: 
            ts_new.write(new_fname)
        else: 
            return ts_new

    
