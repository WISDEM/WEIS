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

        if dlc.IEC_WindType.split('-')[-1] == 'EOG':
            self.EOG(dlc, wind_file_name)
        elif dlc.IEC_WindType.split('-')[-1] == 'EDC':
            self.EDC(dlc, wind_file_name)
        elif dlc.IEC_WindType.split('-')[-1] == 'ECD':
            self.ECD(dlc, wind_file_name)
        elif dlc.IEC_WindType.split('-')[-1] == 'EWS':
            self.EWS(dlc, wind_file_name)
        elif dlc.IEC_WindType.split('-')[-1] == 'Ramp':
            self.Ramp(dlc, wind_file_name)
        elif dlc.IEC_WindType.split('-')[-1] == 'Step':
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
            raise Exception('The EDC gust can only have positive (p) or negative (n) direction, whereas the script receives '+ str(dlc.direction_pn))

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
            raise Exception('The ECD gust can only have positive (p) or negative (n) direction, whereas the script receives '+ str(dlc.direction_pn))
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
            raise Exception('The EWS gust can only have positive (p) or negative (n) direction, whereas the script receives '+ str(dlc.direction_pn))

        if dlc.shear_hv == 'v':
            k_v=1.
            k_h=0.
        elif dlc.shear_hv == 'h':
            k_v=0.
            k_h=1.
        else:
            raise Exception('The EWS gust can only have vertical (v) or horizontal (h) shear, whereas the script receives '+ str(dlc.shear_hv))

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

    def read_wnd(self, wnd_file):
        """
        Reads wnd file and returns data array of size (nt, 9).
        """
        return np.genfromtxt(wnd_file, comments="!")

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

    def write_bts(self, wnd_file, bts_file = None, new_fname = None, HH = None, dt = None, y = None, z = None, propagation=True):
        """
        Writes a .bts wind file based on wind data from a .wnd file.
        Inputs:
            wnd_file (path or TurbSimFile obj): Location of the wnd file to be loaded, 
                                                    or TurbSimFile object from wnd2bts().
            bts_file (path or TurbSimFile obj): Location of the wnd file to be loaded, 
                (optional)                          or TurbSimFile object. If defined, it 
                                                    merges both files using add_turbulence().
            new_fname (str, optional):          Name for .bts file to be written. If not defined,
                                                    function returns TurbSimFile object instead.
            HH (float, opt):                    Hub height, default is self.HH.
            dt (float, opt):                    Desired step size of the .bts file.
            y (array-like, opt):                Array containing y-coordinates of the .bts grid.
            z (array-like, opt):                Array containing z-coordinates of the .bts grid.
            propagation (bool):                 Include the propagation of flow in .bts file, default: True. 
                                                    Note that the propagation can only be calculated if y and 
                                                    z are defined.
        Outputs:
            Function writes the bts file if new_fname is defined.
            If not defined, it returns a TurbSimFile object containing the bts data.
        """

        ## Load .bts file and set grid points of the new .bts file
        if isinstance(bts_file, str): 
            bts_data = turbsim_file.TurbSimFile(bts_file)
            y = bts_data['y']
            z = bts_data['z']
        elif bts_file is not None: 
            bts_data = bts_file
            y = bts_data['y']
            z = bts_data['z']
        elif (y is None) or (z is None):
            raise Exception("No grid positions y and z provided through inputs or bts_file.")

        ## Load .wnd file 
        if isinstance(wnd_file, str):
            data = self.read_wnd(wnd_file)
            wnd_data = self.wnd2bts(data, HH, dt, y, z, propagation)
        else:
            wnd_data = wnd_file

        ## Merge .bts and .wnd files
        if bts_file: 
            merged_file = self.add_turbulence(bts_data, wnd_data, HH = HH, new_fname = new_fname)
        else:
            merged_file = wnd_data

        ## Write .bts file or return TurbSimFile object
        if new_fname:
            merged_file.write(new_fname)
        else:
            merged_file['ID'] = 7
            return merged_file

    def wnd2bts(self, data, HH = None, dt = None, y = None, z = None, propagation=True):
        """
        Converts wnd-style data to bts-style data
        Inputs:
            data (array-like or str):   2D array that contains channels from the .wnd file, or path of .wnd file 
                                            to be loaded.
            HH (float, opt):            Hub height, default is self.HH.
            dt (float, opt):            Desired step size of the .bts file. Note that if you want to create a 
                                            functional .bts file, dt is a necessary input unless the .wnd file 
                                            already contains data for a constant step size dt.
            y (array-like, opt):        Array containing y-coordinates of the .bts grid.
            z (array-like, opt):        Array containing z-coordinates of the .bts grid.
            propagation (bool):         Include the propagation of flow in .bts file, default: True. Note that 
                                            the propagation can only be calculated if y and z are defined.

        Channels in data:
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
        ts = turbsim_file.TurbSimFile()

        if isinstance(data, str):
            data = self.read_wnd(data)

        if dt is not None:
            ## Interpolate data over timesteps dt
            data_old = data
            ts['t'] = np.arange(0, data[-1,0]+dt, dt)
            data = np.empty((len(ts['t']), 9))
            data[:,0] = ts['t']
            for n in range(8):
                data[:,n+1] = np.interp(ts['t'], data_old[:,0], data_old[:,n+1])

        if y is not None:
            if z is None: 
                raise Exception("If input y is provided, input z should be provided, too.")
            else:
                refLength = y[-1]-y[0]
                if HH is None: HH = self.HH

                ## Calculate horizontal velocity for all grid points
                U_grid = ( np.tile(data[:,1][:,np.newaxis, np.newaxis], (1, len(y), len(z))) * 
                        # Horizontal linear shear
                        ( np.tile(((data[:,4] * np.outer(y, np.cos(data[:,2]*np.pi/180))).T / refLength) 
                            [:, :, np.newaxis], (1, 1, len(z))) 
                        # Vertical linear shear
                        +  np.tile((np.outer(data[:,6], np.array(z)-HH) / refLength) 
                            [:, np.newaxis, :], (1, len(y), 1)) 
                        # Power-law wind shear
                        + np.tile((((np.tile(np.array(z)[:,np.newaxis], (1, len(data[:,0])))/HH)**data[:,5]).T) 
                            [:, np.newaxis, :], (1, len(y), 1)) ) 
                        # Gust speed
                        + np.tile(data[:,7][:,np.newaxis, np.newaxis], (1, len(y), len(z))) )

                ## Apply upflow angle and wind direction
                W_grid = np.tile(data[:,3][:, np.newaxis, np.newaxis], (1, len(y), len(z)))
                u = ( (np.cos(data[:,8][:,np.newaxis, np.newaxis]*np.pi/180) * U_grid 
                            - np.sin(data[:,8][:, np.newaxis, np.newaxis]*np.pi/180) * W_grid) 
                            * np.cos(data[:,2][:, np.newaxis, np.newaxis]*np.pi/180) )
                v = - ( (np.cos(data[:,8][:, np.newaxis, np.newaxis]*np.pi/180) * U_grid 
                            - np.sin(data[:,8][:, np.newaxis, np.newaxis]*np.pi/180) * W_grid) 
                            * np.sin(data[:,2][:, np.newaxis, np.newaxis]*np.pi/180) )
                w = ( np.sin(data[:,8][:, np.newaxis, np.newaxis]*np.pi/180) * U_grid 
                            + np.cos(data[:,8][:, np.newaxis, np.newaxis]*np.pi/180) * W_grid)

                ts['y'] = y - np.mean(y)
                ts['z'] = np.array(z)
                ts['t'] = data[:,0]

                ## Add elements to the front of the .bts file to propagate through at t=0
                if propagation and (dt is not None):
                    umean = np.mean( u[:, int((len(y)-1)/2), int((len(z)-1)/2)] )
                    vmean = np.mean( v[:, int((len(y)-1)/2), int((len(z)-1)/2)] )
                    if dt is None:
                        delay = (refLength/2) / (np.sqrt(umean**2 + vmean**2))
                        u = np.concatenate((umean*np.ones((1, len(y), len(z))), u))
                        v = np.concatenate((vmean*np.ones((1, len(y), len(z))), v))
                        w = np.concatenate((np.zeros((1, len(y), len(z))), w))
                        ts['t'] += delay
                        ts['t'] = np.insert(ts['t'], 0, 0)
                    else:
                        delay = int((refLength/2) / (np.sqrt(umean**2 + vmean**2)) / dt)
                        u = np.concatenate((umean*np.ones((delay, len(y), len(z))), u))
                        v = np.concatenate((vmean*np.ones((delay, len(y), len(z))), v))
                        w = np.concatenate((np.zeros((delay, len(y), len(z))), w))
                        ts['t'] = np.concatenate((ts['t'], ts['t'][-1]+np.arange(dt, delay*dt+dt, dt)))

        else:
            if data[:,4].any():
                print("WARNING: the .wnd dataset contains horizontal shear, but no input y is provided. "\
                        "Therefore, the generated .bts file will not contain horizontal shear. "\
                        "If you want to use the shear from the .wnd file, please provide y and z.")
            elif data[:, 5:7].any():
                print("WARNING: the .wnd dataset contains vertical shear, but no input z is provided. "\
                            "Therefore, the generated .bts file will not contain vertical shear. "\
                            "If you want to use the shear from the .wnd file, please provide y and z.")
            
            u = ( (np.cos(data[:,8]*np.pi/180) * (data[:,1] + data[:,7]) 
                            - np.sin(data[:,8]*np.pi/180) * data[:,3]) 
                            * np.cos(data[:,2]*np.pi/180) )
            v = - ( (np.cos(data[:,8]*np.pi/180) * (data[:,1] + data[:,7]) 
                            - np.sin(data[:,8]*np.pi/180) * data[:,3]) 
                            * np.sin(data[:,2]*np.pi/180) )
            w = ( np.sin(data[:,8]*np.pi/180) * (data[:,1] + data[:,7])  
                            + np.cos(data[:,8]*np.pi/180) * data[:,3] )

            ts['t'] = data[:,0]
        
        ts['u'] = np.stack((u, v, w))

        return ts

    def add_turbulence(self, fname, u, v = None, w = None, time = None, HH = None, new_fname = None):
        """
        Creates a new BTS file using the turbulence of an existing BTS file,
        combined with time-varying u (and possibly v and w) signals.
        Superimposed the turbulence of BTS file filename on top of the provided signals.
        Velocity signals can be 1D arrays (time), or 3D arrays that match the dimensions 
        of the turbsim file (Nt, Ny, Nz).
        
        Inputs:
            fname (str, TSFile):    path to BTS file to grab turbulence from, or TurbSimFile.
            u (array, dict):        time-varying velocity in x-direction, 
                                        or dict containing 'u', 'v' and 'w' (and optionally, 't').
            v (array, optional):    time-varying velocity in x-direction (assumed 0).
            u (array, optional):    time-varying velocity in x-direction (assumed 0).
            time (array, optional): time array, must be same size as u (and v and w).
                                        If undefined, velocity inputs must match .bts file size.
            HH (float, opt):        user-defined height to take average WS at. Assumed to be middle 
                                        of grid. Only used if u.ndim = 1.
            new_fname (str, opt):   filename for the new BTS file. If undefined, the function
                                        returns the TurbSimFile object instead.
        """

        if not isinstance(u, (tuple, list, np.ndarray)):
            v = u['u'][1,:,:,:]
            w = u['u'][2,:,:,:]
            if time is None:
                try: time = u['t']
                except: pass
            u = u['u'][0,:,:,:]

        if v is None: v = np.zeros_like(u)
        if w is None: w = np.zeros_like(u)

        ## Check if u, v, and w have same shape
        if (np.shape(u) != np.shape(v)) or (np.shape(u) != np.shape(w)):
            raise Exception('The shape of u {} must be the same as the shapes of v {} and w {}.'\
                                .format(np.shape(u), np.shape(v), np.shape(w)))

        ## Load BTS file
        if isinstance(fname, str): ts = turbsim_file.TurbSimFile(fname)
        else: ts = fname

        # Cut bts file to length of wnd file
        dt_ts = ts['t'][-1] - ts['t'][-2]
        if ts['t'][-1] + dt_ts < time[-1]:
            raise Exception("End time of bts file ({}) must be greater or equal to that of wnd file ({})."\
                                .format(ts['t'][-1], time[-1]) )
        elif ts['t'][-1] > time[-1]:
            tend = np.where(ts['t'] >= time[-1])[0][0]+1
            ts['t'] = ts['t'][:tend]
            ts['u'] = ts['u'][:,:tend,:,:]

        ts_shape = ts['u'].shape

        ## Create 3D velocity vectors based on the shape of the provided velocity vectors
        u_ndims = len(np.shape(u))

        ## If u has one dimension, it must be the time dimension
        if u_ndims == 1:
            if np.shape(u) == np.shape(time):
                # Interpolate provided time vector to bts time indices
                u = np.interp(ts['t'], time, u)
                v = np.interp(ts['t'], time, v)
                w = np.interp(ts['t'], time, w)

            ## TODO: We could possibly add the functionality to provide constant shear here
            ## (without providing veer/horizontal shear), so a (tN x zN) sized u

            elif np.shape(u) != ts_shape[1]:
                # Otherwise must match bts time series length
                raise Exception(\
                    'The shape of u {} must be the same as either time {} or the .bts time {}.'\
                        .format(np.shape(u), np.shape(time), ts_shape[1]))
            
            # Match shape
            u = np.tile(u[:, np.newaxis, np.newaxis], (1, ts_shape[2], ts_shape[3]))
            v = np.tile(v[:, np.newaxis, np.newaxis], (1, ts_shape[2], ts_shape[3]))
            w = np.tile(w[:, np.newaxis, np.newaxis], (1, ts_shape[2], ts_shape[3]))

            ## Determine reference height
            if HH: 
                ## Find z location closest to reference height
                ref_idx = np.argmin(abs(ts['z']-HH))
                ## If ref_height not defined, assume it to be in the middle
            elif ts_shape[3] % 2 == 1: 
                ref_idx = int(ts_shape[3]/2-0.5)
            else: 
                ref_idx = [int(ts_shape[3]/2-1), int(ts_shape[3]/2)]
        
        ## If u has two dimensions, it must be the same size as the bts grid
        elif u_ndims == 2:
            if np.shape(u) != ts_shape[2:]:
                raise Exception(\
                    'If a 2D array, the shape of u {} must be the same as the bts grid size {}.'\
                        .format(np.shape(u), ts_shape[2:]))
            ## TODO: We could possibly add the functionality to provide shear over time here
            ## (without providing veer/horizontal shear)

            else:
                # Match shape
                u = np.tile(u[np.newaxis, :, :], (ts_shape[1], 1, 1))
                v = np.tile(v[np.newaxis, :, :], (ts_shape[1], 1, 1))
                w = np.tile(w[np.newaxis, :, :], (ts_shape[1], 1, 1))
        
        ## If u has three dimensions, it must be (time, y, z)
        elif u_ndims == 3:
            if np.shape(u)[0] == np.size(time):
                # Simple function to interpolate the grid over time
                # TODO: replace for something more efficient
                def interp_grid(ts, time, u, v, w):
                    u_interp = np.empty(ts_shape[1:])
                    v_interp = np.empty(ts_shape[1:])
                    w_interp = np.empty(ts_shape[1:])
                    for n in range(np.shape(u)[1]):
                        for m in range(np.shape(u)[2]):
                            u_interp[:,n,m] = np.interp(ts['t'], time, u[:,n,m])
                            v_interp[:,n,m] = np.interp(ts['t'], time, v[:,n,m])
                            w_interp[:,n,m] = np.interp(ts['t'], time, w[:,n,m])
                    return u_interp, v_interp, w_interp
                
                u, v, w = interp_grid(ts, time, u, v, w)
            
            elif np.shape(u) != ts_shape[1:]:
                raise Exception(\
                'If a 3D array, the shape of u {} must be the same as the bts u vector {}, or the first dimension must match time {}.'\
                    .format(np.shape(u), ts_shape[1:], np.shape(time)))

        ## Calculate mean wind speed at each time instance
        ## Note: in this case, all veer/shear is removed from the bts file,
        ## as it is presumed to be provided in the u/v/w vectors
        U_abs = np.sqrt(u**2 + v**2)[np.newaxis,:,:,:]
        Umean_wnd = np.average(np.average(U_abs, axis=2, keepdims=True), axis=3, keepdims=True)

        ## Calculate mean WS of .bts file and subsequent scaling factor
        Umean_bts = np.mean(np.sqrt(ts['u'][0,:,:,:]**2 + ts['u'][1,:,:,:]**2))
        Umean_grid = np.mean(ts['u'], axis=1, keepdims=True)
        if Umean_bts == 0: 
            scaling = 1
            print("WARNING: mean wind speed of bts file is zero. No scaling applied.")
        else: 
            scaling = Umean_wnd / Umean_bts

        ## Copy old turbsimfile object into new turbsimfile object
        ts_new = ts
        
        ## Change velocity profile as defined
        ## Note that v and w are assumed zero-mean
        ts_new['u'] = ((ts_new['u'] - Umean_grid) * scaling) + np.stack((u, v, w))

        return ts_new
