import numpy as np
import os

class IEC_CoherentGusts():

    def __init__(self):

        self.Turbine_Class    = 'I'    # IEC Wind Turbine Class
        self.Turbulence_Class = 'B'    # IEC Turbulance Class
        self.Vert_Slope       = 0      # Vertical slope of the wind inflow (deg)
        self.TStart           = 30     # Time to start transient conditions (s)
        self.dt               = 0.05   # Transient wind time step (s)
        self.dir_change       = 'both' # '+','-','both': sign for transient events in EDC, EWS
        self.shear_orient     = 'both' # 'v','h','both': vertical or horizontal shear for EWS
        self.z_hub            = 90.    # wind turbine hub height (m)
        self.D                = 126.   # rotor diameter (m)
        
        self.T0               = 0.
        self.TF               = 630.

    def EOG(self, V_hub_in):
        # Extreme operating guest: 6.3.2.2

        self.setup()

        T = 10.5
        t = np.linspace(0., T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)
        __, __, V_e1, __, __ = self.EWM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        V_gust = min([ 1.35*(V_e1 - V_hub), 3.3*(sigma_1/(1+0.1*(self.D/self.Sigma_1))) ])

        V_gust_t = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V_gust_t[i] = 0. - 0.37*V_gust*np.sin(3*np.pi*ti/T)*(1-np.cos(2*np.pi*ti/T))
            else:
                V_gust_t[i] = 0.

        # Write Files
        self.fname_out = []
        self.fname_type = []
        fname = self.case_name + '_EOG_U%2.1f.wnd'%V_hub_in
        data = np.column_stack((t, V, V_dir, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust_t, upflow))
        hd = []
        hd.append('! Extreme operating guest\n')
        hd = self.heading_common(hd)
        hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
        self.write_wnd(fname, data, hd)
        self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
        self.fname_type.append(2)

    def EDC(self, V_hub_in):
        # Extreme direction change: 6.3.2.4

        self.setup()
        
        T = 6.
        t = np.linspace(0., T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        # Transient
        Theta_e = 4.*np.arctan(sigma_1/(V_hub*(1.+0.01*(self.D/self.Sigma_1))))*180./np.pi
        if Theta_e > 180.:
            Theta_e = 180.

        Theta_p = np.zeros_like(t)
        Theta_n = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                Theta_p[i] = 0.5*Theta_e*(1-np.cos(np.pi*ti/T))
                Theta_n[i] = -1*0.5*Theta_e*(1-np.cos(np.pi*ti/T))
            else:
                Theta_p[i] = Theta_e
                Theta_n[i] = -1*Theta_e

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            ## Vert
            fname = self.case_name + '_EDC_P_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_p, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
            hd = []
            hd.append('! Exteme Vertical Wind Shear, positive\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)


        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            ## Vert
            fname = self.case_name + '_EDC_N_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_n, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
            hd = []
            hd.append('! Exteme Vertical Wind Shear, negative\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)
    
    def ECD(self, V_hub_in):
        # Extreme coherent gust with direction change: 6.3.2.5

        self.setup()
        
        T = 10.
        t = np.linspace(0., T, num=int((T/self.dt+1)))

        # constants from standard
        alpha = 0.2
        V_cg = 15 #m/s

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

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

        Theta_p = np.zeros_like(t)
        Theta_n = np.zeros_like(t)
        V = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V[i] = V_hub + 0.5*V_cg*(1-np.cos(np.pi*ti/T))
                Theta_p[i] = 0.5*Theta_cg*(1-np.cos(np.pi*ti/T))
                Theta_n[i] = -1*0.5*Theta_cg*(1-np.cos(np.pi*ti/T))
            else:
                V[i] = V_hub+V_cg
                Theta_p[i] = Theta_cg
                Theta_n[i] = -1*Theta_cg

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            ## Vert
            fname = self.case_name + '_ECD_P_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_p, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
            hd = []
            hd.append('! Exteme coherent gust with direction change, positive\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)


        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            ## Vert
            fname = self.case_name + '_ECD_N_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_n, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust, upflow))
            hd = []
            hd.append('! Exteme coherent gust with direction change, negative\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)

    def EWS(self, V_hub_in):
        # Extreme wind shear: 6.3.2.6
        self.setup()

        T = 12
        t = np.linspace(0, T, num=(int(T/self.dt+1)))

        # constants from standard
        alpha = 0.2
        Beta = 6.4

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        # shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        V_gust = np.zeros_like(t)
        upflow = np.zeros_like(t)

        # Transient
        shear_lin_p = np.zeros_like(t)
        shear_lin_n = np.zeros_like(t)

        for i, ti in enumerate(t):
            shear_lin_p[i] = (2.5+0.2*Beta*sigma_1*(self.D/self.Sigma_1)**(1/4))*(1-np.cos(2*np.pi*ti/T))/V_hub
            shear_lin_n[i] = -1*(2.5+0.2*Beta*sigma_1*(self.D/self.Sigma_1)**(1/4))*(1-np.cos(2*np.pi*ti/T))/V_hub

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'v':
                ## Vert
                fname = self.case_name + '_EWS_V_P_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, np.zeros_like(t), shear_vert, shear_lin_p, V_gust, upflow))
                hd = []
                hd.append('! Exteme Vertical Wind Shear, positive\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'h':
                # Horz
                fname = self.case_name + '_EWS_H_P_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, shear_lin_p, shear_vert, np.zeros_like(t), V_gust, upflow))
                hd = []
                hd.append('! Exteme Horizontal Wind Shear, positive\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'v':
                ## Vert
                fname = self.case_name + '_EWS_V_N_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, np.zeros_like(t), shear_vert, shear_lin_n, V_gust, upflow))
                hd = []
                hd.append('! Exteme Vertical Wind Shear, negative\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'h':
                # Horz
                fname = self.case_name + '_EWS_H_N_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, shear_lin_n, shear_vert, np.zeros_like(t), V_gust, upflow))
                hd = []
                hd.append('! Exteme Horizontal Wind Shear, negative\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

    def heading_common(self, hd):
        hd.append('! IEC Turbine Class %s, IEC Turbulence Category %s\n'%(self.Turbine_Class, self.Turbulence_Class))
        hd.append('! '+('%f'%(self.D)).ljust(14) + ' ' + 'Rotor_Diameter'.ljust(14) + ' - rotor diameter (m)\n')
        hd.append('! '+('%f'%(self.z_hub)).ljust(14) + ' ' + 'hub_height'.ljust(14) + ' - hub height of the wind turbine (m)\n')
        hd.append('! '+('%f'%(self.Sigma_1)).ljust(14) + ' ' + 'Sigma_1'.ljust(14) + ' - logitudinal turbulence scale parameter (m)\n')
        return hd
    
    def heading_variable(self, hd, V_hub_in=[], sigma_1=[]):
        if sigma_1:
            hd.append('! '+('%f'%(sigma_1)).ljust(14) + ' ' + 'sigma_1'.ljust(14) + ' - turbulence standard deviation\n')
        if V_hub_in:
            hd.append('! '+('%f'%(V_hub_in)).ljust(14) + ' ' + 'V_hub'.ljust(14) + ' - wind speed at hub height (m/s)\n')

        return hd

    def write_wnd(self, fname, data, hd):

        # Make sure directory exist
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        # Move transient event to user defined time
        data[:,0] += self.TStart
        data = np.vstack((data[0,:], data, data[-1,:]))
        data[0,0] = self.T0
        data[-1,0] = self.TF

        # Headers
        hd1 = ['Time', 'Wind', 'Wind', 'Vertical', 'Horiz.', 'Pwr. Law', 'Lin. Vert.', 'Gust', 'Upflow']
        hd2 = ['', 'Speed', 'Dir', 'Speed', 'Shear', 'Vert. Shr', 'Shear', 'Speed', 'Angle']
        hd3 = ['(s)', '(m/s)', '(deg)', '(m/s)', '(-)', '(-)', '(-)', '(m/s)', '(deg)']

        self.fpath = os.path.join(self.outdir, fname)
        fid = open(self.fpath, 'w')

        fid.write('! Wind file generated by pyIECWind - IEC 61400-1 3rd Edition\n')
        for ln in hd:
            fid.write(ln)
        fid.write('! ---------------------------------------------------------------\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd1]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd2]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd3]) + '\n')
        for row in data:
            fid.write('  '+''.join([('%.6f' % val).center(12) for val in row]) + '\n')

        fid.close()

    def execute(self, Vtype, V_hub):

        if 'EOG' in Vtype:
            self.EOG(V_hub)
        if 'EDC' in Vtype:
            self.EDC(V_hub)
        if 'ECD' in Vtype:
            self.ECD(V_hub)
        if 'EWS' in Vtype:
            self.EWS(V_hub)

        return self.fname_out, self.fname_type
        