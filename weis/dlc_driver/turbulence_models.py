class IEC_TurbulenceModels():

    def __init__(self):

        self.Turbine_Class    = 'I'      # IEC Wind Turbine Class
        self.Turbulence_Class = 'B'    # IEC Turbulance Class
        self.z_hub            = 90.    # wind turbine hub height (m)

    def setup(self):
        # General turbulence parameters: 6.3
        # Sigma_1: logitudinal turbulence scale parameter

        # Setup
        if self.Turbine_Class == 'I':
            self.V_ref = 50.
            self.Turbine_Class_Num = '1'
        elif self.Turbine_Class == 'II':
            self.V_ref = 42.5
            self.Turbine_Class_Num = '2'
        elif self.Turbine_Class == 'III':
            self.V_ref = 37.5
            self.Turbine_Class_Num = '3'
        elif self.Turbine_Class == 'IV':
            self.V_ref = 30.
            self.Turbine_Class_Num = '4'
        else:
            raise Exception('The wind turbine class is not defined properly')
        self.V_ave = self.V_ref*0.2

        if self.Turbulence_Class == 'A+':
            self.I_ref = 0.18
        elif self.Turbulence_Class == 'A':
            self.I_ref = 0.16
        elif self.Turbulence_Class == 'B':
            self.I_ref = 0.14
        elif self.Turbulence_Class == 'C':
            self.I_ref = 0.12

        if self.z_hub > 60:
            self.Sigma_1 = 42
        else:
            self.Sigma_1 = 0.7*self.z_hub
            
    def NTM(self, V_hub):
        # Normal turbulence model: 6.3.1.3
        b = 5.6
        sigma_1 = self.I_ref*(0.75*V_hub + b)
        return sigma_1

    def ETM(self, V_hub):
        # Extreme turbulence model: 6.3.2.3
        c = 2
        sigma_1 = c*self.I_ref*(0.072*(self.V_ave/c + 3)*(V_hub/c - 4) + 10)
        return sigma_1

    def EWM(self, V_hub):
        # Extreme wind speed model: 6.3.2.1
                
        # Steady
        V_e50 = 1.4*self.V_ref
        V_e1 = 0.8*V_e50
        # Turb
        V_50 = self.V_ref
        V_1 = 0.8*V_50
        sigma_1 = 0.11*V_hub

        return sigma_1, V_e50, V_e1, V_50, V_1