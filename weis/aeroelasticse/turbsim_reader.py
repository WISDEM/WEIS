class TurbsimReader(object):

    def read_input_file(self, input_file_name):
        inpf = open(input_file_name, 'r')

        # Runtime Options
        inpf.readline()
        inpf.readline()
        inpf.readline()
        self.Echo = inpf.readline().split()[0]
        self.RandSeed1 = inpf.readline().split()[0]
        self.RandSeed2 = inpf.readline().split()[0]
        self.WrBHHTP = inpf.readline().split()[0]
        self.WrFHHTP = inpf.readline().split()[0]
        self.WrADHH = inpf.readline().split()[0]
        self.WrADFF = inpf.readline().split()[0]
        self.WrBLFF = inpf.readline().split()[0]
        self.WrADTWR = inpf.readline().split()[0]
        self.WrFMTFF = inpf.readline().split()[0]
        self.WrACT = inpf.readline().split()[0]
        self.Clockwise = inpf.readline().split()[0]
        self.ScaleIEC = inpf.readline().split()[0]

        # Turbine/Model Specifications
        inpf.readline()
        inpf.readline()
        self.NumGrid_Z = inpf.readline().split()[0]
        self.NumGrid_Y = inpf.readline().split()[0]
        self.TimeStep = inpf.readline().split()[0]
        self.AnalysisTime = inpf.readline().split()[0]
        self.UsableTime = inpf.readline().split()[0]
        self.HubHt = inpf.readline().split()[0]
        self.GridHeight = inpf.readline().split()[0]
        self.GridWidth = inpf.readline().split()[0]
        self.VFlowAng = inpf.readline().split()[0]
        self.HFlowAng = inpf.readline().split()[0]

        # Meteorological Boundary Conditions 
        inpf.readline()
        inpf.readline()
        self.TurbModel = inpf.readline().split()[0]
        self.UserFile = inpf.readline().split()[0].replace("'","").replace('"','')
        self.IECstandard = inpf.readline().split()[0]
        self.IECturbc = inpf.readline().split()[0]
        self.IEC_WindType = inpf.readline().split()[0]
        self.ETMc = inpf.readline().split()[0]
        self.WindProfileType = inpf.readline().split()[0]
        self.ProfileFile = inpf.readline().split()[0].replace("'","").replace('"','')
        self.RefHt = inpf.readline().split()[0]
        self.URef = inpf.readline().split()[0]
        self.ZJetMax = inpf.readline().split()[0]
        self.PLExp = inpf.readline().split()[0]
        self.Z0 = inpf.readline().split()[0]


        # Meteorological Boundary Conditions 
        inpf.readline()
        inpf.readline()
        self.Latitude = inpf.readline().split()[0]
        self.RICH_NO = inpf.readline().split()[0]
        self.UStar = inpf.readline().split()[0]
        self.ZI = inpf.readline().split()[0]
        self.PC_UW = inpf.readline().split()[0]
        self.PC_UV = inpf.readline().split()[0]
        self.PC_VW = inpf.readline().split()[0]

        # Spatial Coherence Parameters
        inpf.readline()
        inpf.readline()
        self.SCMod1 = inpf.readline().split()[0]
        self.SCMod2 = inpf.readline().split()[0]
        self.SCMod3 = inpf.readline().split()[0]
        self.InCDec1 = inpf.readline().split()[0]
        self.InCDec2 = inpf.readline().split()[0]
        self.InCDec3 = inpf.readline().split()[0]
        self.CohExp = inpf.readline().split()[0]

        # Spatial Coherence Parameters
        inpf.readline()
        inpf.readline()
        self.CTEventPath = inpf.readline().split()[0]
        self.CTEventFile = inpf.readline().split()[0]
        self.Randomize = inpf.readline().split()[0]
        self.DistScl = inpf.readline().split()[0]
        self.CTLy = inpf.readline().split()[0]
        self.CTLz = inpf.readline().split()[0]
        self.CTStartTime = inpf.readline().split()[0]


