from openmdao.api import ExplicitComponent


class IECWind(ExplicitComponent):
    
    def initialize(self):
        pass

    def setup(self):
        pass

        # inputs
        self.add_input(
            "hub_height",
            val=0.0,
            units="m",
            desc="Height of the hub in the global reference system, i.e. distance rotor center to ground.",
        )
        self.add_input(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Diameter of the rotor used in WISDEM. It is defined as two times the blade length plus the hub diameter.",
        )
        self.add_input(
            "ws_class",
            val="",
            desc="IEC wind turbine class. I - offshore, II coastal, III - land-based, IV - low wind speed site.",
        )
        self.add_input(
            "turb_class",
            val="",
            desc="IEC wind turbine category. A - high turbulence intensity (land-based), B - mid turbulence, C - low turbulence (offshore).",
        )

        # outputs
        self.add_output("AEP", val=0.0, units="kW*h", desc="annual energy production")


    def compute(self, inputs, outputs):

        pass