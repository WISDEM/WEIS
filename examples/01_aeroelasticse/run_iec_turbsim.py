from weis.aeroelasticse.pyIECWind import pyIECWind_extreme, pyIECWind_turb
import os


def example_ExtremeWind():

    iec = pyIECWind_extreme()
    iec.Turbine_Class = 'I'     # IEC Wind Turbine Class
    iec.Turbulence_Class = 'A'  # IEC Turbulance Class
    iec.dt = 0.05               # Transient wind time step (s)
    iec.dir_change = 'both'     # '+','-','both': sign for transient events in EDC, EWS
    iec.z_hub = 30.             # wind turbine hub height (m)
    iec.D = 42.                 # rotor diameter (m)

    iec.outdir = 'wind'
    iec.case_name = 'extreme'

    V_hub = 25
    iec.execute('EWS', V_hub)

def example_TurbulentWind():
    iec = pyIECWind_turb()
    
    iec.Turbulence_Class = 'A'  # IEC Turbulance Class
    iec.z_hub = 90.             # wind turbine hub height (m)
    iec.D = 126.                 # rotor diameter (m)
    iec.AnalysisTime = 30.

    iec.outdir = 'wind'
    iec.case_name = 'turb'
    run_dir         = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
    iec.Turbsim_exe = os.path.join(run_dir, 'local/bin/turbsim')
    iec.debug_level = 1

    IEC_WindType = 'NTM'
    Uref = 10.

    iec.execute(IEC_WindType, Uref)


if __name__=="__main__":

    example_ExtremeWind()
    example_TurbulentWind()