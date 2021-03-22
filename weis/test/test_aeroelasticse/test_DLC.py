"""
Test the DLCs in OpenFAST through aeroelasticSE
"""

import os
import platform
import unittest

import numpy as np

from weis.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from weis.aeroelasticse.FAST_post import FAST_IO_timeseries
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.test.utils import compare_regression_values

this_file_dir = os.path.dirname(os.path.realpath(__file__))


class TestDLC(unittest.TestCase):
    def test_run(self):
        # Turbine inputs
        iec = CaseGen_IEC()
        iec.Turbine_Class = "I"  # Wind class I, II, III, IV
        iec.Turbulence_Class = "B"  # Turbulence class 'A', 'B', or 'C'
        iec.D = 240.0  # Rotor diameter to size the wind grid
        iec.z_hub = 150.0  # Hub height to size the wind grid
        cut_in = 3.0  # Cut in wind speed
        cut_out = 25.0  # Cut out wind speed
        n_ws = 2  # Number of wind speed bins
        TMax = 0.05  # Length of wind grids and OpenFAST simulations, suggested 720 s
        Vrated = 10.59  # Rated wind speed
        Ttrans = max(
            [0.0, TMax - 60.0]
        )  # Start of the transient for DLC with a transient, e.g. DLC 1.4
        TStart = max(
            [0.0, TMax - 600.0]
        )  # Start of the recording of the channels of OpenFAST

        # Initial conditions to start the OpenFAST runs
        u_ref = np.arange(8.0, 10.0)  # Wind speed
        pitch_ref = [0.0, 0.5]  # Pitch values in deg
        omega_ref = [5.5, 6.0]  # Rotor speeds in rpm
        iec.init_cond = {}
        iec.init_cond[("ElastoDyn", "RotSpeed")] = {"U": u_ref}
        iec.init_cond[("ElastoDyn", "RotSpeed")]["val"] = omega_ref
        iec.init_cond[("ElastoDyn", "BlPitch1")] = {"U": u_ref}
        iec.init_cond[("ElastoDyn", "BlPitch1")]["val"] = pitch_ref
        iec.init_cond[("ElastoDyn", "BlPitch2")] = iec.init_cond[
            ("ElastoDyn", "BlPitch1")
        ]
        iec.init_cond[("ElastoDyn", "BlPitch3")] = iec.init_cond[
            ("ElastoDyn", "BlPitch1")
        ]
        iec.init_cond[("HydroDyn", "WaveHs")] = {"U": [3, 4]}
        iec.init_cond[("HydroDyn", "WaveHs")]["val"] = [
            1.101917033,
            1.101917033,
        ]
        iec.init_cond[("HydroDyn", "WaveTp")] = {"U": [3, 4]}
        iec.init_cond[("HydroDyn", "WaveTp")]["val"] = [
            8.515382435,
            8.515382435,
        ]
        iec.init_cond[("HydroDyn", "PtfmSurge")] = {"U": [3.0, 15.0, 25.0]}
        iec.init_cond[("HydroDyn", "PtfmSurge")]["val"] = [4.0, 15.0, 10.0]
        iec.init_cond[("HydroDyn", "PtfmPitch")] = {"U": [3.0, 15.0, 25.0]}
        iec.init_cond[("HydroDyn", "PtfmPitch")]["val"] = [-1.0, 3.0, 1.3]
        iec.init_cond[("HydroDyn", "PtfmHeave")] = {"U": [3.0, 25.0]}
        iec.init_cond[("HydroDyn", "PtfmHeave")]["val"] = [0.5, 0.5]

        # DLC inputs
        wind_speeds = np.linspace(int(cut_in), int(cut_out), int(n_ws))
        iec.dlc_inputs = {}
        iec.dlc_inputs["DLC"] = [1.1]  # , 1.3, 1.4, 1.5, 5.1, 6.1, 6.3]
        iec.dlc_inputs["U"] = [
            wind_speeds,
            # wind_speeds,
            # [Vrated - 2.0, Vrated, Vrated + 2.0],
            # wind_speeds,
            # [Vrated - 2.0, Vrated, Vrated + 2.0, cut_out],
            # [],
            # [],
        ]
        iec.dlc_inputs["Seeds"] = [[1], [1], [], [], [1], [1], [1]]
        # iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
        iec.dlc_inputs["Yaw"] = [[], [], [], [], [], [], []]
        iec.PC_MaxRat = 2.0

        iec.TStart = Ttrans
        iec.TMax = TMax  # wind file length
        iec.transient_dir_change = (
            "both"  # '+','-','both': sign for transient events in EDC, EWS
        )
        iec.transient_shear_orientation = (
            "both"  # 'v','h','both': vertical or horizontal shear for EWS
        )

        # Naming, file management, etc
        iec.wind_dir = "outputs/wind"
        iec.case_name_base = "iea15mw"
        iec.cores = 1

        iec.debug_level = 2
        iec.parallel_windfile_gen = False
        iec.mpi_run = False
        iec.run_dir = "outputs/iea15mw"

        # Run case generator / wind file writing
        case_inputs = {}
        case_inputs[("Fst", "TMax")] = {"vals": [TMax], "group": 0}
        case_inputs[("Fst", "TStart")] = {"vals": [TStart], "group": 0}
        case_inputs[("Fst", "DT")] = {"vals": [0.005], "group": 0}
        case_inputs[("Fst", "DT_Out")] = {"vals": [0.01], "group": 0}  # 0.005
        case_inputs[("Fst", "OutFileFmt")] = {"vals": [2], "group": 0}
        case_inputs[("Fst", "CompHydro")] = {"vals": [1], "group": 0}
        case_inputs[("Fst", "CompSub")] = {"vals": [0], "group": 0}
        case_inputs[("InflowWind", "WindType")] = {"vals": [1], "group": 0}
        case_inputs[("ElastoDyn", "TwFADOF1")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "TwFADOF2")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "TwSSDOF1")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "TwSSDOF2")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "FlapDOF1")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "FlapDOF2")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "EdgeDOF")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "DrTrDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "GenDOF")] = {"vals": ["True"], "group": 0}
        case_inputs[("ElastoDyn", "YawDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmSgDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmSwDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmHvDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmRDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmPDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ElastoDyn", "PtfmYDOF")] = {"vals": ["False"], "group": 0}
        case_inputs[("ServoDyn", "PCMode")] = {"vals": [5], "group": 0}
        case_inputs[("ServoDyn", "VSContrl")] = {"vals": [5], "group": 0}
        run_dir1 = (
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                )
            )
            + os.sep
        )
        if platform.system() == "Windows":
            path2dll = os.path.join(run_dir1, "local/lib/libdiscon.dll")
        elif platform.system() == "Darwin":
            path2dll = os.path.join(run_dir1, "local/lib/libdiscon.dylib")
        else:
            path2dll = os.path.join(run_dir1, "local/lib/libdiscon.so")

        case_inputs[("ServoDyn", "DLL_FileName")] = {"vals": [path2dll], "group": 0}
        case_inputs[("AeroDyn15", "TwrAero")] = {"vals": ["True"], "group": 0}
        case_inputs[("AeroDyn15", "TwrPotent")] = {"vals": [1], "group": 0}
        case_inputs[("AeroDyn15", "TwrShadow")] = {"vals": ["True"], "group": 0}
        case_inputs[("Fst", "CompHydro")] = {"vals": [1], "group": 0}
        case_inputs[("HydroDyn", "WaveMod")] = {"vals": [2], "group": 0}
        case_inputs[("HydroDyn", "WvDiffQTF")] = {"vals": ["False"], "group": 0}
        channels = {}
        for var in [
            "TipDxc1",
            "TipDyc1",
            "TipDzc1",
            "TipDxb1",
            "TipDyb1",
            "TipDxc2",
            "TipDyc2",
            "TipDzc2",
            "TipDxb2",
            "TipDyb2",
            "TipDxc3",
            "TipDyc3",
            "TipDzc3",
            "TipDxb3",
            "TipDyb3",
            "RootMxc1",
            "RootMyc1",
            "RootMzc1",
            "RootMxb1",
            "RootMyb1",
            "RootMxc2",
            "RootMyc2",
            "RootMzc2",
            "RootMxb2",
            "RootMyb2",
            "RootMxc3",
            "RootMyc3",
            "RootMzc3",
            "RootMxb3",
            "RootMyb3",
            "TwrBsMxt",
            "TwrBsMyt",
            "TwrBsMzt",
            "GenPwr",
            "GenTq",
            "RotThrust",
            "RtAeroCp",
            "RtAeroCt",
            "RotSpeed",
            "BldPitch1",
            "BldPitch2",
            "BldPitch3",
            "TTDspSS",
            "TTDspFA",
            "NacYaw",
            # "Wind1VelX",
            # "Wind1VelY",
            # "Wind1VelZ",
            "LSSTipMxa",
            "LSSTipMya",
            "LSSTipMza",
            "LSSTipMxs",
            "LSSTipMys",
            "LSSTipMzs",
            "LSShftFys",
            "LSShftFzs",
            "TipRDxr",
            "TipRDyr",
            "TipRDzr",
        ]:
            channels[var] = True

        case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)

        # for var in var_out+[var_x]:

        # Run FAST cases
        fastBatch = runFAST_pywrapper_batch(
            FAST_ver="OpenFAST", dev_branch=True, post=FAST_IO_timeseries
        )

        # Monopile
        fastBatch.FAST_InputFile = (
            "IEA-15-240-RWT-Monopile.fst"  # FAST input file (ext=.fst)
        )
        run_dir2 = (
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                )
            )
            + os.sep
            + "examples"
            + os.sep
            + "01_aeroelasticse"
            + os.sep
        )
        fastBatch.FAST_directory = os.path.join(
            run_dir2, "OpenFAST_models", "IEA-15-240-RWT", "IEA-15-240-RWT-Monopile"
        )  # Path to fst directory files
        fastBatch.channels = channels
        fastBatch.FAST_runDirectory = iec.run_dir
        fastBatch.case_list = case_list
        fastBatch.case_name_list = case_name_list
        fastBatch.debug_level = 2
        fastBatch.keep_time = True

        _,_,_,out = fastBatch.run_serial()

        train = False
        keys_to_skip = []
        #keys_to_skip = [
        #    "Wind1VelX",
        #    "Wind1VelY",
        #    "Wind1VelZ",
        #    "Wave1Elev",
        #]

        compare_regression_values(
            out,
            "DLC_regression_values_1.pkl",
            directory=this_file_dir,
            tol=1e-1,
            train=train,
            keys_to_skip=keys_to_skip,
        )

        # U-Maine semi-sub
        fastBatch.FAST_InputFile = (
            "IEA-15-240-RWT-UMaineSemi.fst"  # FAST input file (ext=.fst)
        )
        fastBatch.FAST_directory = os.path.join(
            run_dir2, "OpenFAST_models", "IEA-15-240-RWT", "IEA-15-240-RWT-UMaineSemi"
        )  # Path to fst directory files

        _,_,_,out = fastBatch.run_serial()

        compare_regression_values(
            out,
            "DLC_regression_values_2.pkl",
            directory=this_file_dir,
            tol=1e-1,
            train=train,
            keys_to_skip=keys_to_skip,
        )


if __name__ == "__main__":
    unittest.main()
