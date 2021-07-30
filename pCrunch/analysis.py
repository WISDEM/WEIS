__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import fatpack

from pCrunch.io import OpenFASTAscii, OpenFASTBinary #, OpenFASTOutput

# Could use a dict or namedtuple here, but this standardizes things a bit better for users
class FatigueParams:
    """Simple data structure of parameters needed by fatigue calculation."""

    def __init__(self, load2stress=1.0, slope=4.0, ult_stress=1.0, S_intercept=0.0):
        """
        Creates an instance of `FatigueParams`.

        Parameters
        ----------
        load2stress : float (optional)
            Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
        slope : float (optional)
            Wohler exponent in the traditional SN-curve of S = A * N ^ -(1/m)
        ult_stress : float (optional)
            Ultimate stress for use in Goodman equivalent stress calculation
        S_intercept : float (optional)
            Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
        """

        self.load2stress = float(load2stress)
        self.slope = float(slope)
        self.ult_stress = float(ult_stress)
        self.S_intercept = float(S_intercept) if float(S_intercept) > 0.0 else self.ult_stress

    def copy(self):
        return FatigueParams(load2stress=self.load2stress, slope=self.slope,
                             ult_stress=self.ult_stress, S_intercept=self.S_intercept)

class LoadsAnalysis:
    """Implementation of `mlife` in python."""

    def __init__(self, outputs, **kwargs):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        outputs : list
            List of OpenFAST output filepaths or dicts of OpenFAST outputs.
        directory : str (optional)
            If directory is passed, list of files will be treated as relative
            and appended to the directory.
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
        magnitude_channels : dict (optional)
            Additional channels as vector magnitude of other channels.
            Format: 'new-chan': ['chan1', 'chan2', 'chan3']
        trim_data : tuple
            Trim processed outputs to desired times.
            Format: (min, max)
        return_intermediate : bool
        """

        self.outputs = outputs
        self.parse_settings(**kwargs)

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self._directory = kwargs.get("directory", None)
        self._ec = kwargs.get("extreme_channels", True)
        self._mc = kwargs.get("magnitude_channels", {})
        self._fc = kwargs.get("fatigue_channels", {})
        self._td = kwargs.get("trim_data", ())
        
    def process_outputs(self, cores=1, **kwargs):
        """
        Processes all outputs for summary statistics and configured damage
        equivalent loads.
        """

        if cores > 1:
            stats, extrs, dels, damage = self._process_parallel(cores, **kwargs)

        else:
            stats, extrs, dels, damage = self._process_serial(**kwargs)

        summary_stats, extremes, DELs, Damage = self.post_process(
            stats, extrs, dels, damage, **kwargs
        )
        self._summary_stats = summary_stats
        self._extremes = extremes
        self._dels = DELs
        self._damage = Damage

    def _process_serial(self, **kwargs):
        """Process outputs in serieal in serial."""

        summary_stats = {}
        extremes = {}
        DELs = {}
        Damage = {}

        for output in self.outputs:
            filename, stats, extrs, dels, damage = self._process_output(
                output, **kwargs
            )
            summary_stats[filename] = stats
            extremes[filename] = extrs
            DELs[filename] = dels
            Damage[filename] = damage

        return summary_stats, extremes, DELs, Damage

    def _process_parallel(self, cores, **kwargs):
        """
        Process outputs in parallel.

        Parameters
        ----------
        cores : int
        """

        summary_stats = {}
        extremes = {}
        DELs = {}
        Damage = {}

        pool = mp.Pool(cores)
        returned = pool.map(
            partial(self._process_output, **kwargs), self.outputs
        )
        pool.close()
        pool.join()

        for filename, stats, extrs, dels, damage in returned:
            summary_stats[filename] = stats
            extremes[filename] = extrs
            DELs[filename] = dels
            Damage[filename] = damage

        return summary_stats, extremes, DELs, Damage

    def _process_output(self, f, **kwargs):
        """
        Process OpenFAST output `f`.

        Parameters
        ----------
        f : str | OpenFASTOutput
            Path to output or direct output in dict format.
        """

        if isinstance(f, str):
            output = self.read_file(f)

        else:
            output = f

        if self._td:
            output.trim_data(*self._td)

        stats = self.get_summary_stats(output, **kwargs)

        if self._ec is True:
            extremes = output.extremes(output.channels)

        elif isinstance(self._ec, list):
            extremes = output.extremes(self._ec)

        dels, damage = self.get_DELs(output, **kwargs)

        return output.filename, stats, extremes, dels, damage

    def get_summary_stats(self, output, **kwargs):
        """
        Appends summary statistics to `self._summary_statistics` for each file.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        fstats = {}
        for channel in output.channels:
            if channel in ["time", "Time"]:
                continue

            fstats[channel] = {
                "min": float(min(output[channel])),
                "max": float(max(output[channel])),
                "std": float(np.std(output[channel])),
                "mean": float(np.mean(output[channel])),
                "median": float(np.median(output[channel])),
                "abs": float(max(np.abs(output[channel]))),
                "integrated": float(np.trapz(output[channel], x=output["Time"])),
            }

        return fstats

    def get_extreme_events(self, output, channels, **kwargs):
        """
        Returns extreme events of `output`.

        Parameters
        ----------
        output : OpenFASTOutput
        channels : list
        """

        return output.extremes(channels)

    @staticmethod
    def post_process(stats, extremes, dels, damage, **kwargs):
        """Post processes internal data to produce DataFrame outputs."""

        # Summary statistics
        ss = pd.DataFrame.from_dict(stats, orient="index").stack().to_frame()
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        summary_stats = ss.unstack().swaplevel(axis=1)

        # Extreme events
        extreme_table = {}
        for _, d in extremes.items():
            for channel, sub in d.items():
                if channel not in extreme_table.keys():
                    extreme_table[channel] = []

                extreme_table[channel].append(sub)
        extremes = extreme_table

        # Damage and Damage Equivalent Loads
        dels = pd.DataFrame(dels).T
        damage = pd.DataFrame(damage).T

        return summary_stats, extremes, dels, damage

    def read_file(self, f):
        """
        Reads input file `f` and returns an instsance of one of the
        `OpenFASTOutput` subclasses.

        Parameters
        ----------
        f : str
            Filename that is appended to `self.directory`
        """

        if self._directory:
            fp = os.path.join(self._directory, f)

        else:
            fp = f

        try:
            output = OpenFASTAscii(fp, magnitude_channels=self._mc)
            output.read()

        except UnicodeDecodeError:
            output = OpenFASTBinary(fp, magnitude_channels=self._mc)
            output.read()

        return output

    def get_load_rankings(self, ranking_vars, ranking_stats, **kwargs):
        """
        Returns load rankings across all outputs in `self.outputs`.

        Parameters
        ----------
        rankings_vars : list
            List of variables to evaluate for the ranking process.
        ranking_stats : list
            Summary statistic to evalulate. Currently supports 'min', 'max',
            'abs', 'mean', 'std'.
        """

        summary_stats = self.summary_stats.copy().swaplevel(axis=1).stack()

        out = []
        for var, stat in zip(ranking_vars, ranking_stats):

            if not isinstance(var, list):
                var = [var]

            col = pd.MultiIndex.from_product([self.outputs, var])
            if stat in ["max", "abs"]:
                res = (
                    *summary_stats.loc[col][stat].idxmax(),
                    stat,
                    summary_stats.loc[col][stat].max(),
                )

            elif stat == "min":
                res = (
                    *summary_stats.loc[col][stat].idxmin(),
                    stat,
                    summary_stats.loc[col][stat].min(),
                )

            elif stat in ["mean", "std"]:
                res = (
                    np.NaN,
                    ", ".join(var),
                    stat,
                    summary_stats.loc[col][stat].mean(),
                )

            else:
                raise NotImplementedError(
                    f"Statistic '{stat}' not supported for load ranking."
                )

            out.append(res)

        return pd.DataFrame(out, columns=["file", "channel", "stat", "val"])

    @property
    def summary_stats(self):
        """Returns summary statistics for all outputs in `self.outputs`."""

        if getattr(self, "_summary_stats", None) is None:
            raise ValueError("Outputs have not been processed.")

        return self._summary_stats

    @property
    def extreme_events(self):
        """Returns extreme events for all files and channels in `self._ec`."""

        if getattr(self, "_extremes", None) is None:
            raise ValueError("Outputs have not been processed.")

        return self._extremes

    @property
    def DELs(self):
        """Returns damage equivalent loads for all channels in `self._fc`"""

        if getattr(self, "_dels", None) is None:
            raise ValueError("Outputs have not been processed.")

        return self._dels

    @property
    def damage(self):
        """Returns Palmgren/Miner damage for all channels in `self._fc`"""

        if getattr(self, "_damage", None) is None:
            raise ValueError("Outputs have not been processed.")

        return self._damage

    def get_DELs(self, output, **kwargs):
        """
        Appends computed damage equivalent loads for fatigue channels in
        `self._fc`.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        DELs = {}
        D = {}

        for chan, fatparams in self._fc.items():

            try:
                DELs[chan], D[chan] = self._compute_del(
                    output[chan], output.elapsed_time,
                    fatparams.load2stress, fatparams.slope,
                    fatparams.ult_stress, fatparams.S_intercept,
                    **kwargs
                )

            except IndexError:
                print(f"Channel '{chan}' not included in DEL calculation.")
                DELs[chan] = np.NaN
                D[chan] = np.NaN

        return DELs, D

    @staticmethod
    def _compute_del(ts, elapsed, load2stress, slope, Sult, Sc=0.0, **kwargs):
        """
        Computes damage equivalent load of input `ts`.

        Parameters
        ----------
        ts : np.array
            Time series to calculate DEL for.
        elapsed : int | float
            Elapsed time of the time series.
        load2stress : float (optional)
            Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
        slope : int | float
            Slope of the fatigue curve.
        Sult : float (optional)
            Ultimate stress for use in Goodman equivalent stress calculation
        Sc : float (optional)
            Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
        rainflow_bins : int
            Number of bins used in rainflow analysis.
            Default: 100
        goodman_correction: boolean
            Whether to apply Goodman mean correction to loads and stress
            Default: False
        return_damage: boolean
            Whether to compute both DEL and true damage
            Default: False
        """

        bins = kwargs.get("rainflow_bins", 100)
        return_damage = kwargs.get("return_damage", False)
        goodman = kwargs.get("goodman_correction", False)
        Scin = Sc if Sc > 0.0 else Sult
        
        # Working with loads for DELs
        try:
            F, Fmean = fatpack.find_rainflow_ranges(ts, return_means=True)
        except:
            F = Fmean = np.zeros(1)
        if goodman and load2stress > 0.0:
            F = fatpack.find_goodman_equivalent_stress(F, Fmean, Sult/load2stress)
        Nrf, Frf = fatpack.find_range_count(F, bins)
        DELs = Frf ** slope * Nrf / elapsed
        DEL = DELs.sum() ** (1.0 / slope)
        # With fatpack do:
        #curve = fatpack.LinearEnduranceCurve(1.)
        #curve.m = slope
        #curve.Nc = elapsed
        #DEL = curve.find_miner_sum(np.c_[Frf, Nrf]) ** (1 / slope)

        # Compute Palmgren/Miner damage using stress
        D = np.nan # default return value
        if return_damage and load2stress > 0.0:
            try:
                S, Mrf = fatpack.find_rainflow_ranges(ts*load2stress, return_means=True)
            except:
                S = Mrf = np.zeros(1)
            if goodman:
                S = fatpack.find_goodman_equivalent_stress(S, Mrf, Sult)
            Nrf, Srf = fatpack.find_range_count(S, bins)
            curve = fatpack.LinearEnduranceCurve(Scin)
            curve.m = slope
            curve.Nc = 1
            D = curve.find_miner_sum(np.c_[Srf, Nrf])

        return DEL, D


class PowerProduction:
    """Class to generate power production estimates."""

    def __init__(self, turbine_class, **kwargs):
        """
        Creates an instance of `PowerProduction`.

        Parameters
        ----------
        turbine_class : int
        """

        self.turbine_class = turbine_class

    def prob_WindDist(self, windspeed, disttype="pdf"):
        """
        Generates the probability of a windspeed given the cumulative
        distribution or probability density function of a Weibull distribution
        per IEC 61400.

        NOTE: This uses the range of wind speeds simulated over, so if the
        simulated wind speed range is not indicative of operation range, using
        this cdf to calculate AEP is invalid

        Parameters
        ----------
        windspeed : float or list-like
            wind speed(s) to calculate probability of
        disttype : str, optional
            type of probability, currently supports CDF or PDF

        Returns
        -------
        p_bin : list
            list containing probabilities per wind speed bin
        """

        if self.turbine_class in [1, "I"]:
            Vavg = 50 * 0.2

        elif self.turbine_class in [2, "II"]:
            Vavg = 42.5 * 0.2

        elif self.turbine_class in [3, "III"]:
            Vavg = 37.5 * 0.2

        # Define parameters
        k = 2  # Weibull shape parameter
        c = (2 * Vavg) / np.sqrt(np.pi)  # Weibull scale parameter

        if disttype.lower() == "cdf":
            # Calculate probability of wind speed based on WeibulCDF
            wind_prob = 1 - np.exp(-(windspeed / c) ** k)

        elif disttype.lower() == "pdf":
            # Calculate probability of wind speed based on WeibulPDF
            wind_prob = (
                (k / c)
                * (windspeed / c) ** (k - 1)
                * np.exp(-(windspeed / c) ** k)
            )

        else:
            raise ValueError(
                f"The {disttype} probability distribution type is invalid"
            )

        return wind_prob

    def AEP(self, stats, windspeeds, pwr_curve_vars):
        """
        Calculate AEP for simulation cases

        Parameters:
        ----------
        stats : pd.DataFrame
            DataFrame containing summary statistics of each DLC.
        windspeeds : list-like
            List of wind speed values corresponding to each power output in the stats input
            for a single dataset

        Returns:
        --------
        AEP : list
            List of annual energy productions.
        """

        assert len(stats) == len(windspeeds)

        pwr = stats.loc[:, ("GenPwr", "mean")].to_frame()

        # Group and average powers by wind speeds
        pwr["windspeeds"] = windspeeds
        pwr = pwr.groupby("windspeeds").mean()

        # Wind probability
        unique = list(np.unique(windspeeds))
        wind_prob = self.prob_WindDist(unique, disttype="pdf")

        # Calculate AEP
        AEP = np.trapz(pwr.T * wind_prob, unique) * 8760

        perf_data = {"U": unique}
        for var in pwr_curve_vars:
            perf_array = stats.loc[:, (var, "mean")].to_frame()
            perf_array["windspeed"] = windspeeds
            perf_array = perf_array.groupby("windspeed").mean()
            perf_data[var] = perf_array[var]

        return AEP, perf_data
