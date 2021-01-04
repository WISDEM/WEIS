__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
import multiprocessing as mp
from fnmatch import fnmatch
from functools import partial

import numpy as np
import pandas as pd
import fatpack

from pCrunch.io import OpenFASTAscii, OpenFASTBinary, OpenFASTOutput


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
        """

        self.outputs = outputs
        self.parse_settings(**kwargs)

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self._directory = kwargs.get("directory", None)
        self._ec = kwargs.get("extreme_channels", [])
        self._mc = kwargs.get("magnitude_channels", {})
        self._fc = kwargs.get("fatigue_channels", {})
        self._td = kwargs.get("trim_data", ())

    def process_outputs(self, cores=1, **kwargs):
        """
        Processes all outputs for summary statistics and configured damage
        equivalent loads.
        """

        if cores > 1:
            stats, extremes, dels = self._process_parallel(cores, **kwargs)

        else:
            stats, extremes, dels = self._process_serial(**kwargs)

        self.post_process(stats, extremes, dels, **kwargs)

    def _process_serial(self, **kwargs):
        """Process outputs in serieal in serial."""

        summary_stats = {}
        extremes = {}
        DELs = {}

        for output in self.outputs:
            filename, stats, extrs, dels = self._process_output(
                output, **kwargs
            )
            summary_stats[filename] = stats
            extremes[filename] = extrs
            DELs[filename] = dels

        return summary_stats, extremes, DELs

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

        pool = mp.Pool(cores)
        returned = pool.map(
            partial(self._process_output, **kwargs), self.outputs
        )
        pool.close()
        pool.join()

        for filename, stats, extrs, dels in returned:
            summary_stats[filename] = stats
            extremes[filename] = extrs
            DELs[filename] = dels

        return summary_stats, extremes, DELs

    def _process_output(self, f, **kwargs):
        """
        Process OpenFAST output `f`.

        Parameters
        ----------
        f : str | tuple
            Path to output or direct output in dict format.
        """

        if isinstance(f, str):
            if self._directory:
                fp = os.path.join(self._directory, f)

            else:
                fp = f

            output = self.read_file(fp)

        else:
            data, channels, dlc = f
            output = OpenFASTOutput(
                data, channels, dlc, magnitude_channels=self._mc
            )

        if self._td:
            output.trim_data(*self._td)

        stats = self.get_summary_stats(output, **kwargs)
        extremes = output.extremes(self._ec)
        dels = self.get_DELs(output, **kwargs)

        return output.filename, stats, extremes, dels

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
                "abs": float(max(np.abs(output[channel]))),
                "integrated": float(np.trapz(output["Time"], output[channel])),
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

    def post_process(self, stats, extremes, dels, **kwargs):
        """Post processes internal data to produce DataFrame outputs."""

        # Summary statistics
        ss = pd.DataFrame.from_dict(stats, orient="index").stack().to_frame()
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        self._summary_stats = ss.unstack().swaplevel(axis=1)

        # Extreme events
        extreme_table = {}
        for _, d in extremes.items():
            for channel, sub in d.items():
                if channel not in extreme_table.keys():
                    extreme_table[channel] = []

                extreme_table[channel].append(sub)
        self._extremes = extreme_table

        # Damage equivalent loads
        dels = pd.DataFrame(dels).T
        self._dels = dels

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

    def get_DELs(self, output, **kwargs):
        """
        Appends computed damage equivalent loads for fatigue channels in
        `self._fc`.

        Parameters
        ----------
        output : OpenFASTOutput
        """

        DELs = {}
        for chan, slope in self._fc.items():
            try:
                DEL = self._compute_del(
                    output[chan], slope, output.elapsed_time, **kwargs
                )
                DELs[chan] = DEL

            except IndexError as e:
                print(f"Channel '{chan}' not found for DEL calculation.")
                DELs[chan] = np.NaN

        return DELs

    @staticmethod
    def _compute_del(ts, slope, elapsed, **kwargs):
        """
        Computes damage equivalent load of input `ts`.

        Parameters
        ----------
        ts : np.array
            Time series to calculate DEL for.
        slope : int | float
            Slope of the fatigue curve.
        elapsed : int | float
            Elapsed time of the time series.
        rainflow_bins : int
            Number of bins used in rainflow analysis.
            Default: 100
        """

        bins = kwargs.get("rainflow_bins", 100)

        ranges = fatpack.find_rainflow_ranges(ts)
        Nrf, Srf = fatpack.find_range_count(ranges, 100)
        DELs = Srf ** slope * Nrf / elapsed
        DEL = DELs.sum() ** (1 / slope)

        return DEL


# class Power_Production(object):
#     '''
#     Class to generate power production stastics
#     '''
#     def __init__(self, **kwargs):
#         # Turbine parameters
#         self.turbine_class = 2

#         for k, w in kwargs.items():
#             try:
#                 setattr(self, k, w)
#             except:
#                 pass

#         super(Power_Production, self).__init__()

#     def prob_WindDist(self, windspeed, disttype='pdf'):
#         '''
#         Generates the probability of a windspeed given the cumulative distribution or probability
#         density function of a Weibull distribution per IEC 61400.

#         NOTE: This uses the range of wind speeds simulated over, so if the simulated wind speed range
#         is not indicative of operation range, using this cdf to calculate AEP is invalid

#         Parameters:
#         -----------
#         windspeed: float or list-like
#             wind speed(s) to calculate probability of
#         disttype: str, optional
#             type of probability, currently supports CDF or PDF
#         Outputs:
#         ----------
#         p_bin: list
#             list containing probabilities per wind speed bin
#         '''
#         if self.turbine_class == 1:
#             Vavg = 50 * 0.2
#         elif self.turbine_class == 2:
#             Vavg = 42.5 * 0.2
#         elif self.turbine_class == 3:
#             Vavg = 37.5 * 0.2

#         # Define parameters
#         k = 2 # Weibull shape parameter
#         c = (2 * Vavg)/np.sqrt(np.pi) # Weibull scale parameter

#         if disttype.lower() == 'cdf':
#             # Calculate probability of wind speed based on WeibulCDF
#             wind_prob = 1 - np.exp(-(windspeed/c)**k)
#         elif disttype.lower() == 'pdf':
#             # Calculate probability of wind speed based on WeibulPDF
#             wind_prob = (k/c) * (windspeed/c)**(k-1) * np.exp(-(windspeed/c)**k)
#         else:
#             raise ValueError('The {} probability distribution type is invalid'.format(disttype))

#         return wind_prob

#     def AEP(self, stats, windspeeds):
#         '''
#         Get AEPs for simulation cases

#         TODO: Print/Save this someplace besides the console

#         Parameters:
#         ----------
#         stats: dict, list, pd.DataFrame
#             Dict (single case), list(multiple cases), df(single or multiple cases) containing
#             summary statistics.
#         windspeeds: list-like
#             List of wind speed values corresponding to each power output in the stats input
#             for a single dataset

#         Returns:
#         --------
#         AEP: List
#             Annual energy production corresponding to
#         '''

#         # Make sure stats is in pandas df
#         if isinstance(stats, dict):
#             stats_df = pdTools.dict2df(stats)
#         elif isinstance(stats, list):
#             stats_df = pdTools.dict2df(stats)
#         elif isinstance(stats, pd.DataFrame):
#             stats_df = stats
#         else:
#             raise TypeError('Input stats is must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')

#         # Check windspeed length
#         if len(windspeeds) == len(stats_df):
#             ws = windspeeds
#         elif int(len(windspeeds)/len(stats_df.columns.levels[0])) == len(stats_df):
#             ws = windspeeds[0:len(stats_df)]
#             print('WARNING: Assuming the input windspeed array is duplicated for each dataset.')
#         else:
#             raise ValueError(
#                 'Length of windspeeds is not the correct length for the input statistics.')

#         # load power array
#         if 'GenPwr' in stats_df.columns.levels[0]:
#             pwr_array = stats_df.loc[:, ('GenPwr', 'mean')]
#             pwr_array = pwr_array.to_frame()
#         elif 'GenPwr' in stats_df.columns.levels[1]:
#             pwr_array = stats_df.loc[:, (slice(None), 'GenPwr', 'mean')]
#         else:
#             raise ValueError("('GenPwr','Mean') does not exist in the input statistics.")

#         # group and average powers by wind speeds
#         pwr_array['windspeeds'] = ws
#         pwr_array = pwr_array.groupby('windspeeds').mean()
#         # find set of wind speeds
#         ws_set = list(set(ws))
#         # wind probability
#         wind_prob = self.prob_WindDist(ws_set, disttype='pdf')
#         # Calculate AEP
#         AEP = np.trapz(pwr_array.T *  wind_prob, ws_set) * 8760

#         return AEP
