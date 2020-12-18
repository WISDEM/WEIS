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

from pCrunch.io import OpenFASTAscii, OpenFASTBinary


class LoadsAnalysis:
    """Implementation of `mlife` in python."""

    def __init__(self, directory, **kwargs):
        """
        Creates an instance of `pyLife`.

        Parameters
        ----------
        directory : path-like
            Path to OpenFAST output files.
        extensions : list
            List of extensions to include from `directory`.
            Not used if `files` is passed.
            Default: ["*.out", "*.outb"]
        operating_files : list (optional)
            Operating files to read.
            Default: []
        idling_files : list (optional)
            Idling files to read.
            Default: []
        discrete_files : list (optional)
            Discrete files to read.
            Default: []
        aggregate_statistics : bool (optional)
            Flag for calculating aggregate statistics.
            Default: True
        calculated_channels : list (optional)
            Flags for additional calculated channels.
            Default: []
        fatigue_channels : dict (optional)
            Dictionary with format:
            'channel': 'fatigue slope'
        """

        self.directory = directory
        self.parse_settings(**kwargs)

    @staticmethod
    def valid_extension(fp, extensions):
        return any([fnmatch(fp, ext) for ext in extensions])

    def parse_settings(self, **kwargs):
        """Parses settings from input kwargs."""

        self._files = {
            "operating": kwargs.get("operating_files", []),
            "idle": kwargs.get("idling_files", []),
            "discrete": kwargs.get("discrete_files", []),
        }

        self._cc = kwargs.get("calculated_channels", [])
        self._fc = kwargs.get("fatigue_channels", {})

    def process_files(self, cores=1, **kwargs):
        """
        Processes all files for summary statistics, aggregate statistics and
        configured damage equivalent loads.
        """

        if cores > 1:
            stats, dels = self._process_parallel(cores, **kwargs)

        else:
            stats, dels = self._process_serial(**kwargs)

        self.post_process(stats, dels, **kwargs)

    def _process_serial(self, **kwargs):
        """"""

        summary_stats = {}
        DELs = {}

        for i, f in enumerate(self.files):
            filename, stats, dels = self._process_file(f, **kwargs)
            summary_stats[filename] = stats
            DELs[filename] = dels

        return summary_stats, DELs

    def _process_parallel(self, cores, **kwargs):
        """"""

        summary_stats = {}
        DELs = {}

        pool = mp.Pool(cores)
        returned = pool.map(partial(self._process_file, **kwargs), self.files)
        pool.close()
        pool.join()

        for filename, stats, dels in returned:
            summary_stats[filename] = stats
            DELs[filename] = dels

        return summary_stats, DELs

    def _process_file(self, file, **kwargs):
        """"""

        output = self.read_file(file)
        stats = self.get_summary_stats(output, **kwargs)
        dels = self.get_DELs(output, **kwargs)
        return output.filename, stats, dels

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

    def post_process(self, stats, dels, **kwargs):
        """Post processes internal data to produce DataFrame outputs."""

        # Summary statistics
        ss = pd.DataFrame.from_dict(stats, orient="index").stack().to_frame()
        ss = pd.DataFrame(ss[0].values.tolist(), index=ss.index)
        self._summary_stats = ss.unstack().swaplevel(axis=1)

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

        fp = os.path.join(self.directory, f)
        try:
            output = OpenFASTAscii(fp)
            output.read()

        except UnicodeDecodeError:
            output = OpenFASTBinary(fp)
            output.read()

        return output

    def get_load_rankings(self, ranking_vars, ranking_stats, **kwargs):
        """
        Returns load rankings across all files in `self.files`.

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

            col = pd.MultiIndex.from_product([self.files, var])
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
    def operating_files(self):
        return self._files["operating"]

    @property
    def idle_files(self):
        return self._files["idle"]

    @property
    def discrete_files(self):
        return self._files["discrete"]

    @property
    def files(self):
        return [*self.operating_files, *self.idle_files, *self.discrete_files]

    @property
    def filepaths(self):
        return [os.path.join(self.directory, fn) for fn in self._files]

    @property
    def summary_stats(self):
        """Returns summary statistics for all files in `self.files`."""

        if getattr(self, "_summary_stats", None) is None:
            raise ValueError("Files have not been processed.")

        return self._summary_stats

    @property
    def DELs(self):
        """Returns damage equivalent loads for all channels in `self._fc`"""

        if getattr(self, "_dels", None) is None:
            raise ValueError("Files have not been processed.")

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
                DELS[chan] = np.NaN

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
