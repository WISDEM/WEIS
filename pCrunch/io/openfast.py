__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os

import numpy as np
import pandas as pd
from scipy import stats


def dataproperty(f):
    @property
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_data", None) is None:
            raise AttributeError("Output has not been read yet.")
        return f(self, *args, **kwargs)

    return wrapper


class OpenFASTBase:
    """Base OpenFAST output class."""

    def __getitem__(self, chan):
        try:
            idx = np.where(self.channels == chan)[0][0]

        except IndexError:
            raise IndexError(f"Channel '{chan}' not found.")

        return self.data[:, idx]

    def __str__(self):
        return self.description

    @property
    def description(self):
        return getattr(
            self, "_desc", f"Unread OpenFAST output at '{self.filepath}'"
        )

    def trim_data(self, tmin=0, tmax=np.inf):
        """
        Trims `self.data` to the data between `tmin` and `tmax`.

        Parameters
        ----------
        tmin : int | float
            Start time.
        tmax : int | float
            Ending time.
        """

        idx = np.where((self.time >= tmin) & (self.time <= tmax))
        if tmin > max(self.time):
            raise ValueError(
                f"Initial time '{tmin}' is after the end of the simulation."
            )

        self.data = self.data[idx]

    @dataproperty
    def data(self):
        """Returns output data at `self._data`."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @dataproperty
    def time(self):
        return self["Time"]

    @time.setter
    def time(self, time):
        if "Time" in self.channels:
            raise ValueError(f"'Time' channel already exists in output data.")

        self.data = np.append(time, self.data, axis=1)
        self.channels = np.append("Time", self.channels)

    def append_magnitude_channels(self):
        """
        Append the vector magnitude of `channels` to the dataset.

        Parameters
        ----------
        self._magnitude_channels : dict
            Format: 'new-chan' ['chan1', 'chan2', 'chan3'],
        """

        for new_chan, chans in self._magnitude_channels.items():

            if new_chan in self.channels:
                print(f"Channel '{new_chan}' already exists.")
                continue

            arrays = np.array([self[a] for a in chans]).T
            normed = np.linalg.norm(arrays, axis=1).reshape(arrays.shape[0], 1)
            self.data = np.append(self.data, normed, axis=1)
            self.channels = np.append(self.channels, new_chan)

    @property
    def headers(self):
        if getattr(self, "units", None) is None:
            return None

        else:
            return [
                self.channels[k]
                if self.units[k] in ["", "-"]
                else f"{self.channels[k]} [{self.units[k]}]"
                for k in range(self.num_channels)
            ]

    def extremes(self, channels):
        """"""

        sorter = np.argsort(self.channels)
        exists = [c for c in channels if c in self.channels]
        idx = sorter[np.searchsorted(self.channels, exists, sorter=sorter)]

        extremes = {}
        for chan, i in zip(exists, idx):
            idx_max = self.idxmaxs[i]
            extremes[chan] = {
                "time": self.time[idx_max],
                **dict(zip(exists, self.data[idx_max, idx])),
            }

        return extremes

    @dataproperty
    def df(self):
        """Returns `self.data` as a DataFrame."""

        if self.channels is None:
            return pd.DataFrame(self.data)

        return pd.DataFrame(self.data, columns=self.channels)

    @dataproperty
    def num_timesteps(self):
        return self.data.shape[0]

    @dataproperty
    def elapsed_time(self):
        return self.time.max() - self.time.min()

    @dataproperty
    def num_channels(self):
        return self.data.shape[1]

    @dataproperty
    def idxmins(self):
        return np.argmin(self.data, axis=0)

    @dataproperty
    def idxmaxs(self):
        return np.argmax(self.data, axis=0)

    @dataproperty
    def minima(self):
        return np.min(self.data, axis=0)

    @dataproperty
    def maxima(self):
        return np.max(self.data, axis=0)

    @dataproperty
    def ranges(self):
        return self.maxima - self.minima

    @dataproperty
    def variable(self):
        return np.where(self.ranges != 0.0)[0]

    @dataproperty
    def constant(self):
        return np.where(self.ranges == 0.0)[0]

    @dataproperty
    def sums(self):
        return np.sum(self.data, axis=0)

    @dataproperty
    def sums_squared(self):
        return np.sum(self.data ** 2, axis=0)

    @dataproperty
    def sums_cubed(self):
        return np.sum(self.data ** 3, axis=0)

    @dataproperty
    def sums_fourth(self):
        return np.sum(self.data ** 4, axis=0)

    @dataproperty
    def sums_fourth(self):
        return np.sum(self.data ** 4, axis=0)

    @dataproperty
    def second_moments(self):
        return stats.moment(self.data, moment=2, axis=0)

    @dataproperty
    def third_moments(self):
        return stats.moment(self.data, moment=3, axis=0)

    @dataproperty
    def fourth_moments(self):
        return stats.moment(self.data, moment=4, axis=0)

    @dataproperty
    def means(self):
        means = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        means[:, self.constant] = self.minima[self.constant]
        means[:, self.variable] = self.sums[self.variable] / self.num_timesteps
        return means.flatten()

    @dataproperty
    def stddevs(self):
        stddevs = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        stddevs[:, self.variable] = np.sqrt(self.second_moments[self.variable])
        return stddevs.flatten()

    @dataproperty
    def skews(self):
        skews = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        skews[:, self.variable] = (
            self.third_moments[self.variable]
            / np.sqrt(self.second_moments[self.variable]) ** 3
        )
        return skews.flatten()

    @dataproperty
    def kurtosis(self):
        kurtosis = np.zeros(shape=(1, self.num_channels), dtype=np.float64)
        kurtosis[:, self.variable] = (
            self.fourth_moments[self.variable]
        ) / self.second_moments[self.variable] ** 2
        return kurtosis.flatten()


class OpenFASTOutput(OpenFASTBase):
    def __init__(self, data, channels, dlc, **kwargs):
        """
        Creates an instance of `OpenFASTOutput`.

        Parameters
        ----------
        data : np.ndarray
        """

        self._data = data

        if isinstance(channels, list):
            self.channels = np.array(channels)

        else:
            self.channels = channels

        self._dlc = dlc
        self._magnitude_channels = kwargs.get("magnitude_channels", {})
        self.append_magnitude_channels()

    @property
    def filename(self):
        return self._dlc

    @classmethod
    def from_dict(cls, data, name, **kwargs):

        channels = list(data.keys())
        data = np.array(list(data.values())).T

        return cls(data, channels, name, **kwargs)


class OpenFASTBinary(OpenFASTBase):
    """OpenFAST binary output class."""

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTBinary`.

        Parameters
        ----------
        filepath : path-like
        """

        self.filepath = filepath
        self._chan_chars = kwargs.get("chan_char_length", 10)
        self._unit_chars = kwargs.get("unit_char_length", 10)
        self._magnitude_channels = kwargs.get("magnitude_channels", {})

    @property
    def filename(self):
        return os.path.split(self.filepath)[-1]

    def read(self):
        """Reads the binary file."""

        with open(self.filepath, "rb") as f:
            self.fmt = np.fromfile(f, np.int16, 1)[0]

            if self.fmt == 4:
                self._chan_chars = np.fromfile(f, np.int16, 1)[0]
                self._unit_chars = self._chan_chars

            num_channels = np.fromfile(f, np.int32, 1)[0]
            num_timesteps = np.fromfile(f, np.int32, 1)[0]
            num_points = num_channels * num_timesteps
            time_info = np.fromfile(f, np.float64, 2)

            if self.fmt == 3:
                self.slopes = np.ones(num_channels)
                self.offset = np.zeros(num_channels)

            else:
                self.slopes = np.fromfile(f, np.float32, num_channels)
                self.offset = np.fromfile(f, np.float32, num_channels)

            length = np.fromfile(f, np.int32, 1)[0]
            chars = np.fromfile(f, np.uint8, length)
            self._desc = "".join(map(chr, chars)).strip()

            self.build_headers(f, num_channels)
            time = self.build_time(f, time_info, num_timesteps)

            if self.fmt == 3:
                raw = np.fromfile(f, np.float64, count=num_points).reshape(
                    num_timesteps, num_channels
                )
                self.data = np.concatenate(
                    [time.reshape(num_timesteps, 1), raw], 1
                )

            else:
                raw = np.fromfile(f, np.int16, count=num_points).reshape(
                    num_timesteps, num_channels
                )
                self.data = np.concatenate(
                    [
                        time.reshape(num_timesteps, 1),
                        (raw - self.offset) / self.slopes,
                    ],
                    1,
                )

        self.append_magnitude_channels()

    def build_headers(self, f, num_channels):
        """
        Builds the channels, units and headers arrays.

        Parameters
        ----------
        f : file
        num_channels : int
        """

        channels = np.fromfile(
            f, np.uint8, self._chan_chars * (num_channels + 1)
        ).reshape((num_channels + 1), self._chan_chars)
        self.channels = np.array(
            list("".join(map(chr, c)).strip() for c in channels)
        )

        units = np.fromfile(
            f, np.uint8, self._unit_chars * (num_channels + 1)
        ).reshape((num_channels + 1), self._unit_chars)
        self.units = np.array(
            list("".join(map(chr, c)).strip()[1:-1] for c in units)
        )

    def build_time(self, f, info, length):
        """
        Builds the time index based on the file format and index info.

        Parameters
        ----------
        f : file
        info : tuple
            Time index meta information, ('scale', 'offset') or ('t1', 'incr').

        Returns
        -------
        np.ndarray
            Time index for `self.data`.
        """

        if self.fmt == 1:
            scale, offset = info
            data = np.fromfile(f, np.int32, length)
            time = (data - offset) / scale

        else:
            t1, incr = info
            time = t1 + incr * np.arange(length)

        return time


class OpenFASTAscii(OpenFASTBase):
    """
    OpenFAST ASCII output class.

    Built using data from the Mlife post-processor scripts. May not emcompass
    new OpenFAST output formats.
    """

    def __init__(self, filepath, **kwargs):
        """
        Creates an instance of `OpenFASTAscii`.

        Parameters
        ----------
        filepath : path-like
        """

        self.filepath = filepath
        self._magnitude_channels = kwargs.get("magnitude_channels", {})

    @property
    def filename(self):
        return os.path.split(self.filepath)[-1]

    @property
    def time(self):
        return self.data[:, 0]

    def read(self):
        """Reads the ASCII file."""

        with open(self.filepath, "rb") as f:
            chandata, unitdata = self.parse_header(f)
            self.build_headers(chandata, unitdata)
            self.data = np.fromfile(f, float, sep="\t").reshape(
                -1, len(self.channels)
            )

        self.append_magnitude_channels()

    def parse_header(self, f):
        """Reads the header data for file."""

        _start = False
        data = []

        for _line in f:

            line = _line.replace(b"\xb7", b"-").decode().strip()
            data.append(line)

            if _start:
                break

            if line.startswith("Time"):
                _start = True

        self._desc = " ".join([h.replace('"', "") for h in data[:-2]]).strip()

        chandata, unitdata = data[-2:]
        return chandata, unitdata

    def build_headers(self, chandata, unitdata):
        """Unpacks channels and units and builds the combined headers."""

        self.channels = np.array([c for c in chandata.split("\t")])
        self.units = np.array([u[1:-1] for u in unitdata.split("\t")])
