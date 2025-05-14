import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_time_series(df, channels=None, fig=None):
    """
    Plots time series from defined channels within the dataframe.
    """
    
    if channels is None:
        channels = ['Wind1VelX','BldPitch1','GenSpeed','GenTq','GenPwr']

    if fig is None:
        fig, ax = plt.subplots(len(channels), 1, figsize=(10,3*len(channels)), tight_layout=True)

    for n, channel in enumerate(channels):
        ax[n].plot(df['Time'], df[channel])
        ax[n].set_xlabel('Time')
        ax[n].set_ylabel(channel)
        ax[n].grid()


def plot_characteristic_loads(data, cases=None, channels=None):
    """
    Plots characteristic loads from yaml file for defined channels and DLC cases.
    """

    if cases is None:
        cases = ['1.1','1.3']
    
    if channels is None:
        channels = ['RootMyb1','TwrBsMyt','LSSTipMya','YawBrMyp']

    for case in cases:
        fig, ax = plt.subplots(len(channels), 1, figsize=(10,3*len(channels)), tight_layout=True)

        ax[0].set_title(f"Case {case}")
        for n, channel in enumerate(channels):
            ax[n].scatter(data[case][channel]['wind_speed'], data[case][channel]['load_values'])
            ax[n].axhline(y=data[case][channel]['characteristic_load'], color='k', linestyle='--')
            ax[n].set_xlabel('Wind speed')
            ax[n].set_ylabel(channel)
            ax[n].grid()


