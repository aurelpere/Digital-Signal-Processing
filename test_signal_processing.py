#!/usr/bin/python3
# coding: utf-8
"""
this is test_signal_processing.py
"""
import matplotlib
import numpy as np
import pywt
import pandas as pd
import pytest
from signal_processing import Fft
from signal_processing import Wavelet
from signal_processing import Peaks
from signal_processing import Interface

matplotlib.use("Agg")


def test_fft_transform():
    "test function of fft_transform"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    df_fft = Fft.fft_transform(acc)
    assert df_fft.columns == ['magnitude']
    assert df_fft.index.name == 'freq'
    df_fft.reset_index(inplace=True)
    assert df_fft[df_fft['freq'] == 2.0].loc[2, 'magnitude'] > 9
    assert df_fft[df_fft['freq'] == 1.0].loc[1, 'magnitude'] > 5
    assert df_fft[df_fft['freq'] >= 8.0].loc[:, 'magnitude'].values.any() < 1.5


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_plot_fft():
    "test function of plot_fft"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    df_fft = Fft.fft_transform(acc)
    bars = Fft.plot_fft('dataset2/jumps/Raw Data.csv', acc)
    yvalues = []
    for item in bars:
        yvalues.append(item.get_height())
    assert yvalues == list(df_fft['magnitude'].values)


def test_jumps_detect_fft():
    "test function of detect_fft"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    results = Fft.jumps_detect_fft('dataset2/jumps/Raw Data.csv', acc)
    assert results == 1


def test_wavelet_transform():
    "test function of wavelet transform"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    result = Wavelet.wavelet_transform(acc)
    assert result.any() == pywt.cwt(acc.values, np.arange(1, 101, 1),
                                    'gaus1')[0].any()


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_plot_wavelet():
    "test function of plot_wavelet"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    heatmap = Wavelet.plot_wavelet('dataset2/jumps/Raw Data.csv', acc)
    assert issubclass(type(heatmap), matplotlib.axes.SubplotBase)


def test_jumps_detect_wavelet():
    "test function of jumps_detect_wavelet"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    results = Wavelet.jumps_detect_wavelet('dataset2/jumps/Raw Data.csv', acc)
    assert results == 1


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_plot_peaks():
    "test function of plot_peaks"
    dfcsv = pd.read_csv('dataset2/jumps/Raw Data.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    results = Peaks.plot_peaks('dataset2/jumps/Raw Data.csv', acc)
    # pylint: disable=unused-variable
    x_plot, y_plot = results.get_xydata().T
    assert y_plot.any() == acc.values.any()

def test_jumps_detect_peak():
    "test function of jumps_detect_peak"
    dfcsv = pd.read_csv('dataset1/saut1.csv', sep=';')
    acc = dfcsv['Absolute acceleration (m/s^2)']
    results = Peaks.jumps_detect_peak('dataset1/saut1.csv', acc)
    assert results == 1


def test_jumps_detect_folder():
    "test function of jumps_detect_folder"
    peaks = Interface.jumps_detect_folder('dataset2/jumps', 'peaks')
    assert peaks[0] == 0
    fft = Interface.jumps_detect_folder('dataset2/jumps', 'fft')
    assert fft[0] == 8
    wavelet = Interface.jumps_detect_folder('dataset2/jumps', 'wavelet')
    assert wavelet[0] == 13
