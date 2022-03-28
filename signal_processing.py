#!/usr/bin/python3
# coding: utf-8
"""
this is signal_processing.py
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pywt
import detecta
import pandas as pd


############################################################
# Traitement du signal (Digital signal processing)
############################################################
class Fft():

    @staticmethod
    def fft_transform(acc):
        "return a dataframe of fft transform of acc series"
        sample_rate = 100  # phyphox enregistre 100 points par seconde
        array_fft = np.fft.fft(acc.values)
        mag = np.sqrt(array_fft.real**2 + array_fft.imag**2)
        mag = mag * 2 / len(array_fft)
        mag = mag[0:int(len(mag) / 2)]
        mag[0] = 0
        freq = np.arange(0, len(mag), 1) * (sample_rate / len(array_fft))
        df_fft = pd.DataFrame(data={'freq': freq, 'magnitude': mag})
        df_fft['freq'] = df_fft['freq'].round(decimals=0)
        df_fft = df_fft.groupby(['freq']).agg('sum')
        return df_fft

    @staticmethod
    def plot_fft(csv, acc):
        "plot a fft transform of acc series signal of csv file"
        df_fft = Fft.fft_transform(acc)
        plt.figure()
        plt.xlim([0, 10])
        plt.ylim([0, max(df_fft['magnitude'])])
        bars = plt.bar(df_fft.index.values, df_fft['magnitude'].values)
        plt.title(f'FFT transform de l enregistrement {csv}')
        plt.xlabel('f (Hz)', fontsize=16)  ###a revoir pour avoir la frequence
        plt.ylabel('magnitude', fontsize=16)
        plt.show()
        return bars

    @staticmethod
    def jumps_detect_fft(csv, acc):
        """"fft analysis on csv and acc series
        calibrated to detect a frequency of 1 Hz"""
        df_fft = Fft.fft_transform(acc)
        if df_fft.loc[2.0, 'magnitude'] >= 8 and (
                df_fft.loc[3.0, 'magnitude'] <= 6
                and df_fft.loc[4.0, 'magnitude'] <= 6
                and df_fft.loc[5.0, 'magnitude'] <= 6):
            # les sauts ont un pic important à 2 Hz et
            # la course et la marche ont des pics à 3Hz,4Hz et 5Hz
            # plus importants que pour le saut
            print(f'methode fft:saut détecté pour {csv}')
            return 1
        else:
            print(f'methode fft:pas de saut détecté pour {csv}')
            return 0


class Wavelet():

    @staticmethod
    def wavelet_transform(acc):
        "return a numpy array of wavelet transform of acc series"
        mother_wavelet = 'gaus1'
        # sym2 is better from empirical point of view and litterature
        # but sym2 not available in cwt pywt.wavelist(kind='continuous'))
        # sampling_rate : 0.01  (100Hz)
        # scale_for_2Hz_sym2_signal : 10  (pywt.scale2frequency('gaus1',10)/0.01==2.)
        scales = np.arange(1, 101, 1)
        # pylint: disable=unused-variable
        coeff, freq = pywt.cwt(acc.values, scales, mother_wavelet)
        return coeff

    @staticmethod
    def plot_wavelet(csv, acc):
        "plot a wavelet transform of acc series signal of csv file"
        dfcoeff = pd.DataFrame(data=Wavelet.wavelet_transform(acc))
        plt.figure()
        heatmap = sns.heatmap(dfcoeff,
                              cmap='coolwarm',)
        plt.title(f'wavelet transform de l enregistrement {csv}')
        plt.show()
        return heatmap

    @staticmethod
    def jumps_detect_wavelet(csv, acc):
        """analyse par transformée en ondelette calibrée pour détecter une scale
        de 12 à 24 de 100 centisecondes"""
        coeff = Wavelet.wavelet_transform(acc)
        dfacc = pd.DataFrame(data=coeff).abs()
        sum_list = []
        for i in np.arange(0, len(dfacc.columns) - 100, 20):
            # calibration duree signal 100 centisecondes
            sliding_column_list = list(range(i, i + 100))
            # calibration scale du signal 13:25
            sum_result = dfacc.loc[13:25,
                                   sliding_column_list].sum(axis=1).sum(axis=0)
            sum_list.append(sum_result)
        if max(sum_list) > 35000:
            print(f'methode wavelet:saut détecté pour {csv}')
            return 1
        else:
            print(f'methode wavelet:pas de saut détecté pour {csv}')
            return 0


class Peaks():

    @staticmethod
    def plot_peaks(csv, acc):
        "plot peaks of acc series signal of csv file"
        plt.figure()
        plt.xlim([0, len(acc)])
        plt.ylim([0, np.max(acc)])
        plot, = plt.plot(acc.index.values, acc.values)
        plt.title(f'Accélération absolue de l enregistrement {csv}')
        plt.xlabel('time (centisecondes)', fontsize=16)
        plt.ylabel('acceleration (m/s^2)', fontsize=16)
        plt.show()
        return plot

    @staticmethod
    def jumps_detect_peak(csv, acc):
        "analyse par pics calibrée sur une intensité d'accélération de 55 m/s^2"
        result = detecta.detect_peaks(acc.values,
                                      mph=55,
                                      mpd=50,
                                      threshold=0,
                                      show=False)
        if result.size == 0:
            print(f'methode peaks:aucun saut détecté pour {csv}')
            return 0
        else:
            print(f'methode peaks:saut détecté pour {csv}')
            return 1


class Interface():

    @staticmethod
    def plot_signal(csv, dsp='peaks'):
        """produit un graphique de l'enregistrement csvfile
        avec traitement 'fft','wavelet' ou 'peaks'"""
        dfcsv = pd.read_csv(csv, sep=';')
        acc = dfcsv['Absolute acceleration (m/s^2)']
        if dsp == 'fft':
            Fft.plot_fft(csv, acc)
        elif dsp == 'wavelet':
            Wavelet.plot_wavelet(csv, acc)
        elif dsp == 'peaks':
            Peaks.plot_peaks(csv, acc)
        else:
            print("type 'peaks', 'fft', or 'wavelet' please")

    @staticmethod
    def jumps_detect_folder(csv_folder, dsp='peaks'):
        """affiche si les csv dans le repertoire csv_folder en entrée
        sont des sauts ou non
        csv_folder : repertoire contenant les fichiers csv à analyser"""
        files_csv = sorted(
            [csv for csv in os.listdir(csv_folder) if '.csv' in csv])
        results = []
        for csv in files_csv:
            df_csv = pd.read_csv(f'{csv_folder}/{csv}', sep=';')
            acc = df_csv['Absolute acceleration (m/s^2)']
            if dsp == 'peaks':
                results.append(Peaks.jumps_detect_peak(csv, acc))
            elif dsp == 'fft':
                results.append(Fft.jumps_detect_fft(csv, acc))
            elif dsp == 'wavelet':
                results.append(Wavelet.jumps_detect_wavelet(csv, acc))
            else:
                print("type 'peaks', 'fft', or 'wavelet' please")
        nb_saut_detecte = results.count(1)
        print(nb_saut_detecte)
        nb_enregistrement = len(results)
        print(
            f'nombre de sauts détectés : {nb_saut_detecte} sur {nb_enregistrement} enregistrements\n'
        )
        return nb_saut_detecte, nb_enregistrement


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Interface.jumps_detect_folder('dataset1', 'peaks')
        Interface.jumps_detect_folder('dataset1', 'fft')
        Interface.jumps_detect_folder('dataset1', 'wavelet')
    else:
        Interface.jumps_detect_folder(sys.argv[1], 'peaks')
        Interface.jumps_detect_folder(sys.argv[1], 'fft')
        Interface.jumps_detect_folder(sys.argv[1], 'wavelet')
