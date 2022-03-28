#!/usr/bin/python3
# coding: utf-8
"""
this is test_machine_learning.py
"""
import os
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pywt
import sklearn.model_selection
import sklearn.metrics
import sklearn.decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pytest
from machine_learning import Pipeline
from machine_learning import MachineLearning
matplotlib.use("Agg")

def test_process_fft():
    "test function of process_fft"
    dfraw = pd.read_csv('dataset1/saut1.csv', sep=';')
    dfraw = dfraw.sort_values('Time (s)', axis=0)
    df_fft=Pipeline.process_fft(dfraw)
    dfraw = dfraw[:600]
    fft = np.fft.fft(dfraw['Absolute acceleration (m/s^2)'].values)
    fft = np.sqrt(fft.real ** 2 + fft.imag ** 2)
    fft = fft * 2 / len(dfraw)
    assert df_fft.columns==['saut']
    assert len(df_fft)==600
    assert df_fft['saut'].values.any()==fft.any()

def test_process_wavelet():
    "test function of process_wavelet"
    dfraw = pd.read_csv('dataset1/saut1.csv', sep=';')
    dfraw = dfraw.sort_values('Time (s)', axis=0)
    df_wavelet=Pipeline.process_wavelet(dfraw)
    dfraw = dfraw[:600]
    array_raw = dfraw['Absolute acceleration (m/s^2)'].values
    coeff, freq = pywt.cwt(array_raw, np.arange(1,101,1), 'gaus1')
    pca = sklearn.decomposition.PCA(n_components=1)
    coeff_pca = pca.fit_transform(coeff)
    assert df_wavelet.columns==['saut']
    assert len(df_wavelet)==100
    assert coeff_pca.any()==df_wavelet.values.any()

def test_process_a_from_signal():
    "test function of process_a_from_signal"
    dfresult=Pipeline.process_a_from_signal('dataset1/saut1.csv')
    assert dfresult.columns==['saut']
    assert len(dfresult)==600
    dfraw = pd.read_csv('dataset1/saut1.csv', sep=';')
    dfraw = dfraw.sort_values('Time (s)', axis=0)
    dfraw = dfraw[:600]
    assert dfraw['Absolute acceleration (m/s^2)'].values.any()==dfresult['saut'].values.any()

@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_machinelearning():
    "test function of machinelearning"
    dfresult=MachineLearning.machinelearning('dataset2/jumps','dataset2/random','randomforest',0)
    assert list(dfresult.index)==['accuracy', 'precision', 'recall', 'AUC']
    assert dfresult.columns=='randomforest'
    assert len(dfresult)==4

def test_compute_scores():
    "test function of compute_scores"
    df_dataset = Pipeline.build_dataset('dataset2/jumps', 'dataset2/random', 0)
    X = df_dataset[[i for i in list(df_dataset.columns) if i != 'saut']]
    y = df_dataset['saut']
    clf = MachineLearning.dico_classifier['gtree']()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=0)
    clf.fit(X_train, y_train)
    _predicted = clf.predict(X_test)
    result=MachineLearning.compute_scores(y_test,_predicted,'gtree')
    assert result[0]==0.8181818181818182
    assert result[1]==1
    assert result[2]==0.3333333333333333
def test_compute_confusion_matrix():
    "test function of compute_confusion_matrix"
    df_dataset = Pipeline.build_dataset('dataset2/jumps', 'dataset2/random', 0)
    X = df_dataset[[i for i in list(df_dataset.columns) if i != 'saut']]
    y = df_dataset['saut']
    clf = MachineLearning.dico_classifier['gtree']()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=0)
    clf.fit(X_train, y_train)
    _predicted = clf.predict(X_test)
    _accuracy=MachineLearning.compute_scores(y_test, _predicted, 'gtree')[0]
    result=MachineLearning.compute_confusion_matrix(y_test,_predicted,_accuracy,'gtree')
    resultdf=result[0]
    assert resultdf.loc[0,0]==8
    assert resultdf.loc[0, 1] == 0
    assert resultdf.loc[1, 0] == 2
    assert resultdf.loc[1,1] == 1
    assert list(resultdf.columns)==[0,1]
    assert list(resultdf.index)==[0,1]
    #dataplotted=result[1].plot_data
    #yplot=result[1].lines.get_ydata()
    plt.close()

@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_plot_precision_recall():
    df_dataset = Pipeline.build_dataset('dataset2/jumps', 'dataset2/random', 0)
    X = df_dataset[[i for i in list(df_dataset.columns) if i != 'saut']]
    y = df_dataset['saut']
    clf = MachineLearning.dico_classifier['gtree']()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=0)
    clf.fit(X_train, y_train)
    _predicted = clf.predict(X_test)
    result=MachineLearning.plot_precision_recall(clf,X_test,y_test,'gtree',_predicted)
    x_plot, y_plot = result.get_xydata().T
    assert list(x_plot)==[0.2727272727272727,1,1]
    assert list(y_plot)==[1,0.3333333333333333,0]
    plt.close()

@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_plot_roc():
    df_dataset = Pipeline.build_dataset('dataset2/jumps', 'dataset2/random', 0)
    X = df_dataset[[i for i in list(df_dataset.columns) if i != 'saut']]
    y = df_dataset['saut']
    clf = MachineLearning.dico_classifier['gtree']()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=0)
    clf.fit(X_train, y_train)
    _predicted = clf.predict(X_test)
    result=MachineLearning.plot_roc(clf,X_test,y_test,'gtree')
    assert result[0]==0.6666666666666666
    x_plot, y_plot = result[1].get_xydata().T
    assert list(x_plot)==[0, 0, 1]
    assert list(y_plot)==[0, 0.3333333333333333,1]
    plt.close()
@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.:UserWarning")
def test_ml_loop():
    "test function of ml_loop"
    dfresult=MachineLearning.ml_loop('dataset2/jumps','dataset2/random',0)
    assert list(dfresult.columns)==['knn','naiveb','randomforest','gtree','neural']
    assert list(dfresult.index)==['accuracy', 'precision', 'recall', 'AUC']
    plt.close('all')