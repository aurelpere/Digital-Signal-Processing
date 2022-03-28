#!/usr/bin/python3
# coding: utf-8
"""
this is machine_learning.py
"""
import os
import argparse
import pandas as pd
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

class Pipeline():
    @staticmethod
    def process_fft(dfraw):
        "process dataframe to fft transform its signal"
        dfraw = dfraw[:600]
        array_raw = dfraw['Absolute acceleration (m/s^2)'].values
        array_fft = np.fft.fft(array_raw)
        mag = np.sqrt(array_fft.real**2 + array_fft.imag**2)
        mag = mag * 2 / len(array_raw)
        df_result = pd.DataFrame(data=mag).rename(columns={0: 'saut'})
        return df_result

    @staticmethod
    def process_wavelet(dfraw):
        "process dataframe to wavelet transform its signal"
        dfraw = dfraw[:600]
        array_raw = dfraw['Absolute acceleration (m/s^2)'].values
        mother_wavelet = 'gaus1'  # sym2 is better from empirical point of view and litterature
        # sym2 not available in cwt pywt.wavelist(kind='continuous'))
        # sampling_rate = 0.01  # 100Hz
        # scale_for_2Hz_sym2_signal = 10  # pywt.scale2frequency('gaus1',10)/0.01==2.0
        scales = np.arange(1, 101, 1)
        coeff, freq = pywt.cwt(array_raw, scales, mother_wavelet)
        pca = sklearn.decomposition.PCA(n_components=1)
        coeff_pca = pca.fit_transform(coeff)
        df_result = pd.DataFrame(data=coeff_pca.flatten()).rename(
            columns={0: 'saut'})
        return df_result

    @staticmethod
    def process_a_from_signal(csvfile, dsp=0):
        "return dataframe of absolute acceleration or fft or wavelet of csv signal (6 s)"
        dfraw = pd.read_csv(csvfile, sep=';')
        dfraw = dfraw.sort_values('Time (s)', axis=0)
        if dsp == 'fft':
            df_result = Pipeline.process_fft(dfraw)
        elif dsp == 'wavelet':
            df_result = Pipeline.process_wavelet(dfraw)
        else:
            df_result = dfraw[:600].drop(
                labels=[
                    'Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)',
                    'Acceleration z (m/s^2)'
                ],
                axis=1).rename(columns={'Absolute acceleration (m/s^2)': 'saut'})
        return df_result

    @staticmethod
    def build_dataset(jumpfolder, randomfolder, dsp):
        "return a dataframe of all csv signals input in jumpfolder and randomfolder"
        cwd = os.getcwd()
        os.chdir(jumpfolder)
        files_jump = sorted(os.listdir())
        df_0 = pd.DataFrame(data=[0] * 600,
                            columns=['init'],
                            index=np.arange(0, 600, 1))
        for i in files_jump:
            df_line_i = Pipeline.process_a_from_signal(
                i, dsp).rename(columns={'saut': f'saut_{files_jump.index(i)}'})
            df_0 = pd.merge(df_0, df_line_i, left_index=True, right_index=True)
        os.chdir(cwd)
        df_0.drop('init', axis=1, inplace=True)
        df_0.loc['saut'] = 1
        os.chdir(randomfolder)
        files_random = sorted(os.listdir())
        for i in files_random:
            df_line_i = Pipeline.process_a_from_signal(
                i, dsp).rename(columns={'saut': f'random_{files_random.index(i)}'})
            df_line_i.loc['saut'] = 0
            df_0 = pd.merge(df_0, df_line_i, right_index=True, left_index=True)
        os.chdir(cwd)
        return df_0.T

class MachineLearning():
    @staticmethod
    def machinelearning(jumpfolder, randomfolder, classifier, dsp=0):
        "process machine learning on jumfolder and randomfolder data with classifier"
        # dataset
        df_dataset = Pipeline.build_dataset(jumpfolder, randomfolder, dsp)
        X = df_dataset[[i for i in list(df_dataset.columns) if i != 'saut']]
        y = df_dataset['saut']
        # classifier model training
        clf = MachineLearning.dico_classifier[classifier]()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, random_state=0)
        clf.fit(X_train, y_train)
        # predictions
        _predicted = clf.predict(X_test)
        # scores
        _accuracy, _precision, _recall = MachineLearning.compute_scores(y_test, _predicted,
                                                        classifier)
        # confusion matrix
        MachineLearning.compute_confusion_matrix(y_test, _predicted, _accuracy, classifier)
        # courbes precision-recall
        MachineLearning.plot_precision_recall(clf, X_test, y_test, classifier, _predicted)
        # roc
        roc_auc_clf = MachineLearning.plot_roc(clf, X_test, y_test, classifier)[0]
        return pd.DataFrame(data=(_accuracy, _precision, _recall, roc_auc_clf),
                            index=['accuracy', 'precision', 'recall', 'AUC'],
                            columns=[classifier])

    @staticmethod
    def compute_scores(y_test, _predicted, classifier):
        "compute machine learning scores"
        _accuracy = sklearn.metrics.accuracy_score(y_test, _predicted)
        _precision = sklearn.metrics.precision_score(y_test, _predicted)
        _recall = sklearn.metrics.recall_score(y_test, _predicted)
        print(str(classifier) + ' Accuracy: {:.2f}'.format(_accuracy))
        print(str(classifier) + ' Precision: {:.2f}'.format(_precision))
        print(str(classifier) + ' Recall: {:.2f}'.format(_recall))
        return (_accuracy, _precision, _recall)

    @staticmethod
    def compute_confusion_matrix(y_test, _predicted, _accuracy, classifier):
        "compute confusion matrix"
        confusion_clf = sklearn.metrics.confusion_matrix(y_test, _predicted)
        df_clf = pd.DataFrame(confusion_clf,
                              index=list(range(0, 2)),
                              columns=list(range(0, 2)))
        plt.figure(figsize=(5.5, 4))
        ax_heatmap=sns.heatmap(df_clf, annot=True, vmin=0, vmax=11, cmap="Blues")
        plt.title(str(classifier) + ' \nAccuracy:{0:.3f}'.format(_accuracy))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return df_clf,ax_heatmap
    @staticmethod
    def plot_precision_recall(clf, X_test, y_test, classifier, _predicted):
        "plot precision recall curve"
        _precision = sklearn.metrics.precision_score(y_test, _predicted)
        _recall = sklearn.metrics.recall_score(y_test, _predicted)
        y_score_clf = clf.predict_proba(X_test)
        y_score_df = pd.DataFrame(data=y_score_clf)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            y_test, y_score_df[1])
        closest_zero = np.argmin(np.abs(thresholds))
        closest_zero_p = precision[closest_zero]
        closest_zero_r = recall[closest_zero]
        plt.figure()
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        result,=plt.plot(precision, recall)
        plt.title(
            str(classifier) +
            ' Precision-Recall Curve \nprecision :{:0.2f}'.format(_precision) +
            ' recall: {:0.2f}'.format(_recall))
        plt.plot(closest_zero_p,
                 closest_zero_r,
                 'o',
                 markersize=12,
                 fillstyle='none',
                 c='r',
                 mew=3)
        plt.xlabel('Precision', fontsize=16)
        plt.ylabel('Recall', fontsize=16)
        plt.show()
        return result
    @staticmethod
    def plot_roc(clf, X_test, y_test, classifier):
        "plot roc curve"
        y_score_clf = clf.predict_proba(X_test)
        y_score_df = pd.DataFrame(data=y_score_clf)
        fpr_clf, tpr_clf, _ = sklearn.metrics.roc_curve(y_test, y_score_df[1])
        roc_auc_clf = sklearn.metrics.auc(fpr_clf, tpr_clf)
        plt.figure()
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        result,=plt.plot(fpr_clf,
                 tpr_clf,
                 lw=3,
                 label=str(classifier) +
                 ' ROC curve (area = {:0.2f})'.format(roc_auc_clf))
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC curve ' + str(classifier) +
                  ' \nAUC:{0:.3f}'.format(roc_auc_clf),
                  fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.show()
        return roc_auc_clf,result

    dico_classifier = { 'knn': KNeighborsClassifier,
                        'naiveb': GaussianNB,
                        'randomforest': RandomForestClassifier,
                        'gtree': GradientBoostingClassifier,
                        'neural': MLPClassifier}

    @staticmethod
    def ml_loop(jumpfolder, randomfolder, dsp=0):
        """process machine learning for all classifiers in dico_classifier
        and plot a heatmap of their accuracy, precision, recall, AUC"""
        df_result = pd.DataFrame(data=(0, 0, 0, 0),
                                 columns=['init'],
                                 index=['accuracy', 'precision', 'recall', 'AUC'])
        for clf in MachineLearning.dico_classifier:
            print(clf)
            result_ml = MachineLearning.machinelearning(jumpfolder, randomfolder, clf, dsp)
            df_result = pd.merge(df_result,
                                 result_ml,
                                 right_index=True,
                                 left_index=True)
        df_result.drop('init', axis=1, inplace=True)
        MachineLearning.plot_heatmap(df_result, dsp)
        return df_result
    @staticmethod
    def plot_heatmap(dataframe, dsp):
        "plot heatmap of accuracy, precision, recall, AUC"
        plt.figure()
        sns.heatmap(dataframe, annot=True, vmin=0, vmax=1, cmap="Blues")
        plt.title('scores des classifiers - dsp:' + str(dsp))
        plt.ylabel('scores')
        plt.xlabel('modeles')
        plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--jumps",
                        default=f'{path}/dataset2/jumps',
                        help="folder with jumps signal csv",
                        type=str)
    parser.add_argument("--random",
                        default=f'{path}/dataset2/random',
                        help="folder with random signal csv",
                        type=str)
    parser.add_argument("--dsp",
                        default=0,
                        help="dsp to process : 0 or fft or wavelet",
                        type=str)
    args = parser.parse_args()
    jumpfolderarg = args.jumps
    randomfolderarg = args.random
    dsparg = args.dsp
    MachineLearning.ml_loop(jumpfolderarg, randomfolderarg, dsparg)
