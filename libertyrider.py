#!/usr/bin/python3
# coding: utf-8

#litterature source:
#Coursera
#Stackoverflow
#Docs

#25.02.22.10h
#analyse du problème: reponse par machine learning et/ou par dsp
#pour le dsp,un apprentissage m'est nécessaire
#25.02.22.11h experimentations pour créer le dataset
#25.02.22.12h30 pause repas
#25.02.22.13h30 verification de l'intégrité du dataset
#26.02.22 10h optimisation du machine learning pour experimenter sur plusieurs classifiers
#26.02.22 13h30 digestion de la litterature sur cwt et fft
#26.02.22 16h implementation des wavelet et fft dans les fonctions de pipeline et de ML
#26.02.22 18h test des trois modèles

###data utilisée :
# 13 enregistrements de sauts
# et 28 enregistrements de signaux d'accelerations divers (mouvement des bras, montée et descente escalier
#montée sur une chaise, repos, passage position assis à debout et inversement, etc.
# petit dataset -> machine learning plutot que deep learning
# 1er essai de feature engineering : on va "couper" chaque enregistrement à la même
# longueur (6s) et chaque "feature" sera un enregistrement d'accélération en z à la
# date t.
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import sklearn.model_selection
import sklearn.metrics
import sklearn.decomposition
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

#def plot_fft(csvfile):

#def plot_wavelet(csvfile):

def process_z_from_signal(csvfile,dsp=0):
    "renvoie une serie de la composante z (avec process dsp ou non) à partir du csv du signal coupé à 6s"
    dfraw=pd.read_csv(csvfile,sep=';')
    dfraw=dfraw.sort_values('Time (s)', axis=0)
    if dsp=='fft':
        dfraw = dfraw[:600]
        array_raw = dfraw['Acceleration z (m/s^2)'].values
        array_fft = np.fft.fft(array_raw)
        mag=np.sqrt(array_fft.real**2+array_fft.imag**2)
        mag=mag*2/len(array_raw)
        s=pd.Series(data=mag)
    elif dsp=='wavelet':
        dfraw = dfraw[:600]
        array_raw = dfraw['Acceleration z (m/s^2)'].values
        mother_wavelet='gaus1'#sym2 is better from empirical point of view and litterature
        #sym2 not available in cwt pywt.wavelist(kind='continuous'))
        sampling_rate=0.01#100Hz
        scale_for_2Hz_sym2_signal=10 #pywt.scale2frequency('gaus1',10)/0.01==2.0
        scales=np.arange(1,101,1)
        coeff, freq = pywt.cwt(array_raw, scales, mother_wavelet)
        pca = sklearn.decomposition.PCA(n_components=1)
        coeff_pca = pca.fit_transform(coeff)
        s=pd.Series(data=coeff_pca.flatten())
    else:
        dfraw = dfraw[:600]
        s=dfraw['Acceleration z (m/s^2)']
    return s

def build_dataset(folder_saut,folder_divers,dsp):
    "renvoie un dataframe à partir des csv dans folder_saut et folder_divers avec dsp ou non"
    os.chdir(folder_saut)
    files_saut = os.listdir()
    files_saut=sorted(files_saut)
    for i in files_saut:
        if i=='Raw Data.csv':
            series_0=process_z_from_signal(i,dsp)
            series_0.name='saut_0'
        elif i=='Raw Data1.csv':
            series_1 = process_z_from_signal(i,dsp)
            series_1.name = 'saut_1'
            df_0= pd.concat([series_0,series_1], axis=1)
        else:
            s_line_i=process_z_from_signal(i,dsp)
            s_line_i.name='saut_'+str(files_saut.index(i))
            df_0=pd.merge(df_0,s_line_i,left_index=True,right_index=True)
    df_0.loc['saut']=1
    os.chdir(folder_divers)
    files_divers = os.listdir()
    files_divers = sorted(files_divers)
    for i in files_divers:
        s_line_i = process_z_from_signal(i,dsp)
        s_line_i.name = 'divers_' + str(files_divers.index(i))
        s_line_i.loc['saut'] = 0
        df_0=pd.merge(df_0,s_line_i,right_index=True,left_index=True)
    return df_0.T


###machine learning

dico_classifier={'knn':KNeighborsClassifier,
          'naiveb':GaussianNB,'randomforest':RandomForestClassifier,
          'gtree':GradientBoostingClassifier,'neural':MLPClassifier}

def ml(jumpfolder,randomfolder,classifier,dsp=0):
    "lance le machine learning classifier sur les data dans jumpfolder et randomfolder avec le dsp ou non"
    # dataset

    df = build_dataset(jumpfolder, randomfolder,dsp)
    X = df[[i for i in list(df.columns) if i != 'saut']]
    y = df['saut']
    # classifier random forest model training
    clf = dico_classifier[classifier]()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)

    # predictions
    forest_predicted = clf.predict(X_test)

    # scores
    _accuracy=sklearn.metrics.accuracy_score(y_test, forest_predicted)
    _precision=sklearn.metrics.precision_score(y_test, forest_predicted)
    _recall=sklearn.metrics.recall_score(y_test, forest_predicted)
    print(str(classifier)+' Accuracy: {:.2f}'.format(_accuracy))
    print(str(classifier)+' Precision: {:.2f}'.format(_precision))
    print(str(classifier)+' Recall: {:.2f}'.format(_recall))

    # confusion matrix
    confusion_clf = sklearn.metrics.confusion_matrix(y_test, forest_predicted)
    df_clf = pd.DataFrame(confusion_clf,
                          index=[i for i in range(0, 2)], columns=[i for i in range(0, 2)])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(df_clf, annot=True, vmin=0, vmax=11, cmap="Blues")
    plt.title(str(classifier)+' \nAccuracy:{0:.3f}'.format(_accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # courbes precision-recall
    y_score_clf = clf.predict_proba(X_test)
    y_score_df = pd.DataFrame(data=y_score_clf)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_score_df[1])
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall)
    plt.title(str(classifier)+' Precision-Recall Curve \nprecision :{:0.2f}'.format(_precision)+' recall: {:0.2f}'.format(_recall))
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.show()

    # courbes roc
    y_score_clf = clf.predict_proba(X_test)
    y_score_df = pd.DataFrame(data=y_score_clf)
    fpr_clf, tpr_clf, _ = sklearn.metrics.roc_curve(y_test, y_score_df[1])
    roc_auc_clf = sklearn.metrics.auc(fpr_clf, tpr_clf)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_clf, tpr_clf, lw=3, label=str(classifier)+' ROC curve (area = {:0.2f})'.format(roc_auc_clf))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve '+str(classifier)+' \nAUC:{0:.3f}'.format(roc_auc_clf), fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.show()

    return pd.DataFrame(data=(_accuracy,_precision,_recall,roc_auc_clf),index=['accuracy','precision','recall','AUC'],columns=[classifier])

def ml_loop(dsp=0):
    df = pd.DataFrame(data=(0, 0, 0, 0), columns=['init'], index=['accuracy', 'precision', 'recall', 'AUC'])
    for clf in dico_classifier:
        print(clf)
        result_ml = ml(jumpfolder, randomfolder, clf,dsp)
        df = pd.merge(df, result_ml, right_index=True, left_index=True)
    df = df.drop('init', axis=1)
    print(df)
    plt.figure()
    sns.heatmap(df, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.title('scores des classifiers - dsp:'+str(dsp))
    plt.ylabel('scores')
    plt.xlabel('modeles')
    plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--jumps",
                        default=path+'/jumps',
                        help="folder with jumps signal csv",
                        dest='jumps_folder',
                        type=str)
    parser.add_argument("--random",
                        default=path+'/random',
                        help="folder with random signal csv",
                        dest='rand_folder',
                        type=str)
    args=parser.parse_args()
    jumpfolder=args.jumps_folder
    randomfolder=args.rand_folder

    ml_loop(0)
    ml_loop('fft')
    ml_loop('wavelet')




