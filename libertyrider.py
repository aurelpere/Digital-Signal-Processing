#!/usr/bin/python3
# coding: utf-8
##############################################
# démarche temporalisée
##############################################
# 25.02.22.10h
# analyse du problème: reponse par machine learning et/ou par dsp
# pour le dsp,un apprentissage m'est nécessaire
# 25.02.22.11h experimentations pour créer le dataset
# 25.02.22.13h30 verification de l'intégrité du dataset
# 26.02.22 10h optimisation du machine learning pour experimenter sur plusieurs classifiers
# 26.02.22 13h30 digestion de la litterature sur cwt et fft
# 26.02.22 16h implementation des wavelet et fft dans les fonctions de pipeline et de ML
# 26.02.22 18h test des trois modèles de machine learning
# 28.02.22 11h relecture de la consigne : pas de machine learning, rédaction de la méthode dsp
# 28.02.22 15h création du dataset de sauts, de marche et de course
# 28.02.22 15h verification des données et complément sur la rédaction de la méthode dsp
# 28.02.22 17h traitement par dsp fft et wavelet et analyse graphique des transformées obtenues
# 28.02.22 18h implementation python de la méthode pics avec stackoverflow
# 29.02.22 10h resultats decevants pour la methode pics de stackoverflow. implementation de detecta
# 29.02.22 11h implementation fonction csv avec methode detecta
# 29.02.22 12h implementation fonction csv avec methode fft
# 29.02.22 15h implemetation fonction csv avec methode wavelet
# 29.02.22 17h réalisation des mesures timeit
# 29.02.22 18h terminé

################################################
# Méthodologie proposée:
################################################

# la différence entre un signal de saut sur place et un signal de marche ou de course
# étant trivialement l'accélération selon x et y dans un repère carthésien où x,y sont
# horizontaux et z vertical, on pourrait procéder en distinguant la présence de données
# positives dans les données en x et y.
# mais le repere du téléphone est en principe celui dispo ici : http://developer.android.com/images/axis_device.png
# Compte tenu de l'orientation du téléphone dans ma poche, ce serait 'y' qui serait
# a priori l'axe le plus pertinent.
# L'exercice se complique a priori si on ne connait pas l'orientation du téléphone mais
# les enregistrements de phyphox montrent une présence du signal de saut, de marche et de course
# sur les 3 axes. La réduction à un seul axe ne devrait donc en principe pas poser de problèmes.
# Cependant, une petite analyse des signaux montre que
# l'intensité des signaux en 'y'  varie du simple au quadruple selon les sauts
# Elle est plus importante en 'y' pour les sauts que pour la marche (environ le double)
# Elle est d'un ordre de grandeur similaire pour la course
# L'orientation du téléphone dans la poche n'était probablement pas identique selon chaque
# enregistrement.
# Les données me semblent plus fiables lorsqu'on prend la composante
# 'Absolute acceleration' qui semble correspondre à la norme du vecteur acceleration
# l'intensité des sauts est sensiblement la même pour chaque saut
# l'intensité est 50% à 100% plus importante pour les sauts que pour la marche

# On se place dans la perspective d'extrapoler pour détecter un signal de chute de moto, donc on va
# axer la méthode sur les algorithmes de "traitement du signal" (digital signal processing)

# d'apres l'article "step detection" de wikipedia, un algorithme de type "online"
# serait plus adapté à la détection de signaux sur une appli mobile puisque le signal
# serait analysé quand il arrive
# Dans notre cas, on utilisera un algorithme offline et on réfléchira à sa
# transformation possible en algorithme online pour une implémentation mobile

# la classification wikipedia des algorithmes "offline" est
# top-down, bottom-up, sliding window, et global methods
# mais les fondements mathématiques me semblent mal définir la
# classification proposée

# on va chercher à utiliser plusieurs algorithmes de détection de signaux:

# 1. on utilisera une méthode fondée sur la détection de pics avec un treshold

# pourquoi? parceque les signaux de saut et de pas different en intensité.
# la méthode de détection de pics est une moyenne calculéé sur un intervale défini (smoothed z
# algorithm) ou un rééchantillonage filtré sur un treshold (bibliothèque detecta)
# et permet avec un treshold d'identifier des pics dans les signaux.
# Il suffit donc de mettre un treshold adapté pour identifier un pic d'une hauteur "typique"
#  du signal à identifier
#  (l'accélération dans un saut est un peu plus importante que l'accélération
#  lors de la marche)
# Smoothed Z Score Algorithm de stackoverflow:
# (Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". Stack Overflow.
# Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08))
# plusieurs bibliothèques sont comparées ici https://blog.ytotech.com/2015/11/01/findpeaks-in-python/


# 2. on utilisera ensuite une méthode fondée sur la transformée de fourrier, un treshold sur l'ampleur
# des pics de fréquences et éventuellement apres avoir appliqué un filtre coupe-haut sur le signal
# d'entrée
# pourquoi? parceque la tranformée de fourrier permet de représenter le signal en fonction de ses
# fréquences et non en fonction du temps.
# elle est souvent utilisée pour identifier des signaux periodiques.
# Dans notre cas c'est cependant un signal non périodique qu'on cherche à identifier pour le saut.
# Mais si les signaux de saut different en fréquence des signaux de marches ou de course, on pourra
# distinguer les deux. Le filtre coupe-haut sert à couper les hautes fréquences qui vont brouiller
# les résultats de la transformée de fourrier en représentant de nombreuses fréquences parasites.
# bibliothèque numpy.fft,

# 3. on utilisera enfin une méthode de convolution à partir d'une onde mère
# avec un algorithmes de détection de wavelet (ondelette)
# pourquoi? parceque la transformée en ondelette continue est souvent utilisée (cwt)
# pour identifier des signaux non périodiques ce qui est le cas pour les enregistrements de saut.
# la transformée en ondelettes (wavelet) permet de réprésenter un signal en fonction
# de sa "scale" (échelle) et du temps. L'échelle est difficile à comprendre mais peut intuitivement
# etre interprétée comme l'inverse de la fréquence.  Il faut utiliser une "ondelette mère" (mother wavelet)
# adaptée au signal à identifier, et un intervale de scale adapté également,
# car la méthode repose sur une convolution de cette ondelette mère sur l'onde "fille"
# du signal à identifier par "fenetre glissante" (sliding window) sur les données d'entrée.
# Il suffira ensuite de distinguer les signaux dont la "scale" est "typique" du signal à identifier
# bibliothèque pywt.cwt

################################################
# implementation dans une appli mobile:
################################################

# il sera nécessaire de traduire le code en swift ou kotlin.
# Les bibliothèques utilisées sont open source et
# peuvent donc être librement adaptées en kotlin ou swift moyennant un effort
# de traduction à évaluer.
# dans une logique de détection continue (algorithme online) on cherchera à
# découper le signal continue en signaux discrets de x secondes et on pourra
# lui appliquer l'algorithme offline. la période de x secondes sera à adapter à la fréquence
# des signaux qu'on cherche à détecter. Pour des sauts dont la fréquence est d'environ 1Hz,
# et la marche dont la fréquence est d'environ 1Hz à 4Hz une période de 2 secondes suffit
# mais il faudra réfléchir à optimiser cette période pour minimiser la probabilité
# de "couper" le signal tout en conservant une rapidité de détection convenable.

################################################
# stratégie d'amélioration continue:
################################################

# trivialement on pourrait améliorer le modèle avec les coordonnées gps en x et y (z étant l'altitude)
# pour détecter la marche et la course. Mais si on n'utilise pas les coordonnées gps:

# l'amélioration consisterait d'abbord à analyser et comprendre les faux positifs et faux négatifs

# Pour améliorer le modèle proposé, on pourra jouer sur les fréquences, les 'tresholds',
# les ondelettes mères pour la methode wavelet

# on pourrait aussi tester d'autres pistes en termes d'algorithmes:
# clustering
# cumsum # sources: http://www.wriley.com/PTTI_2008_Preprint_27.pdf
# stars algorithm with t-test # sources: http://www.wriley.com/PTTI_2008_Preprint_27.pdf
# analyse de densité spectrale
# méthode de Hilbert-Huang. # sources :https://hal.archives-ouvertes.fr/hal-01819670/document
# reduction de bruit avec algorithme # sources : loess https://ggbaker.ca/data-science/content/filtering.html


# on pourrait ensuite utiliser du machine learning supervisé à partir de dataset plus larges
# et analyser les faux positifs et faux négatifs.
# d'apres un test effectué en ayant mal lu la consigne, le modèle knn peu gourmand en ressource
# fonctionne assez bien par rapport à d'autres modèles plus gourmands en ressources.

# on pourrait imaginer un combo d'algorithmes de détection si certains modèles marches mieux
# sur des faux positifs ou faux négatifs "typiques"

# enfin, la rapidité des modèles pourra être évaluée et améliorée
# en première analyse, n étant le nombre de points
# la complexité de la méthode d'identification de pics varie en O(n)
# la complexité de la méthode fft varie en O(n log n),
# la complexité de la méthode wavelet varie en O(n)
# mais on utilise une "sliding window" tres gourmande en ressource

# ###############################################
# Résultats
# ###############################################
# dataset1:
# 3 enregistrements de courses de 30 s
# 4 enregistrements de marches de 30 s
# 4 enregistrements de 1 saut d'environ 10 s
# 1 enregistrement de 2 sauts d'environ 10 s
#
# résultats (matrice de détection, 1 pour saut détecté, 0 pour saut non détecté)
#
#        methode   dsp='peaks'      dsp='fft'       dsp='wavelet'
# saut1                 1               1               1
# saut2                 1               1               1
# saut3                 1               1               1
# saut4                 1               1               1
# 2sauts                1               1               1
# marche1               0               0               0
# marche2               0               0               0
# marche3               0               1               0
# marche4               0               0               0
# course1               1               0               0
# course2               1               0               0
# course3               1               0               0
# timeit                 0.093 s         0.140 s         48.064 s
#
# dataset2
# 13 enregistrements de saut d'environ 10 s
#
# la méthode peaks detecte les 5 sauts du premier dataset mais aucun du deuxieme dataset,
# ne détecte aucun saut pour les marches, et détecte des sauts pour les courses.
# cela s'explique car les pics des ondes des courses sont de meme ordre de grandeur
# que ceux des sauts et les pics des sauts du deuxieme dataset sont moins importants.
#
# la méthode fft détecte 5 sauts du premier dataset et 8 saut sur 13 du deuxieme dataset,
# détecte un saut pour une des marches, et ne détecte pas de saut pour les courses.
# la faible magnitude sur les resultats fft à 2Hz pour les sauts s'explique car
# il ne s'agit pas d'un signal périodique, on a donc calibré
# une détection de magnitude à 2Hz relativement faible et une détection des magnitudes
# à 3,4 et 5Hz plus importante pour détecter les marches et courses dont
# les signaux périodiques produisent d'importants pics en transformés de fourrier
#
# la méthode wavelet détecte tous les sauts des deux datasets et ne détecte pas de
# sauts pour les marches et les courses mais prend 500 fois plus de temps que
# les deux autres méthodes.

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import sys
import pywt
import detecta

############################################################
# Traitement du signal (Digital signal processing)
############################################################


def plot_signal(csvfile, dsp=0):
    """produit un graphique de l'enregistrement csvfile avec traitement 'fft','wavelet' ou 0 (rien)"""
    df = pd.read_csv(csvfile, sep=';')
    a = df['Absolute acceleration (m/s^2)']
    if dsp == 'fft':
        sample_rate = 100  # phyphox enregistre 100 points par seconde
        array_fft = np.fft.fft(a.values)
        mag = np.sqrt(array_fft.real ** 2 + array_fft.imag ** 2)
        mag = mag * 2 / len(array_fft)
        mag = mag[0 : int(len(mag) / 2)]
        mag[0] = 0
        freq = np.arange(0, len(mag), 1) * (sample_rate / len(array_fft))
        df = pd.DataFrame(data={'freq': freq, 'magnitude': mag})
        df['freq'] = df['freq'].round(decimals=0)
        df = df.groupby(['freq']).agg('sum')
        plt.figure()
        plt.xlim([0, 10])
        plt.ylim([0, max(df['magnitude'])])
        plt.bar(df.index.values, df['magnitude'].values)
        plt.title('FFT de l enregistrement {}'.format(str(csvfile)))
        plt.xlabel('f (Hz)', fontsize=16)  ###a revoir pour avoir la frequence
        plt.ylabel('magnitude', fontsize=16)
        plt.show()
    elif dsp == 'wavelet':
        mother_wavelet = 'gaus1'  # sym2 seems better
        # but sym2 not available in cwt pywt.wavelist(kind='continuous'))
        sampling_rate = 0.01  # 100Hz
        scale_for_2Hz_sym2_signal = (
            10  # pywt.scale2frequency('gaus1',10)/0.01==2.0
        )
        scales = np.arange(1, 101, 1)
        coeff, freq = pywt.cwt(a.values, scales, mother_wavelet)
        a = pd.DataFrame(data=coeff)
        plt.figure()
        sns.heatmap(
            a.loc[:, 0:700],
            cmap='coolwarm',
        )
        plt.show()
    else:
        plt.figure()
        plt.xlim([0, len(a)])
        plt.ylim([0, np.max(a)])
        plt.plot(a.index.values, a.values)
        plt.title(
            'Accélération absolue de l enregistrement {}'.format(str(csvfile))
        )
        plt.xlabel('time (centisecondes)', fontsize=16)
        plt.ylabel('acceleration (m/s^2)', fontsize=16)
        plt.show()


def jumps_detect_csv(csv_folder, dsp='peaks'):
    """affiche si les csv dans le repertoire csv_folder en entrée sont des sauts ou non
    csv_folder : repertoire contenant les fichiers csv à analyser
    dsp :
    'peaks' pour une analyse par pics calibrée sur une intensité d'accélération de 55 m/s^2
    'fft' pour une analyse par transformée de fourrier calibrée pour détecter une fréquence de 1Hz
    'wavelet' pour une analyse par transformée en ondelette calibrée pour détecter une scale
    de 12 à 24 de 100 centisecondes"""
    files_csv = os.listdir(csv_folder)
    files_csv = sorted([k for k in files_csv if '.csv' in k])
    result_positive = []
    result_negative = []
    for csv in files_csv:
        df = pd.read_csv(csv_folder + '/' + csv, sep=';')
        a = df['Absolute acceleration (m/s^2)']
        if dsp == 'peaks':
            result = detecta.detect_peaks(
                a.values, mph=55, mpd=50, threshold=0, show=False
            )
            if result.size == 0:
                print(
                    'methode peaks:aucun saut détecté pour {}'.format(str(csv))
                )
                result_negative.append(1)
            else:
                print('methode peaks:saut détecté pour {}'.format(str(csv)))
                result_positive.append(1)
        elif dsp == 'fft':
            sample_rate = 100  # phyphox enregistre 100 points par seconde
            array_fft = np.fft.fft(a.values)
            mag = np.sqrt(array_fft.real ** 2 + array_fft.imag ** 2)
            mag = mag * 2 / len(array_fft)
            mag = mag[0 : int(len(mag) / 2)]
            mag[0] = 0
            freq = np.arange(0, len(mag), 1) * (sample_rate / len(array_fft))
            df = pd.DataFrame(data={'freq': freq, 'magnitude': mag})
            df['freq'] = df['freq'].round(decimals=0)
            df = df.groupby(['freq']).agg('sum')
            if df.loc[2.0, 'magnitude'] >= 8 and (
                df.loc[3.0, 'magnitude'] <= 6
                and df.loc[4.0, 'magnitude'] <= 6
                and df.loc[5.0, 'magnitude'] <= 6
            ):
                # les sauts ont un pic important à 2 Hz et
                # la course et la marche ont des pics à 3Hz,4Hz et 5Hz
                # plus importants que pour le saut
                print('methode fft:saut détecté pour {}'.format(str(csv)))
                result_positive.append(1)
            else:
                print(
                    'methode fft:pas de saut détecté pour {}'.format(str(csv))
                )
                result_negative.append(1)
        else:  # ('wavelet')
            mother_wavelet = 'gaus1'  # sym2 is better from empirical point of view and litterature
            # sym2 not available in cwt pywt.wavelist(kind='continuous'))
            sampling_rate = 0.01  # 100Hz
            scale_for_2Hz_sym2_signal = (
                10  # pywt.scale2frequency('gaus1',10)/0.01==2.0
            )
            scales = np.arange(1, 101, 1)
            coeff, freq = pywt.cwt(a.values, scales, mother_wavelet)
            # pca = sklearn.decomposition.PCA(n_components=1)
            # coeff_pca = pca.fit_transform(coeff)
            # a = pd.Series(data=coeff_pca.flatten())
            a = pd.DataFrame(data=coeff)
            a = a.abs()
            sum_list = []
            for i in range(len(a.columns) - 100):
                # calibration duree signal 100 centisecondes
                sliding_column_list = [k for k in range(i, i + 100)]
                sum = (
                    a.loc[13:25, sliding_column_list].sum(axis=1).sum(axis=0)
                )  # calibration scale du signal 13:25
                sum_list.append(sum)
            if max(sum_list) > 35000:
                print('methode wavelet:saut détecté pour {}'.format(str(csv)))
                result_positive.append(1)
            else:
                print(
                    'methode wavelet:pas de saut détecté pour {}'.format(
                        str(csv)
                    )
                )
                result_negative.append(1)
    nb_saut_detecte = len(result_positive)
    nb_enregistrement = len(result_positive) + len(result_negative)
    print(
        'nombre de sauts détectés : {} sur {} enregistrements\n'.format(
            str(nb_saut_detecte), str(nb_enregistrement)
        )
    )


if __name__ == '__main__':

    if len(sys.argv) == 1:
        jumps_detect_csv('dataset1', 'peaks')
        jumps_detect_csv('dataset1', 'fft')
        jumps_detect_csv('dataset1', 'wavelet')
    else:
        jumps_detect_csv(sys.argv[1], 'peaks')
        jumps_detect_csv(sys.argv[1], 'fft')
        jumps_detect_csv(sys.argv[1], 'wavelet')
    # except:
    #     print (err)
    #     print('vous devez lancer le script en tapant python libertyrider.py path_to_folder_to_process')

    #### machine learning ####

    # ml_loop(0) or ml_loop('fft') or ml_loop('wavelet') for machine learning

    #### display ####

    # os.chdir('folder') folder dataset1 ou dataset2/jumps ou dataset2/random
    # files = os.listdir()
    # files = sorted(files)
    # for i in files:
    # df = pd.read_csv(i, sep=';')
    # a = df['Absolute acceleration (m/s^2)']
    # detecta.detect_peaks(a.values,mph=55, mpd=50, threshold=0, show=True)
    # plot_signal(i, 'dsp') 'peaks' ou 'wavelet' ou 'fft'

    # timeit

    # f="jumps_detect_csv('dataset2','wavelet')"
    # t1 = timeit.Timer(f, "from __main__ import jumps_detect_csv")
    # print("jump_detect ran:", t1.timeit(number=1), "s")


# ############################################################
# #machine learning (for information)
# ############################################################
# import sklearn.model_selection
# import sklearn.metrics
# import sklearn.decomposition
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier

# #########data utilisée :#########
#
# # 13 enregistrements de sauts
# # et 28 enregistrements de signaux d'accelerations divers (mouvement des bras, montée et descente escalier
# #montée sur une chaise, repos, passage position assis à debout et inversement, etc.
# # petit dataset -> machine learning plutot que deep learning
# # 1er essai de feature engineering : on va "couper" chaque enregistrement à la même
# # longueur (6s) et chaque "feature" sera un enregistrement d'accélération en z à la
# # date t.
#
# #pipeline
#
# def process_z_from_signal(csvfile,dsp=0):
#     "renvoie une serie de la composante z (avec process dsp ou non) à partir du csv du signal coupé à 6s"
#     dfraw=pd.read_csv(csvfile,sep=';')
#     dfraw=dfraw.sort_values('Time (s)', axis=0)
#     if dsp=='fft':
#         dfraw = dfraw[:600]
#         array_raw = dfraw['Acceleration z (m/s^2)'].values
#         array_fft = np.fft.fft(array_raw)
#         mag=np.sqrt(array_fft.real**2+array_fft.imag**2)
#         mag=mag*2/len(array_raw)
#         s=pd.Series(data=mag)
#     elif dsp=='wavelet':
#         dfraw = dfraw[:600]
#         array_raw = dfraw['Acceleration z (m/s^2)'].values
#         mother_wavelet='gaus1'#sym2 is better from empirical point of view and litterature
#         #sym2 not available in cwt pywt.wavelist(kind='continuous'))
#         sampling_rate=0.01#100Hz
#         scale_for_2Hz_sym2_signal=10 #pywt.scale2frequency('gaus1',10)/0.01==2.0
#         scales=np.arange(1,101,1)
#         coeff, freq = pywt.cwt(array_raw, scales, mother_wavelet)
#         pca = sklearn.decomposition.PCA(n_components=1)
#         coeff_pca = pca.fit_transform(coeff)
#         s=pd.Series(data=coeff_pca.flatten())
#     else:
#         dfraw = dfraw[:600]
#         s=dfraw['Acceleration z (m/s^2)']
#     return s
#
# def build_dataset(folder_saut,folder_divers,dsp):
#     "renvoie un dataframe à partir des csv dans folder_saut et folder_divers avec dsp ou non"
#     os.chdir(folder_saut)
#     files_saut = os.listdir()
#     files_saut=sorted(files_saut)
#     for i in files_saut:
#         if i=='Raw Data.csv':
#             series_0=process_z_from_signal(i,dsp)
#             series_0.name='saut_0'
#         elif i=='Raw Data1.csv':
#             series_1 = process_z_from_signal(i,dsp)
#             series_1.name = 'saut_1'
#             df_0= pd.concat([series_0,series_1], axis=1)
#         else:
#             s_line_i=process_z_from_signal(i,dsp)
#             s_line_i.name='saut_'+str(files_saut.index(i))
#             df_0=pd.merge(df_0,s_line_i,left_index=True,right_index=True)
#     df_0.loc['saut']=1
#     os.chdir(folder_divers)
#     files_divers = os.listdir()
#     files_divers = sorted(files_divers)
#     for i in files_divers:
#         s_line_i = process_z_from_signal(i,dsp)
#         s_line_i.name = 'divers_' + str(files_divers.index(i))
#         s_line_i.loc['saut'] = 0
#         df_0=pd.merge(df_0,s_line_i,right_index=True,left_index=True)
#     return df_0.T


###machine learning

# dico_classifier={'knn':KNeighborsClassifier,
#           'naiveb':GaussianNB,'randomforest':RandomForestClassifier,
#           'gtree':GradientBoostingClassifier,'neural':MLPClassifier}
#
# def ml(jumpfolder,randomfolder,classifier,dsp=0):
#     "lance le machine learning classifier sur les data dans jumpfolder et randomfolder avec le dsp ou non"
#     # dataset
#
#     df = build_dataset(jumpfolder, randomfolder,dsp)
#     X = df[[i for i in list(df.columns) if i != 'saut']]
#     y = df['saut']
#     # classifier random forest model training
#     clf = dico_classifier[classifier]()
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
#     clf.fit(X_train, y_train)
#
#     # predictions
#     forest_predicted = clf.predict(X_test)
#
#     # scores
#     _accuracy=sklearn.metrics.accuracy_score(y_test, forest_predicted)
#     _precision=sklearn.metrics.precision_score(y_test, forest_predicted)
#     _recall=sklearn.metrics.recall_score(y_test, forest_predicted)
#     print(str(classifier)+' Accuracy: {:.2f}'.format(_accuracy))
#     print(str(classifier)+' Precision: {:.2f}'.format(_precision))
#     print(str(classifier)+' Recall: {:.2f}'.format(_recall))
#
#     # confusion matrix
#     confusion_clf = sklearn.metrics.confusion_matrix(y_test, forest_predicted)
#     df_clf = pd.DataFrame(confusion_clf,
#                           index=[i for i in range(0, 2)], columns=[i for i in range(0, 2)])
#
#     plt.figure(figsize=(5.5, 4))
#     sns.heatmap(df_clf, annot=True, vmin=0, vmax=11, cmap="Blues")
#     plt.title(str(classifier)+' \nAccuracy:{0:.3f}'.format(_accuracy))
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     # courbes precision-recall
#     y_score_clf = clf.predict_proba(X_test)
#     y_score_df = pd.DataFrame(data=y_score_clf)
#
#     precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_score_df[1])
#     closest_zero = np.argmin(np.abs(thresholds))
#     closest_zero_p = precision[closest_zero]
#     closest_zero_r = recall[closest_zero]
#
#     plt.figure()
#     plt.xlim([0.0, 1.01])
#     plt.ylim([0.0, 1.01])
#     plt.plot(precision, recall)
#     plt.title(str(classifier)+' Precision-Recall Curve \nprecision :{:0.2f}'.format(_precision)+' recall: {:0.2f}'.format(_recall))
#     plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
#     plt.xlabel('Precision', fontsize=16)
#     plt.ylabel('Recall', fontsize=16)
#     plt.show()
#
#     # courbes roc
#     y_score_clf = clf.predict_proba(X_test)
#     y_score_df = pd.DataFrame(data=y_score_clf)
#     fpr_clf, tpr_clf, _ = sklearn.metrics.roc_curve(y_test, y_score_df[1])
#     roc_auc_clf = sklearn.metrics.auc(fpr_clf, tpr_clf)
#
#     plt.figure()
#     plt.xlim([-0.01, 1.00])
#     plt.ylim([-0.01, 1.01])
#     plt.plot(fpr_clf, tpr_clf, lw=3, label=str(classifier)+' ROC curve (area = {:0.2f})'.format(roc_auc_clf))
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.title('ROC curve '+str(classifier)+' \nAUC:{0:.3f}'.format(roc_auc_clf), fontsize=16)
#     plt.legend(loc='lower right', fontsize=13)
#     plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
#     plt.show()
#
#     return pd.DataFrame(data=(_accuracy,_precision,_recall,roc_auc_clf),index=['accuracy','precision','recall','AUC'],columns=[classifier])
#
# def ml_loop(dsp=0):
#     path = os.getcwd()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--jumps",
#                         default=path + 'dataset2/jumps',
#                         help="folder with jumps signal csv",
#                         dest='jumps_folder',
#                         type=str)
#     parser.add_argument("--random",
#                         default=path + 'dataset2/random',
#                         help="folder with random signal csv",
#                         dest='rand_folder',
#                         type=str)
#     args = parser.parse_args()
#     jumpfolder = args.jumps_folder
#     randomfolder = args.rand_folder
#
#     df = pd.DataFrame(data=(0, 0, 0, 0), columns=['init'], index=['accuracy', 'precision', 'recall', 'AUC'])
#     for clf in dico_classifier:
#         print(clf)
#         result_ml = ml(jumpfolder, randomfolder, clf,dsp)
#         df = pd.merge(df, result_ml, right_index=True, left_index=True)
#     df = df.drop('init', axis=1)
#     print(df)
#     plt.figure()
#     sns.heatmap(df, annot=True, vmin=0, vmax=1, cmap="Blues")
#     plt.title('scores des classifiers - dsp:'+str(dsp))
#     plt.ylabel('scores')
#     plt.xlabel('modeles')
#     plt.show()
