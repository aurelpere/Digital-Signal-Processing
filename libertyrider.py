#!/usr/bin/python3
# coding: utf-8


#25.02.22.10h
#analyse du problème: reponse par machine learning ou par dsp
#dsp : un apprentissage m'est nécessaire

########################################
#premiere solution en machine learning sans dsp
########################################

###data utilisée :
# 13 enregistrements de sauts
# et 28 enregistrements de signaux d'accelerations divers
# petit dataset -> machine learning plutot que deep learning
# 1er essai de feature engineering : on va "couper" chaque enregistrement à la même
# longueur (6s) et chaque "feature" sera un enregistrement d'accélération en z à la
# date t.
import pandas as pd
import os
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse



def get_z_from_signal(csvfile):
    dfraw=pd.read_csv(csvfile,sep=';')
    dfraw=dfraw.sort_values('Time (s)', axis=0)
    dfraw = dfraw[:600]
    dfraw=dfraw['Acceleration z (m/s^2)']
    return dfraw

def build_dataset(folder_saut,folder_divers):
    os.chdir(folder_saut)
    files_saut = os.listdir()
    files_saut=sorted(files_saut)
    for i in files_saut:
        if i=='Raw Data.csv':
            series_0=get_z_from_signal(i)
            series_0.name='saut_0'
        elif i=='Raw Data1.csv':
            series_1 = get_z_from_signal(i)
            series_1.name = 'saut_1'
            df_0= pd.concat([series_0,series_1], axis=1)
        else:
            df_line_i=get_z_from_signal(i)
            df_line_i.name='saut_'+str(files_saut.index(i))
            df_0=pd.merge(df_0,df_line_i,left_index=True,right_index=True)
    df_0.loc['saut']=1
    os.chdir(folder_divers)
    files_divers = os.listdir()
    files_divers = sorted(files_divers)
    for i in files_divers:
        df_line_i = get_z_from_signal(i)
        df_line_i.name = 'divers_' + str(files_divers.index(i))
        df_line_i.loc['saut'] = 0
        df_0=pd.merge(df_0,df_line_i,right_index=True,left_index=True)
    return df_0.T

#25.02.22.11h experimentations pour créer le dataset
#25.02.22.12h30 pause repas
#25.02.22.13h30 verification de l'intégrité du dataset

###machine learning

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

    #dataset
    df = build_dataset(jumpfolder,randomfolder)
    X = df[[i for i in list(df.columns) if i != 'saut']]
    y = df['saut']

    #classifier random forest model training
    clf = sklearn.ensemble.RandomForestClassifier()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)

    #predictions
    forest_predicted = clf.predict(X_test)

    # scores
    print('Accuracy: {:.2f}'.format(sklearn.metrics.accuracy_score(y_test, forest_predicted)))
    print('Precision: {:.2f}'.format(sklearn.metrics.precision_score(y_test, forest_predicted)))
    print('Recall: {:.2f}'.format(sklearn.metrics.recall_score(y_test, forest_predicted)))


    #confusion matrix
    confusion_clf = sklearn.metrics.confusion_matrix(y_test, forest_predicted)
    df_clf = pd.DataFrame(confusion_clf,
                         index=[i for i in range(0, 2)], columns=[i for i in range(0, 2)])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(df_clf, annot=True,vmin=0, vmax=11,cmap="Blues")
    plt.title('Random forest \nAccuracy:{0:.3f}'.format(sklearn.metrics.accuracy_score(y_test, forest_predicted)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xlabel('Predicted label');

    #courbes precision-recall
    y_score_clf = clf.predict_proba(X_test)
    y_score_df = pd.DataFrame(data=y_score_clf)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_score_df[1])
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize=12, fillstyle='none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.show()

    #courbes roc
    y_score_clf = clf.predict_proba(X_test)
    y_score_df = pd.DataFrame(data=y_score_clf)
    fpr_clf, tpr_clf, _ = sklearn.metrics.roc_curve(y_test, y_score_df[1])
    roc_auc_clf = sklearn.metrics.auc(fpr_clf, tpr_clf)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_clf, tpr_clf, lw=3, label='RandomForest ROC curve (area = {:0.2f})'.format(roc_auc_clf))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (RandomForest Classifier)\nAUC:{0:.3f}'.format(roc_auc_clf), fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.show()


