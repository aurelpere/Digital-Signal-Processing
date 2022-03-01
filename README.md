# Digital-Signal-Processing
little jumps signals identification from smartphone accelerator (phyphox datasets)


This is final version (1/03/22)

usage: libertyrider.py folder_to_process
if folder is not specified libertyrider.py will process 'dataset1' folder


################################################
#Résultats
################################################
#dataset1:
# 3 enregistrements de courses de 30 s
# 4 enregistrements de marches de 30 s
# 4 enregistrements de 1 saut d'environ 10 s
# 1 enregistrement de 2 sauts d'environ 10 s

# résultats (matrice de détection, 1 pour saut détecté, 0 pour saut non détecté)

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
#timeit                 0.093 s         0.140 s         48.064 s

#dataset2
#13 enregistrements de saut d'environ 10 s

# la méthode peaks detecte les 5 sauts du premier dataset mais aucun du deuxieme dataset,
# ne détecte aucun saut pour les marches, et détecte des sauts pour les courses.
# cela s'explique car les pics des ondes des courses sont de meme ordre de grandeur
# que ceux des sauts et les pics des sauts du deuxieme dataset sont moins importants.

# la méthode fft détecte 5 sauts du premier dataset et 8 saut sur 13 du deuxieme dataset,
# détecte un saut pour une des marches, et ne détecte pas de saut pour les courses.
# la faible magnitude sur les resultats fft à 2Hz pour les sauts s'explique car
# il ne s'agit pas d'un signal périodique, on a donc calibré
# une détection de magnitude à 2Hz relativement faible et une détection des magnitudes
# à 3,4 et 5Hz plus importante pour détecter les marches et courses dont
# les signaux périodiques produisent d'importants pics en transformés de fourrier

#la méthode wavelet détecte tous les sauts des deux datasets et ne détecte pas de
# sauts pour les marches et les courses mais prend 500 fois plus de temps que
# les deux autres méthodes.


Resultats machine learning 26/02 (pour info): 


<img src=26.02_synthese_sans_dsp.png>

<img src=26.02_synthese_fft.png>

<img src=26.02_synthese_wavelet.png> 
