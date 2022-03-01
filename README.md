# Digital-Signal-Processing<br>
little jumps signals identification from smartphone accelerator (phyphox datasets)<br>
<br>
<br>
This is final version (1/03/22)<br>
<br>
usage: libertyrider.py folder_to_process<br>
if folder is not specified libertyrider.py will process 'dataset1' folder<br>
<br>
<br>
###############################################<br>
Résultats<br>
###############################################<br>
dataset1:<br>
3 enregistrements de courses de 30 s<br>
4 enregistrements de marches de 30 s<br>
4 enregistrements de 1 saut d'environ 10 s<br>
1 enregistrement de 2 sauts d'environ 10 s<br>
<br>
résultats (matrice de détection, 1 pour saut détecté, 0 pour saut non détecté)<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     methode&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   dsp='peaks'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      dsp='fft'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       dsp='wavelet'<br>
saut1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               1      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
saut2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                 1     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          1   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            1<br>
saut3&#9&#9                 1               1               1<br>
saut4&#9&#9                 1               1               1<br>
2sauts&#9&#9                1               1               1<br>
marche1&#9&#9               0               0               0<br>
marche2&#9&#9               0               0               0<br>
marche3               0               1               0<br>
marche4               0               0               0<br>
course1               1               0               0<br>
course2               1               0               0<br>
course3               1               0               0<br>
timeit                 0.093 s         0.140 s         48.064 s<br>
<br>
dataset2<br>
13 enregistrements de saut d'environ 10 s<br>
<br>
la méthode peaks detecte les 5 sauts du premier dataset mais aucun du deuxieme dataset,<br>
ne détecte aucun saut pour les marches, et détecte des sauts pour les courses.<br>
cela s'explique car les pics des ondes des courses sont de meme ordre de grandeur<br>
que ceux des sauts et les pics des sauts du deuxieme dataset sont moins importants.<br>

la méthode fft détecte 5 sauts du premier dataset et 8 saut sur 13 du deuxieme dataset,<br>
détecte un saut pour une des marches, et ne détecte pas de saut pour les courses.<br>
la faible magnitude sur les resultats fft à 2Hz pour les sauts s'explique car<br>
il ne s'agit pas d'un signal périodique, on a donc calibré<br>
une détection de magnitude à 2Hz relativement faible et une détection des magnitudes<br>
à 3,4 et 5Hz plus importante pour détecter les marches et courses dont<br>
les signaux périodiques produisent d'importants pics en transformés de fourrier<br>
<br>
la méthode wavelet détecte tous les sauts des deux datasets et ne détecte pas de<br>
sauts pour les marches et les courses mais prend 500 fois plus de temps que<br>
les deux autres méthodes.<br>
<br>
<br>
Resultats machine learning 26/02 (pour info): <br>


<img src=26.02_synthese_sans_dsp.png>

<img src=26.02_synthese_fft.png>

<img src=26.02_synthese_wavelet.png>
