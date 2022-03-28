[![Test-Lint-Format](https://github.com/aurelpere/Digital-Signal-Processing/actions/workflows/main.yml/badge.svg)](https://github.com/aurelpere/Digital-Signal-Processing/actions/workflows/main.yml) ![test-coverage badge](./coverage-badge.svg) [![Maintainability](https://api.codeclimate.com/v1/badges/f7fef0519a3f8312dd6e/maintainability)](https://codeclimate.com/github/aurelpere/Digital-Signal-Processing/maintainability)

# Digital-Signal-Processing<br>
little jumps signals identification from smartphone accelerator (phyphox datasets) : hire process test of libertyrider<br>
<br>

## usage: 

### machine learning
`python3 machine_learning_.py --jumps jumpfolder --random randomfolder --dsp dsp_method`
>will process machine learning on jumpfolder and randomfolder with dsp_method (0,fft or wavelet)
<br>

\
`python3 machine_learning.py` 
>will process machine learning on dataset2/jumps and dataset2/random without dsp 
<br>

### signal processing
`python3 signal_processing.py`
>will process jump detections with peaks, fft and wavelet methods on dataset1 folder
<br>

\
`python3 signal_processing.py folder_to_process`
>will process jump detections with peaks, fft and wavelet methods on folder_to_process
<br>


## signal processing results
dataset1:<br>
3 enregistrements de courses de 30 s<br>
4 enregistrements de marches de 30 s<br>
4 enregistrements de 1 saut d'environ 10 s<br>
1 enregistrement de 2 sauts d'environ 10 s<br>
<br>
résultats (matrice de détection, 1 pour saut détecté, 0 pour saut non détecté)<br>
<br>
<img src=resultats.png>
<br>
dataset2<br>
13 enregistrements de saut d'environ 10 s<br>
<br>
la méthode peaks detecte les 5 sauts du premier dataset aucun sauts du deuxieme dataset,<br>
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
sauts pour les marches et les courses mais prend 30 fois plus de temps que<br>
les deux autres méthodes.<br>
<br>
<br>

## machine learning results
Resultats machine learning 26/02 (pour info): <br>


<img src=26.02_synthese_sans_dsp.png>

<img src=26.02_synthese_fft.png>

<img src=26.02_synthese_wavelet.png>


## methodology

25.02.22.10h
analyse du problème: reponse par machine learning et/ou par dsp
pour le dsp,un apprentissage m'est nécessaire<br>
25.02.22.11h experimentations pour créer le dataset<br>
25.02.22.13h30 verification de l'intégrité du dataset<br>
26.02.22 10h optimisation du machine learning pour experimenter sur plusieurs classifiers<br>
26.02.22 13h30 digestion de la litterature sur cwt et fft<br>
26.02.22 16h implementation des wavelet et fft dans les fonctions de pipeline et de ML<br>
26.02.22 18h test des trois modèles de machine learning<br>
28.02.22 11h relecture de la consigne : pas de machine learning, rédaction de la méthode dsp<br>
28.02.22 15h création du dataset de sauts, de marche et de course<br>
28.02.22 15h verification des données et complément sur la rédaction de la méthode dsp<br>
28.02.22 17h traitement par dsp fft et wavelet et analyse graphique des transformées obtenues<br>
28.02.22 18h implementation python de la méthode pics avec stackoverflow<br>
29.02.22 10h resultats decevants pour la methode pics de stackoverflow. implementation de detecta<br>
29.02.22 11h implementation fonction csv avec methode detecta<br>
29.02.22 12h implementation fonction csv avec methode fft<br>
29.02.22 15h implemetation fonction csv avec methode wavelet<br>
29.02.22 17h réalisation des mesures timeit<br>
29.02.22 18h terminé<br>
03.03.22 14h amélioration de l'algorithme  dsp 'wavelet' en augmentant le pas de la sliding windows<br>
de 1 à 10. L'algorithme prend 20 fois moins de temps (on passe de timeit 50 s à 3 s)<br>


la différence entre un signal de saut sur place et un signal de marche ou de course
étant trivialement l'accélération selon x et y dans un repère carthésien où x,y sont
horizontaux et z vertical, on pourrait procéder en distinguant la présence de données
positives dans les données en x et y.
mais le repere du téléphone est en principe celui dispo ici : http://developer.android.com/images/axis_device.png
Compte tenu de l'orientation du téléphone dans ma poche, ce serait 'y' qui serait
a priori l'axe le plus pertinent.
L'exercice se complique a priori si on ne connait pas l'orientation du téléphone mais
les enregistrements de phyphox montrent une présence du signal de saut, de marche et de course
sur les 3 axes. La réduction à un seul axe ne devrait donc en principe pas poser de problèmes.
Cependant, une petite analyse des signaux montre que
l'intensité des signaux en 'y'  varie du simple au quadruple selon les sauts
Elle est plus importante en 'y' pour les sauts que pour la marche (environ le double)
Elle est d'un ordre de grandeur similaire pour la course
L'orientation du téléphone dans la poche n'était probablement pas identique selon chaque
enregistrement.
Les données me semblent plus fiables lorsqu'on prend la composante
'Absolute acceleration' qui semble correspondre à la norme du vecteur acceleration
l'intensité des sauts est sensiblement la même pour chaque saut
l'intensité est 50% à 100% plus importante pour les sauts que pour la marche

On se place dans la perspective d'extrapoler pour détecter un signal de chute de moto, donc on va
axer la méthode sur les algorithmes de "traitement du signal" (digital signal processing)

d'apres l'article "step detection" de wikipedia, un algorithme de type "online"
serait plus adapté à la détection de signaux sur une appli mobile puisque le signal
serait analysé quand il arrive
Dans notre cas, on utilisera un algorithme offline et on réfléchira à sa
transformation possible en algorithme online pour une implémentation mobile

la classification wikipedia des algorithmes "offline" est
top-down, bottom-up, sliding window, et global methods
mais les fondements mathématiques me semblent mal définir la
classification proposée

on va chercher à utiliser plusieurs algorithmes de détection de signaux:

1. on utilisera une méthode fondée sur la détection de pics avec un treshold

pourquoi? parceque les signaux de saut et de pas different en intensité.
la méthode de détection de pics est une moyenne calculéé sur un intervale défini (smoothed z
algorithm) ou un rééchantillonage filtré sur un treshold (bibliothèque detecta)
et permet avec un treshold d'identifier des pics dans les signaux.
Il suffit donc de mettre un treshold adapté pour identifier un pic d'une hauteur "typique"
 du signal à identifier
 (l'accélération dans un saut est un peu plus importante que l'accélération
 lors de la marche)
Smoothed Z Score Algorithm de stackoverflow:
(Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". Stack Overflow.
Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08))
plusieurs bibliothèques sont comparées ici https://blog.ytotech.com/2015/11/01/findpeaks-in-python/


2. on utilisera ensuite une méthode fondée sur la transformée de fourrier, un treshold sur l'ampleur
des pics de fréquences et éventuellement apres avoir appliqué un filtre coupe-haut sur le signal
d'entrée

pourquoi? parceque la tranformée de fourrier permet de représenter le signal en fonction de ses
fréquences et non en fonction du temps.
elle est souvent utilisée pour identifier des signaux periodiques.
Dans notre cas c'est cependant un signal non périodique qu'on cherche à identifier pour le saut.
Mais si les signaux de saut different en fréquence des signaux de marches ou de course, on pourra
distinguer les deux. Le filtre coupe-haut sert à couper les hautes fréquences qui vont brouiller
les résultats de la transformée de fourrier en représentant de nombreuses fréquences parasites.
bibliothèque numpy.fft,

3. on utilisera enfin une méthode de convolution à partir d'une onde mère
avec un algorithmes de détection de wavelet (ondelette)

pourquoi? parceque la transformée en ondelette continue est souvent utilisée (cwt)
pour identifier des signaux non périodiques ce qui est le cas pour les enregistrements de saut.
la transformée en ondelettes (wavelet) permet de réprésenter un signal en fonction
de sa "scale" (échelle) et du temps. L'échelle est difficile à comprendre mais peut intuitivement
etre interprétée comme l'inverse de la fréquence.  Il faut utiliser une "ondelette mère" (mother wavelet)
adaptée au signal à identifier, et un intervale de scale adapté également,
car la méthode repose sur une convolution de cette ondelette mère sur l'onde "fille"
du signal à identifier par "fenetre glissante" (sliding window) sur les données d'entrée.
Il suffira ensuite de distinguer les signaux dont la "scale" est "typique" du signal à identifier
bibliothèque pywt.cwt


## mobile app implementation

il sera nécessaire de traduire le code en swift ou kotlin.
Les bibliothèques utilisées sont open source et
peuvent donc être librement adaptées en kotlin ou swift moyennant un effort
de traduction à évaluer.
dans une logique de détection continue (algorithme online) on cherchera à
découper le signal continue en signaux discrets de x secondes et on pourra
lui appliquer l'algorithme offline. la période de x secondes sera à adapter à la fréquence
des signaux qu'on cherche à détecter. Pour des sauts dont la fréquence est d'environ 1Hz,
et la marche dont la fréquence est d'environ 1Hz à 4Hz une période de 2 secondes suffit
mais il faudra réfléchir à optimiser cette période pour minimiser la probabilité
de "couper" le signal tout en conservant une rapidité de détection convenable.


## continuous improvement strategy

trivialement on pourrait améliorer le modèle avec les coordonnées gps en x et y (z étant l'altitude)
pour détecter la marche et la course. Mais si on n'utilise pas les coordonnées gps:

l'amélioration consisterait d'abbord à analyser et comprendre les faux positifs et faux négatifs

Pour améliorer le modèle proposé, on pourra jouer sur les fréquences, les 'tresholds',
les ondelettes mères pour la methode wavelet

on pourrait aussi tester d'autres pistes en termes d'algorithmes:
clustering
cumsum # sources: http://www.wriley.com/PTTI_2008_Preprint_27.pdf
stars algorithm with t-test # sources: http://www.wriley.com/PTTI_2008_Preprint_27.pdf
analyse de densité spectrale
méthode de Hilbert-Huang. # sources :https://hal.archives-ouvertes.fr/hal-01819670/document
reduction de bruit avec algorithme # sources : loess https://ggbaker.ca/data-science/content/filtering.html


on pourrait ensuite utiliser du machine learning supervisé à partir de dataset plus larges
et analyser les faux positifs et faux négatifs.
d'apres un test effectué en ayant mal lu la consigne, le modèle knn peu gourmand en ressource
fonctionne assez bien par rapport à d'autres modèles plus gourmands en ressources.

on pourrait imaginer un combo d'algorithmes de détection si certains modèles marches mieux
sur des faux positifs ou faux négatifs "typiques"

enfin, la rapidité des modèles pourra être évaluée et améliorée
en première analyse, n étant le nombre de points
la complexité de la méthode d'identification de pics varie en O(n)
la complexité de la méthode fft varie en O(n log n),
la complexité de la méthode wavelet varie en O(n)
mais on utilise une "sliding window" tres gourmande en ressource
pour interpréter les résultats

