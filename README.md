# Digital-Signal-Processing
little jumps signals identification from smartphone accelerator


This is beta version with machine learning random forest classifier.
A version with dsp algorithm will be updated later.

usage: libertyrider.py [-h] [--jumps JUMPS_FOLDER] [--random RAND_FOLDER]

options:
  -h, --help            show this help message and exit
  --jumps JUMPS_FOLDER  folder with jumps signal csv
  --random RAND_FOLDER  folder with random signal csv
  
  By default, if jumps folder and random folder are not specified, libertyrider.py searches jumps folder and random folder in the current working directory

Resultats provisoires 26/02 : 
<img src=26.02_synthese_sans_dsp.png>

<img src=26.02_synthese_fft.png>

<img src=26.02_synthese_wavelet.png> 
