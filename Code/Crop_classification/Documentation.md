# Crop classification 

### Dataset

##### Dataset_all_bands

Fichier Python qui permet de récupérer l'ensemble des bandes spectrales pour une période données. Nous utilisons en donnée d'entrée les fichiers RPG. Toutes les données sont ensuite enregistré dans un fichier json 

##### Dataset

Fichier Python qui permet de récupérer uniquement le NDVI et EVI pour une période données. Nous utilisons en donnée d'entrée les fichiers RPG.
Toutes les données sont ensuite enregistré dans un fichier json 

##### Homogene_Dataset_all_bands

Fichier Python qui permet de récupérer l'ensemble des bandes spectrales pour une période données. Nous utilisons en donnée d'entrée les fichiers RPG. 
La différence avec le premier fichier et que ici nous faisons en sorte de récupérer le même nombre de culture pour chaque group de culture
Toutes les données sont ensuite enregistré dans un fichier json 

### Test

##### All_test

Ce notebook regroupe l'ensemble des tests réaliser sur la classification de crop de manière général

##### LSTM

Ce notebook regroupe l'ensemble des tests réaliser sur les LSTM

##### Test fullbands

Ce notebook regroupe l'ensemble des tests réaliser sur la classification de crop en utilisant toutes les bandes spectrales

##### preprocessing

Ce fichier est un mini module qui regroupe toutes les fonctions de préprocessing des données lorsque nous les récupérons depuis le fichier json

##### gee_data 

Ce fichier est un mini module qui regroupe toutes les fonctions permettant la récupération de bandes spectral

##### Prod_multicrop 

Fichier de production de la classification Multicrop

##### Train multicrop 

Fichier de training de la classification Multicrop

##### Prod Prairies

Fichier de production de la classification Prairies

##### Train Prairies 

Fichier de training de la classification Prairies