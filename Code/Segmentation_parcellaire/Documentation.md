# Segmentation parcellaire 

##### Shapefile_filter 

Pour la création des datasets, nous utilisons les shapefiles venant du [RPG](https://geoservices.ign.fr/rpg). Les shapefiles contiennent un grand nombre de parcelles que nous ne souhaitons pas utiliser. En utilisant ce script, cela va créer de nouveaux shapefiles sans les parcelles "gelé" et "divers". Nous aurions pu effectuer le filtre lors de la sélection de la parcelle. Mais ce script a pour but d’alléger les shapefiles que nous utilisons. 

##### Dataset

Nous prenons l'entièreté des shapefiles présent dans le fichier spécifié. Pour chaque région, nous allons calculer le nombre d'images que nous souhaitons récupérer. Ce nombre d'images dépend du nombre d'image total et du poids du fichier par rapport aux autres. De manière itérative, nous tirons une parcelle aléatoire dans le shapefile ouvert. Nous prenons le centre de cette parcelle et nous l'utilisons pour récupérer l'image satellite avec l'API de Mapbox. Nous découpons également le shapefile de la même taille que l'image satellite. Pour finir nous transformons le shapefile découpé en fichier texte. 

##### Training

Pour l'entraînement, nous reprenons les données créées avec le fichier "Dataset". Les données nécessitent d'être transférées sur les [GPU](https://jupyter.itkweb.fr/tree/StageLeandre/YOLO/data) directement. Une fois les données au bon endroit, nous pouvons lancer l'entraînement de YOLO. Nous pouvons retrouver l'ensemble des entraînements [ici](https://jupyter.itkweb.fr/tree/StageLeandre/YOLO/scripts/runs/segment). Pour chaque entraînement, nous pouvons retrouver les poids des modèles entraîné ainsi que des informations complémentaires sur l'entraînement (courbe historique...).

##### Results

Une fois l'entraînement effectué, nous pouvons faire une petite analyse sur l'entrainement ainsi que les résultats sur les données de test. Cela peut nous donner les informations suivantes : 
* Présence d'over fitting 
* Précision, Recall, mPA50
* IoU

##### Production

Après l'entraînement, nous pouvons également directement tester le modèle en entrant ses propres coordonnées GPS.