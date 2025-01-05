# STAGE 

### Organisation

```asciidoc
main  
├── segmentation_parcellaire\
│   ├── scripts\
│   │   ├── yaml\
│   │   │   ├── yolo_sat.yaml
│   │   ├── Dataset.ipynb
│   │   ├── Production.ipynb
│   │   ├── Results.ipynb
│   │   ├── Shapefiles_filtrage.ipynb
│   │   ├── Training.ipynb
│   ├── Documentation.md
├── crop_classification\
│   ├── scripts\
│   │   ├── Dataset\
│   │   │   ├── Dataset.py
│   │   │   ├── Dataset_all_bands.py
│   │   │   ├── Homogene_Dataset_all_bands.py
│   │   ├── Test\
│   │   │   ├── All_test.ipynb
│   │   │   ├── LSTM.ipynb
│   │   │   ├── Test Fullbands.ipynb
│   │   ├── Prod_Multicrop.ipynb
│   │   ├── Prod_Prairies.ipynb
│   │   ├── Train_Multicrop.ipynb
│   │   ├── preprocessing.py
│   │   ├── gee_data.py
│   ├── Documentation.md
├── biomass\
│   ├── scripts\
│   │   ├── Explorer.ipynb
│   ├── Documentation.md
├── streamlit\
│   ├── scripts\
│   │   ├── Classifier.py
│   │   ├── Contouring.py
│   │   ├── main.py
├── api\
│   ├── scripts\
│   │   ├── Classifier.py
│   │   ├── Contouring.py
│   │   ├── main.py
│   ├── Dockerfile
├── models\
│   ├── segmentation.pt
│   ├── multicrop_classification.h5
│   ├── RandomForestPrairie.joblib
├── shapefile\
│   ├── Parcelle.cpg
│   ├── Parcelle.dbf
│   ├── Parcelle.shp
│   ├── Parcelle.shx
├── requirements.txt
```