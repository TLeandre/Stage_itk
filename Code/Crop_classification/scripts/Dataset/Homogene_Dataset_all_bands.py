import ee

import geopandas as gpd
import pandas as pd
import os 
import numpy as np
import random
import json

from typing import Union, Tuple 

from joblib import Parallel, delayed

def get_filenames(directory: str) -> np.ndarray:
    """retrieves shapefile name and size 

    Args:
        directory (str): directory path 

    Returns:
        np.ndarray: shapefile name and size
    """
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.shp'):
            filenames.append(filename)
         
    return filenames

def get_parcelles_dispo(ref_cultures: pd.core.frame.DataFrame, shapefile_data: gpd.geodataframe.GeoDataFrame) -> pd.core.frame.DataFrame:
    """Get all parcelles available

    Args:
        ref_cultures (pd.core.frame.DataFrame): code cultures
        shapefile_data (gpd.geodataframe.GeoDataFrame): group of parcels 

    Returns:
        pd.core.frame.DataFrame: parcelles available
    """
    parcelles_dispo = pd.DataFrame(columns=["CODE_GROUPE_CULTURE", "Nombre"])

    # Parcourir les lignes des deux premiers tableaux
    for i, row1 in shapefile_data.iterrows():
        code_group = int(row1["CODE_GROUPE_CULTURE"])
        nombre1 = row1["Nombre"]
        
        row2 = ref_cultures.loc[ref_cultures["CODE_GROUPE_CULTURE"] == code_group]
        nombre2 = row2["Nombre"].iloc[0]
        
        code_culture = row2["CODE_GROUPE_CULTURE"].iloc[0]
        nombre_dispo = min(nombre1, nombre2)
        tabtemp = pd.DataFrame({"CODE_GROUPE_CULTURE":  [code_culture], "Nombre": [nombre_dispo]})
        parcelles_dispo = pd.concat([parcelles_dispo,tabtemp], ignore_index=True)
    return parcelles_dispo

def get_coordinates(x: int, shapefile: gpd.geodataframe.GeoDataFrame) -> Union[np.array, str, int]:
    """Retrieve information from randomly selected parcel 

    Args:
        x (int): random parcel id
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 

    Returns:
        Union[np.array, str, int]: plot contour coordinates, id, code group 
    """
    parcel = shapefile.loc[shapefile['ID_PARCEL'] == str(x)]
    point = parcel.geometry.centroid.iloc[0]

    id_parcelle = f'{point.x}_{point.y}'
    coord_parcelle = np.array(parcel.exterior.iloc[0].coords).tolist()
    code = int(parcel.CODE_GROUP.iloc[0])

    return coord_parcelle, id_parcelle, code

def maskS2clouds(image: ee.image.Image) -> ee.image.Image:
    """Satellite image filtering on cloud pourcentage

    Args:
        image (ee.image.Image): satellite image 

    Returns:
        ee.image.Image: satellite image filtered
    """
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask) \
        .select("B.*") \
        .copyProperties(image, ["system:time_start"])

def reduce_region(image: ee.image.Image, geometry: ee.geometry.Geometry) -> ee.image.Image:
    """Reduction to keep a single value for ndvi and evi on the plot 

    Args:
        image (ee.image.Image): satellite image
        geometry (ee.geometry.Geometry): parcelle geometry

    Returns:
        ee.image.Image: single ndvi/evi values on the plot 
    """
    mean_bands = image.select('B.*').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10
    )
    return image.set('mean_bands', [mean_bands.get('B1'),
                                    mean_bands.get('B2'),
                                    mean_bands.get('B3'),
                                    mean_bands.get('B4'),
                                    mean_bands.get('B5'),
                                    mean_bands.get('B6'),
                                    mean_bands.get('B7'),
                                    mean_bands.get('B8'),
                                    mean_bands.get('B8A'),
                                    mean_bands.get('B9'),
                                    mean_bands.get('B10'),
                                    mean_bands.get('B11'),
                                    mean_bands.get('B12')])

def write_json(json_data: dict, path: str) -> None:
    """Downloading data in json files 

    Args:
        json_data (dict): parcel informations (ndvi,evi,date, ...)
        path (str): path to json file
    """
    try:
        with open(f'{path}','' 'r') as f:
            json_file = json.load(f)
            #print json_data.values() # View Previous entries
            json_file.update(json_data)
            f.close()
    except Exception as e:
        json_file = json_data
    
    with open(f'{path}', 'w') as f:
        f.write(json.dumps(json_file))
        f.close()

def get_timeSeries_ndvi_evi(i: int, x: int, shapefile: gpd.geodataframe.GeoDataFrame, start_date: str, end_date: str, path: str, region: str):
    """data retrieval and recording 

    Args:
        i (int): i th values 
        x (int): random id
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 
        start_date (str): image recovery from start_date
        end_date (str): to end_date
        path (str): path to json file 
        region (str): region name 
    """
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-agroitk")

        coord_parcelle, id_parcelle, code_group = get_coordinates(x, shapefile)
        geometry = ee.Geometry.Polygon([coord_parcelle])

        collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        ).map(maskS2clouds)

        collection = collection.map(lambda image: reduce_region(image, geometry))

        date_series = collection.aggregate_array('system:time_start').getInfo()
        bands_series = np.array(collection.aggregate_array('mean_bands').getInfo())
        
        data = {}
        data[i] = {}
        data[i]["id"] = id_parcelle
        data[i]["code_group"] = code_group
        data[i]["data"] = {}
        data[i]["data"]["date"] = date_series

        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']

        for j, band in enumerate(bands):
            data[i]["data"][band] = bands_series[:,j].tolist()


        write_json(data, path)
        print(f'Execution : {i}')

    except Exception as e:
        print(f'error : {e}')
        pass
    
def __main__():
    iteration = 0
    start_date = '2022-01-01'  # Date de début
    end_date = '2022-12-31'    # Date de fin
    path = "../data/data_homo_fullbands.json" # Lieu d'enregistreement des données
    directory = "../data/shapefiles/"
    nombre_parcelles = 20
    
    filenames = get_filenames(directory)

    ref_cultures = pd.read_csv(f"../data/REF_CULTURES_GROUPES_CULTURES_2020.csv", on_bad_lines='skip', sep=';')
    ref_cultures = ref_cultures[['CODE_GROUPE_CULTURE', 'LIBELLE_GROUPE_CULTURE']]
    ref_cultures = ref_cultures.drop_duplicates().reset_index(drop="True")
    ref_cultures['Nombre'] = nombre_parcelles

    for i, filename in enumerate(filenames):
        print(f'Loading {filename} : {i+1}/{len(filenames)}')
        shapefile = gpd.read_file(f'{directory}{filename}').to_crs("EPSG:4326")
        shapefile_data = pd.DataFrame(shapefile.groupby(['CODE_GROUP']).count()).reset_index().iloc[:, [0, 3]]
        shapefile_data.columns = ['CODE_GROUPE_CULTURE','Nombre']
        print('Shapefile load')

        parcelles_dispo = get_parcelles_dispo(ref_cultures, shapefile_data)
        print(parcelles_dispo)

        for par, code in enumerate(parcelles_dispo['CODE_GROUPE_CULTURE']):
            print(f'Code parcelle {code}: {par+1}/{len(parcelles_dispo['CODE_GROUPE_CULTURE'])} ')
            id_temp = []
            temp = shapefile.loc[shapefile["CODE_GROUP"] == str(code), :].reset_index()
            nb = parcelles_dispo.loc[parcelles_dispo["CODE_GROUPE_CULTURE"] == int(code), "Nombre"].iloc[0]
            while len(id_temp)<nb:
                x = random.randint(0, len(temp)-1)
                while temp['ID_PARCEL'][x] in id_temp:
                    x = random.randint(0, len(temp)-1)

                get_timeSeries_ndvi_evi(iteration, temp['ID_PARCEL'][x], shapefile, start_date, end_date, path)
                id_temp.append(temp['ID_PARCEL'][x])
                iteration += 1

        for i, row in parcelles_dispo.iterrows():
            code_culture = row["CODE_GROUPE_CULTURE"]
            nombre = row["Nombre"]
            autre_nombre = ref_cultures.loc[ref_cultures["CODE_GROUPE_CULTURE"] == code_culture, "Nombre"]
            ref_cultures.loc[ref_cultures["CODE_GROUPE_CULTURE"] == code_culture, "Nombre"] = nombre - autre_nombre + nombre_parcelles
        print('Mise a jours des nombres réalisé')

__main__()