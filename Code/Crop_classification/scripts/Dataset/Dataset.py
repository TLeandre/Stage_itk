import ee

import geopandas as gpd
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

def random_coordinates(shapefile: gpd.geodataframe.GeoDataFrame) -> Union[np.array, str, int]:
    """Retrieve information from randomly selected parcel 

    Args:
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 

    Returns:
        Union[np.array, str, int]: plot contour coordinates, id, code group 
    """
    x = random.randint(0, len(shapefile))

    while (shapefile['SURF_PARC'][x] > 1) == False:
        x = random.randint(0, len(shapefile))

    point = shapefile.geometry[x].centroid
    id_parcelle = f'{point.x}_{point.y}'

    coord_parcelle = np.array(shapefile.exterior[x].coords).tolist()

    code_group = int(shapefile['CODE_GROUP'][x])
    
    return coord_parcelle, id_parcelle, code_group

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

def compute_ndvi(image: ee.image.Image) -> ee.image.Image:
    """Add NDVI Bands 

    Args:
        image (ee.image.Image): satellite image 

    Returns:
        ee.image.Image: ndvi satellite image 
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def compute_evi(image: ee.image.Image) -> ee.image.Image:
    """Add EVI Bands

    Args:
        image (ee.image.Image): satellite image 

    Returns:
        ee.image.Image: evi satellite image 
    """
    #2.5 * ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))
    evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR' : image.select('B8').divide(10000),
        'RED' : image.select('B4').divide(10000),
        'BLUE': image.select('B2').divide(10000)}).rename('EVI')
    return image.addBands(evi)

def reduce_region(image: ee.image.Image, geometry: ee.geometry.Geometry) -> ee.image.Image:
    """Reduction to keep a single value for ndvi and evi on the plot 

    Args:
        image (ee.image.Image): satellite image
        geometry (ee.geometry.Geometry): parcelle geometry

    Returns:
        ee.image.Image: single ndvi/evi values on the plot 
    """
    mean_ndvi = image.select('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10
    )
    mean_evi = image.select('EVI').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10
    )
    return image.set('mean_ndvi', mean_ndvi.get('NDVI'), 'mean_evi', mean_evi.get('EVI'))

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

def get_timeSeries_ndvi_evi(i: int, shapefile: gpd.geodataframe.GeoDataFrame, start_date: str, end_date: str, path: str, region: str):
    """data retrieval and recording 

    Args:
        i (int): i th values 
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 
        start_date (str): image recovery from start_date
        end_date (str): to end_date
        path (str): path to json file 
        region (str): region name 
    """
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-agroitk")

        coord_parcelle, id_parcelle, code_group= random_coordinates(shapefile)
        if code_group == 18 or code_group == 19:
            prairie = True
        else : 
            prairie = False

        geometry = ee.Geometry.Polygon([coord_parcelle])

        collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        ).map(maskS2clouds)

        collection = collection.map(compute_ndvi).map(compute_evi).map(lambda image: reduce_region(image, geometry))

        ndvi_series = collection.aggregate_array('mean_ndvi').getInfo()
        evi_series = collection.aggregate_array('mean_evi').getInfo()
        date_series = collection.aggregate_array('system:time_start').getInfo()

        data = {}
        data[i] = {}
        data[i]["id"] = id_parcelle
        data[i]["region"] = region
        data[i]["code_group"] = code_group
        data[i]["prairie"] = prairie
        data[i]["data"] = {}
        data[i]["data"]["date"] = date_series
        data[i]["data"]["ndvi"] = ndvi_series
        data[i]["data"]["evi"] = evi_series

        write_json(data, path)
        print(f'Execution {i}')

    except Exception as e:
        print(e)
        pass
    
def __main__():
    start_nb = 0
    data_nb = 3200 #nb per shapefile
    start_date = '2022-01-01'  # Date de début
    end_date = '2022-12-31'    # Date de fin
    path = "../data/data_region.json" # Lieu d'enregistreement des données
    directory = "../data/shapefiles/"
    

    filenames = get_filenames(directory)

    for i, filename in enumerate(filenames):
        print(f'Loading {filename} : {i+1}/{len(filenames)}')
        shapefile = gpd.read_file(f'{directory}{filename}').to_crs("EPSG:4326")
        print('Shapefile load')

        for j in range(start_nb, start_nb+data_nb):
            get_timeSeries_ndvi_evi(j, shapefile, start_date, end_date, path, filename[:-4])
        start_nb += data_nb

__main__()