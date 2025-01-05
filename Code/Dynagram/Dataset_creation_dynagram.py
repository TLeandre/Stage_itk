import os 
import ee
import time
import json
import random


import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from datetime import datetime

from itksensors.custom_exception import SoilException
from itksensors.soil_client import SoilClient
from itksensors.soil_dto import SoilDto
from itksensors.station_dto import GeoLocation
from itksensors.weather import WeatherStationClient
from soil.io import create_voxel, SoilVoxel, TextureParams
from weather.fao import eto_daily
from weather.radiation import rg_daily

from agrosim.launcher import launch
from dynagram.sim_wrapper import SimWrapper



from typing import Union, Tuple 

from joblib import Parallel, delayed

import dynagram

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

    while (shapefile['SURF_PARC'][x] > 1 and shapefile['CODE_GROUP'][x] == '18') == False:
        x = random.randint(0, len(shapefile))

    point = shapefile.geometry[x].centroid
    lon = float(point.x)
    lat = float(point.y)

    coord_parcelle = np.array(shapefile.exterior[x].coords).tolist()
    
    return coord_parcelle, lon, lat

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

def get_timeSeries(i: int, shapefile: gpd.geodataframe.GeoDataFrame, 
                   start_date: str, end_date: str, path: str, region: str,
                   weather_connection, soil_connection):
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

        coord_parcelle, lon, lat = random_coordinates(shapefile)

        ## Simulateur biomass
        out = get_biomass(lat, lon, start_date, end_date, weather_connection, soil_connection)
        biomass = np.sum([specie["biomass"] for specie in out["species"]], axis=0)

        geometry = ee.Geometry.Polygon([coord_parcelle])
        collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        ).map(maskS2clouds)

        collection = collection.map(lambda image: reduce_region(image, geometry))

        date_series = collection.aggregate_array('system:time_start').getInfo()
        bands_series = np.array(collection.aggregate_array('mean_bands').getInfo())

        biomass_series, idx = sort_biomass(out['t'], date_series, biomass)

        date_series = np.array(date_series)[idx].tolist()
        
        data = {}
        data[i] = {}
        data[i]["id"] = f'{lon}_{lat}'
        data[i]["data"] = {}
        data[i]["data"]["date"] = date_series
        data[i]["data"]["biomass"] = biomass_series

        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
        
        for j, band in enumerate(bands):
            band_grouped = np.split(bands_series[:,j].tolist(), idx[1:])
            avg_band = [np.mean(bg) for bg in band_grouped]
            data[i]["data"][band] = avg_band


        write_json(data, path)
        print(f'Execution {i}')

    except Exception as e:
        print(e)
        pass
    

def get_meteo(station_name: str, latitude: float, longitude: float, weather_connection: WeatherStationClient,
              date: list[datetime] = None, delete_station: bool = True) -> list[dict]:
    """
    Get the meteo for a specific parcel between some date

    Args :
        station_name (str): station name for sensors
        latitude (float): latitude
        longitude (float): longitude
        weather_connection (WeatherStationClient): weather client to managed station creation
        date (list[datetime]): dates between which the meteo will be downloaded
        delete_station (bool): station is deleted or not after loading the data

    Returns :
        weather_data (list[dict]): meteo outputs for different dates
    """

    start_date, end_date = (datetime(2022, 1, 1), datetime(2023, 12, 31)) \
        if date is None else (date[0], date[1])

    station_name += "_weather"
    start_date_station = start_date
    end_date_station = end_date
    station_list = weather_connection.retrieve_station_list()
    station_list_name = [station_list[i]['name'] for i in range(len(station_list))]

    if station_name not in station_list_name:
        print(f"Creating station {station_name} for weather")
        r = weather_connection.create_station(station_name, str(latitude), str(longitude), 'MAXAR',
                                              date_installation=start_date_station,
                                              date_uninstallation=end_date_station)

        if not r:
            raise SoilException("Station creation failed for weather")
        time.sleep(5)
    else:
        print(f"Station {station_name} already exists for weather")

    weather_data = weather_connection.retrieve_station(station_name, start_date, end_date)

    if delete_station:
        weather_connection.delete_station(station_name)

    return weather_data


def get_hwsd_soil_data(station_name: str, latitude: float, longitude: float, soil_client: SoilClient,
                       delete_station: bool) -> dict:
    """
    Usefully function to extract soil data from HWSD soil database on sensor

    Args:
        station_name (str):
        latitude (float):
        longitude (float):
        soil_client (SoilClient): sensor soil client connection
        delete_station (bool): True if delete station after get data otherwise False

    Returns:
        (dict)
    """
    station_name += "_soil"
    geometry = GeoLocation(lat=latitude, long=longitude)

    soil_dto = SoilDto(name=station_name, code=station_name, coordinates=geometry)

    # check if station already exist
    soil_station_before_creation = soil_client.list_station()
    is_station_name_exist = soil_client.is_station_name_already_exist(station_name, soil_station_before_creation)

    if not is_station_name_exist:
        print(f"Creating station {station_name} for soil")
        r = soil_client.create_station(soil_dto)
        if not r:
            raise SoilException("Station creation failed for soil")
        time.sleep(5)
        if not soil_client.is_station_name_already_exist(station_name, soil_client.list_station()):
            raise SoilException("Station creation failed for soil")
    else:
        print(f"Station {station_name} already exists for soil")

    # Get soil composition
    soil_composition = soil_client.get_soil_station_data(code=station_name)

    if delete_station:
        soil_client.delete_station(station_name)

    return soil_composition


def compute_eto(weather: dict, latitude: float) -> list[float]:
    """
    Compute the evapotranspiration for different dates

    Args :
        weather (dict): meteo of the parcel
        latitude (float): latitude of the parcel [°]

    Returns :
        eto (list[float]): evapotranspiration for different dates
    """
    eto = []
    for w in weather:
        date = datetime.strptime(w['date'][:10], '%Y-%m-%d')
        _eto = eto_daily(date, latitude * 2 * 3.14 / 365, w['minTemperature'], w['maxTemperature'], w["solarRadiation"],
                         w['relativeHumidity'], w['windSpeed'])
        eto.append(_eto)
    return eto


def aggregate_data(weather: list, et0: float, voxel: SoilVoxel):
    """
    Aggregate the temperature, rain, par and et0

    Args :
        weather (list): meteo of the parcel
        et0 (float): evapotranspiration
        voxel (SoilVoxel): HWSD soil data

    Returns :
        aggregated_data (dict[list[float]]): Dict of inputs
    """
    factor_par = 0.5
    whc = voxel.wc_fc()
    pwp = voxel.wc_pwp()
    
    aggregated_data = {"soil": {"whc": whc, "pwp": pwp},
                       "species": {"species_list": ["Lol.per", "Tri.rep", "Ely.rep", "Ant.odo"]},
                       "weather": {"date": [], "temp": [], "rain": [], "par": [], "etp_pot": []}}

    for i in range(len(weather)):
        aggregated_data["weather"]["date"].append(weather[i]["date"][:10])
        aggregated_data["weather"]["temp"].append(weather[i]["meanTemperature"])
        aggregated_data["weather"]["rain"].append(weather[i]["totalPrecipitation"])
        aggregated_data["weather"]["par"].append(weather[i]["solarRadiation"] * factor_par / 100.)
        aggregated_data["weather"]["etp_pot"].append(et0[i])

    return aggregated_data


def get_biomass(latitude, longitude, start_date, end_date, weather_connection, soil_connection):
    print('Biomass recovery')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    station_name = f"biomass_sim_{round(latitude, 5)}_{round(longitude,5)}"

    """First get the soil data to compute WHC and PWP"""
    soil_data = get_hwsd_soil_data(station_name, latitude, longitude, soil_connection, delete_station=True)
    soil = soil_data["topSoilComposition"]
    voxel = create_voxel(TextureParams(clay_fraction=soil["clayPercentage"] / 100.,
                                       om_fraction=soil["organicCarbonPercentage"] * 1.724 / 100.,
                                       sand_fraction=soil["sandPercentage"] / 100.,
                                       stone_fraction=soil["gravelPercentage"] / 100.,
                                       thickness=200.))

    """Then get meteo data"""
    weather_data = get_meteo(station_name, latitude, longitude, weather_connection, date=(start_date, end_date),
                             delete_station=True)
    
    evapotranspiration = compute_eto(weather_data, latitude)

    """Then aggregate the data and write the file"""
    aggregated_data = aggregate_data(weather_data, evapotranspiration, voxel)
    aggregated_data['management'] = {}

    out = launch(SimWrapper, aggregated_data)
    
    return out

def sort_biomass(dates_str, dates_timestamp, biomass):
    dates_simu = [datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S').date() for date_str in dates_str]
    dates_rs = [datetime.fromtimestamp(timestamp // 1000).date() for timestamp in dates_timestamp]

    dates, idx = np.unique(dates_rs, return_index=True)

    to_keep = np.array([date_simu in dates_rs for date_simu in dates_simu])
    biomass = np.array(biomass)[to_keep].tolist()

    return biomass, idx

def __main__():
    start_nb = 149 
    data_nb = 800 #nb per shapefile
    start_date = '2022-01-01'  # Date de début
    end_date = '2022-12-31'    # Date de fin
    path = "../data/biomass_simu.json" # Lieu d'enregistreement des données
    directory = "../data/shapefiles/"
    sensor_login = 'ag-kilimo'
    sensor_password = 'AcA2aeMx'
    sensor_instance = 'Prod'

    print('Connection')
    weather_connection = WeatherStationClient(login=sensor_login, pwd=sensor_password, instance=sensor_instance)
    soil_connection = SoilClient(login=sensor_login, pwd=sensor_password, instance=sensor_instance)
    print('Successful Connection')
    
    filenames = get_filenames(directory)

    for i, filename in enumerate(filenames):
        print(f'Loading {filename} : {i+1}/{len(filenames)}')
        shapefile = gpd.read_file(f'{directory}{filename}').to_crs("EPSG:4326")
        print('Shapefile load')

        nb_core = 5
        Parallel(n_jobs=nb_core)(delayed(get_timeSeries)(j, shapefile, start_date, end_date, path, filename[:-4], weather_connection, soil_connection) for j in range(start_nb, start_nb+data_nb))
        start_nb += data_nb
        
        #for j in range(start_nb, start_nb+data_nb):
            #get_timeSeries(j, shapefile, start_date, end_date, path, filename[:-4], weather_connection, soil_connection)
        #start_nb += data_nb
    print('END')

__main__()