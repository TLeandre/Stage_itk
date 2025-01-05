import ee

import geopandas as gpd
import os 
import numpy as np
import datetime

from typing import Union, Tuple 

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
        ee.image.Image: single ndvi/evi values on the parcel 
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

def reduce_region_full_bands(image: ee.image.Image, geometry: ee.geometry.Geometry) -> ee.image.Image:
    """Reduction to keep a single value for all bands on parcel 

    Args:
        image (ee.image.Image): satellite image
        geometry (ee.geometry.Geometry): parcelle geometry

    Returns:
        ee.image.Image: single bands values on the parcel 
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

def get_timeSeries_ndvi_evi(shapefile: gpd.geodataframe.GeoDataFrame, start_date: str, end_date: str) -> Union[list, list]:
    """Recovery of ndvi timeseries and associated date 

    Args:
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 
        start_date (str): image recovery from start_date
        end_date (str): to end_date

    Returns:
        Union[list, list]: date and ndvi timeseries
    """
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-agroitk")

        coord_parcelle = np.array(shapefile.exterior[0].coords).tolist()

        geometry = ee.Geometry.Polygon([coord_parcelle])

        collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        ).map(maskS2clouds)

        collection = collection.map(compute_ndvi).map(compute_evi).map(lambda image: reduce_region(image, geometry))

        ndvi_series = collection.aggregate_array('mean_ndvi').getInfo()
        #evi_series = collection.aggregate_array('mean_evi').getInfo()
        date_series = collection.aggregate_array('system:time_start').getInfo()

        return date_series, ndvi_series

    except Exception as e:
        return 0
    
def get_timeSeries_full_bands(shapefile: gpd.geodataframe.GeoDataFrame, start_date: str, end_date: str):
    """Recovery of all bands timeseries and associated date 

    Args:
        shapefile (gpd.geodataframe.GeoDataFrame): group of parcels 
        start_date (str): image recovery from start_date
        end_date (str): to end_date

    Returns:
        Union[list, list]: date and bands timeseries
    """
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-agroitk")

        coord_parcelle = np.array(shapefile.exterior[0].coords).tolist()

        geometry = ee.Geometry.Polygon([coord_parcelle])

        collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        ).map(maskS2clouds)

        collection = collection.map(lambda image: reduce_region_full_bands(image, geometry))

        date_series = collection.aggregate_array('system:time_start').getInfo()
        bands_series = np.array(collection.aggregate_array('mean_bands').getInfo())

        return date_series, bands_series

    except Exception as e:
        print(e)
        return 0