import ee
import os 
import json
import base64
import joblib
import random
import datetime

from io import BytesIO
from typing import Union, Tuple 
from scipy.spatial import distance
from scipy.signal import savgol_filter

import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

class Classifier:

    def __init__(self) -> None:
        self.start_date = '2022-01-01'  # Date de début
        self.end_date = '2022-12-31'    # Date de fin
        self.rfc = joblib.load("/api-prediction-prairie/models/RandomForestPrairie.joblib")

    def maskS2clouds(self, image: ee.image.Image) -> ee.image.Image:
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

    def compute_ndvi(self, image: ee.image.Image) -> ee.image.Image:
        """Add NDVI Bands 

        Args:
            image (ee.image.Image): satellite image 

        Returns:
            ee.image.Image: ndvi satellite image 
        """
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    def compute_evi(self, image: ee.image.Image) -> ee.image.Image:
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

    def reduce_region(self, image: ee.image.Image, geometry: ee.geometry.Geometry) -> ee.image.Image:
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

    def get_timeSeries_ndvi_evi(self, polygone: gpd.geodataframe.GeoDataFrame) -> Union[list, list]:
        """Recovery of ndvi timeseries and associated date 

        Args:
            polygone (gpd.geodataframe.GeoDataFrame): coordinates of the parcel 

        Returns:
            Union[list, list]: date and ndvi timeseries
        """
        try:
            service_account = 'api-prediction-prairie@ee-agroitk.iam.gserviceaccount.com'
            credentials = ee.ServiceAccountCredentials(service_account, '/api-prediction-prairie/ee-agroitk-a58b9b9b9b5b.json')
            ee.Initialize(credentials, project="ee-agroitk")

            geometry = ee.Geometry.Polygon(polygone.tolist())

            collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                            .filterDate(self.start_date, self.end_date)
                            .filterBounds(geometry)
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                            ).map(self.maskS2clouds)

            collection = collection.map(self.compute_ndvi).map(self.compute_evi).map(lambda image: self.reduce_region(image, geometry))

            ndvi_series = collection.aggregate_array('mean_ndvi').getInfo()
            #evi_series = collection.aggregate_array('mean_evi').getInfo()
            date_series = collection.aggregate_array('system:time_start').getInfo()

            return date_series, ndvi_series

        except Exception as e:
            print(f'error {e}')
            return 0, 0
        
    def month_ndvi(self, date: list, ndvi: list) -> list:
        """Transformation into a fixed timeserie of 1 value every two weeks

        Args:
            date (list): date timeseries
            ndvi (list): ndvi timeseries

        Returns:
            list: new ndvi timeseries
        """
        monthly_ndvi = []
        for i in range(1, 13):
            datetimes = [datetime.datetime.fromtimestamp(t/1000) for t in date]
            is_first_half = [d.month == i and d.day <= 15 for d in datetimes] # bool list of first 15 days of month
            is_second_half = [d.month == i and d.day > 15 for d in datetimes] # bool list of last 15 days of month

            ## First half
            sum_ndvi_first = sum(v for v, is_f in zip(ndvi, is_first_half) if is_f)
            count_of_values_first = sum(is_first_half)
            try:
                monthly_ndvi.append(sum_ndvi_first / count_of_values_first)
            except ZeroDivisionError:
                monthly_ndvi.append(None)

            # Second half 
            sum_ndvi_second = sum(v for v, is_s in zip(ndvi, is_second_half) if is_s)
            count_of_values_second = sum(is_second_half)
            try:
                monthly_ndvi.append(sum_ndvi_second / count_of_values_second)
            except ZeroDivisionError:
                monthly_ndvi.append(None)

        return monthly_ndvi
    
    def interpolate_linear(self, timeseries: list) -> list:
        """linear interpolation of timeseries

        Args:
            timeseries (list): timeseries to interpolate 

        Returns:
            list: interpolate timeserie
        """
        # Boucle pour l'interpolation linéaire
        for i in range(len(timeseries)):
            if timeseries[i] is None:
                # Trouver les indices des valeurs non nulles les plus proches
                j = i - 1
                while timeseries[j] is None:
                    j -= 1
                k = i + 1
                while k < len(timeseries) and timeseries[k] is None:
                    k += 1
                # Calculer la valeur intermédiaire
                if j < 0:
                    valeur_intermediaire = timeseries[k]
                elif k >= len(timeseries):
                    valeur_intermediaire = timeseries[-1]
                else:
                    valeur_intermediaire = timeseries[j] + (timeseries[k] - timeseries[j]) * (i - j) / (k - j)
                # Remplacer la valeur None par la valeur intermédiaire
                timeseries[i] = valeur_intermediaire
        
        none_indices = [i for i, liste in enumerate(timeseries) if liste is None]
        
        try : 
            last = none_indices[0]
            for i in range(last, len(timeseries)):
                timeseries[i] = timeseries[last-1]
        except IndexError:
            pass
        return timeseries
    
    def correction(self, timeseries: list) -> list:
        """timeseries correction with linear interpolation and polynomial regression (5)

        Args:
            timeseries (list): timeseries

        Returns:
            list: corrected timeserie 
        """
        interpolated = self.interpolate_linear(timeseries)
        polynomial_regression = savgol_filter(interpolated, window_length=11, polyorder=5, mode="nearest")
        max_poly = [max(interpolated[i], polynomial_regression[i]) for i in range(len(interpolated))]

        return max_poly
    
    def plot_ndvi(self, ndvi_series: list, ndvi_corrected: list) -> str:
        """Display NDVI curve ( original and corrected)

        Args:
            ndvi_series (list): original ndvi
            ndvi_corrected (list): corrected ndvi 

        Returns:
            str: image plot encode
        """
        dates = [mdates.date2num(datetime.datetime(2022, month, day)) for month in range(1, 13) for day in [1, 15]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, ndvi_series, label='NDVI série')
        ax.plot(dates, ndvi_corrected, label='NDVI corrigé')

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        ax.set_xticks(dates)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # Fixer les limites de l'axe des y
        ax.set_ylim(0, 1)
        ax.legend()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode()

        # Fermer la figure pour libérer la mémoire
        plt.close(fig)
        print(type(image_base64))

        return image_base64
    


    def prediction(self, polygone: np.array) -> dict:
        """Get informations about prairie prediction

        Args:
            polygone (np.array): coordinates of the parcel

        Returns:
            dict: bool prairie, plot ndvi encode
        """

        date_series, ndvi_real = self.get_timeSeries_ndvi_evi(polygone)

        ndvi_series = self.month_ndvi(date_series, ndvi_real)
        ndvi_corrected  = self.correction(ndvi_series)

        ndvi_curve = self.plot_ndvi(ndvi_series, ndvi_corrected)

        prediction = self.rfc.predict([ndvi_corrected])
    
        if prediction[0] == 1:
            print("C'est une Prairie")
            return True, ndvi_curve
        else:
            print("Ce n'est pas une Prairie")
            return False, ndvi_curve