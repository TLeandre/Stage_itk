import ee
import os 
import joblib
import datetime

from typing import Union, Tuple 

import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import gee_data 
import preprocessing as pp


class Classifier:

    def __init__(self) -> None:
        self.start_date = '2022-01-01'  # Date de début
        self.end_date = '2022-12-31'    # Date de fin
        self.rfc = joblib.load("../models/RandomForestPrairie.joblib")
        self.model = self.construct_model("../models/multicrop_classification.h5")
    
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
        ax.set_title("NDVI Curve")
        ax.legend()

        return fig
    
    def construct_model(self, path: str) -> keras.src.models.sequential.Sequential:
        """building the model skeleton

        Args:
            path (str): path to weight model

        Returns:
            keras.src.models.sequential.Sequential: model
        """
        model_dense = tf.keras.Sequential([
            tf.keras.Input((312, )),
            Dense(312),
            Dense(156),
            Dropout(0.1),
            Dense(78),
            Dropout(0.1),
            Dense(39),
            Dense(11, activation=tf.nn.softmax)
        ])
    
        model_dense.summary()

        model_dense.load_weights(f"{path}")

        return model_dense

    def prediction(self, polygone: np.array) -> dict:
        """Get informations about prairie prediction

        Args:
            polygone (np.array): coordinates of the parcel

        Returns:
            dict: bool prairie, plot ndvi encode
        """

        crops = ['Autres céréales _ Plante à fibre _ Riz','Blé tendre','Maïs grain et ensilage',
             'Orge','Protéagineux _ leguminause à grain','Colza','Tournesol','Autres oléagineux',
             'Fourrage_Estives et landes_Prairies permanentes_Prairies temporaires',
             'Vergers_Vignes_Fruits à coque_Oliviers','Autres cultures industrielles_Légumes ou fleurs']

        #Prairie
        date_series, ndvi_real = gee_data.get_timeSeries_ndvi_evi(polygone, self.start_date, self.end_date)
        ndvi_series = pp.month_timeserie(date_series, ndvi_real)
        ndvi_corrected  = pp.correction(ndvi_series)
        prediction_prairie = self.rfc.predict([ndvi_corrected])

        ##Crop 
        date_series, bands_series = gee_data.get_timeSeries_full_bands(polygone, self.start_date, self.end_date)
        bands_to_predict = pp.transform_data(bands_series, date_series)
        prediction_crop = np.argmax(self.model.predict(bands_to_predict, verbose=0))

        ndvi_curve = self.plot_ndvi(ndvi_series, ndvi_corrected)

        return ndvi_curve, bool(prediction_prairie[0]), crops[prediction_crop]