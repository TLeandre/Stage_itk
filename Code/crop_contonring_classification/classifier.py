import joblib

import numpy as np

import tensorflow as tf

import keras
from keras.layers import Dense, Dropout
from gee_data import EeInit
import gee_data 
import preprocessing as pp


class Classifier:

    def __init__(self, project) -> None:
       
        self.ee =  EeInit(project=project).ee
        self.rfc = joblib.load("models/RandomForestPrairie.joblib")
        self.model = self.construct_model("models/multicrop_classification.h5")
    
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

    def prediction(self, polygone: np.array, start_date: str, end_date: str) -> dict:
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
        date_series, ndvi_real = gee_data.get_timeSeries_ndvi_evi(polygone, start_date, end_date, self.ee)
        ndvi_series = pp.month_timeserie(date_series, ndvi_real)
        ndvi_corrected  = pp.correction(ndvi_series)
        prediction_prairie = self.rfc.predict([ndvi_corrected])

        ##Crop 
        date_series, bands_series = gee_data.get_timeSeries_full_bands(polygone, start_date, end_date, self.ee)
        bands_to_predict = pp.transform_data(bands_series, date_series)
        prediction_crop = np.argmax(self.model.predict(bands_to_predict, verbose=0))

        return bool(prediction_prairie[0]), crops[prediction_crop]