import geopandas as gpd
import numpy as np

import datetime

from scipy.spatial import distance
from scipy.signal import savgol_filter

from typing import Union, Tuple 


def month_ndvi(date: list, ndvi: list) -> list:
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

def month_fullbands(date, band) -> list:
    """Transformation into a fixed timeserie of 1 value every two weeks

    Args:
        date (list): date timeseries
        band (list): band timeseries

    Returns:
        list: band timeserie corrected
    """
    monthly = []
    
    for i in range(1, 13):
        datetimes = [datetime.datetime.fromtimestamp(t/1000) for t in date]
        is_first_half = [d.month == i and d.day <= 15 for d in datetimes] # bool list of first 15 days of month
        val = np.array([x for x, m in zip(band, is_first_half) if m and x is not None])
        
        if len(val) > 0 :
            monthly.append(val.sum() / len(val))
        else :
            monthly.append(None)

        
        is_second_half = [d.month == i and d.day > 15 for d in datetimes] # bool list of last 15 days of month
        val = np.array([x for x, m in zip(band, is_second_half) if m and x is not None])
        
        if len(val) > 0 :
            monthly.append(val.sum() / len(val))
        else :
            monthly.append(None)
        
    return correction(monthly)


def interpolate_linear(timeseries: list) -> list:
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

def correction(timeseries: list) -> list:
    """timeseries correction with linear interpolation and polynomial regression (5)

    Args:
        timeseries (list): timeseries

    Returns:
        list: corrected timeserie 
    """
    interpolated = interpolate_linear(timeseries)
    polynomial_regression = savgol_filter(interpolated, window_length=11, polyorder=5, mode="nearest")
    max_poly = [max(interpolated[i], polynomial_regression[i]) for i in range(len(interpolated))]

    return max_poly

def normalize_parcelles(parcelles: dict) -> Union[list, list, list, list]:
    """Normalize training data from dict

    Args:
        parcelles (dict): all parcelles data

    Returns:
        Union[list, list, list, list]: segmentation data information from parcelles
    """
    X = []
    X_flat = []
    y = []
    c = []
    for i in range(len(parcelles)):
        try :
            y.append(parcelles[str(i)]['code_group'])

            if parcelles[str(i)]['code_group'] in [1,2,3,5,6,7]:
                c.append(parcelles[str(i)]['code_group'])
            elif parcelles[str(i)]['code_group'] in [4,9,14]:
                c.append(0)
            elif parcelles[str(i)]['code_group'] in [15, 8]:
                c.append(4)
            elif parcelles[str(i)]['code_group'] in [16, 17, 18, 19]:
                c.append(8)
            elif parcelles[str(i)]['code_group'] in [20, 21, 22, 23]:
                c.append(9)
            elif parcelles[str(i)]['code_group'] in [24, 25]:
                c.append(10)
                
            parcel = parcelles[str(i)]['data']
            arrays = np.array([value for keys, value in parcel.items()])
            
            indices = np.where(arrays[1] == None)
            date = np.delete(arrays[0], indices)
    
            montly = []
            for j in range(1, 14):
                band = np.delete(arrays[j], indices)
                montly.append(month_fullbands(date, band))
                
            X.append(np.transpose(np.array(montly)))
            X_flat.append(np.array(montly).flatten())
        except Exception:
            pass

    return X, X_flat, y, c