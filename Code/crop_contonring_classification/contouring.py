import io
import cv2
import sys
import requests

import numpy as np

from PIL import Image
from ultralytics import YOLO
from typing import Union, Tuple 
from PIL.JpegImagePlugin import JpegImageFile
from ultralytics.engine.results import Results

class Contouring:

    def __init__(self, api_token:str) -> None:
        self.api_token = api_token
        self.pixel = 512
        self.offset = np.array([0.0096000, 0.0064000])
        self.coef_seg = 0.001
        self.model = YOLO("models/contouring.pt")
    
    def create_box(self, center: np.ndarray) -> np.ndarray:
        """create a coordinate box around the center of a random parcel

        Args:
            center (np.ndarray): Coordinates of the point

        Returns:
            np.ndarray: coordinate box
        """
        lower = center - self.offset/2
        upper = center + self.offset/2

        return np.array([lower, upper]).reshape(-1)
    
    def _request(self, url: str) -> Union[requests.models.Response, None]:
        """api response

        Args:
            url (str): url request

        Returns:
            Union[requests.models.Response, None]: response
        """
        response = requests.get(url)
        if response.status_code != 200:
            sys.stderr.write(f"GET request failed. Status code: {response.status_code}\n")
            response = None
        return response

    def _request_image_box(self, lon_min: np.float64, lat_min: np.float64, lon_max: np.float64, 
                        lat_max: np.float64) -> requests.models.Response:
        """api response for a satellite image

        Args:
            lon_min (np.float64): longitude min
            lat_min (np.float64): latitude min
            lon_max (np.float64): longitude max
            lat_max (np.float64): latitude max

        Returns:
            requests.models.Response: response
        """
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[{lon_min},{lat_min},{lon_max},{lat_max}]/{self.pixel}x{self.pixel}?logo=false&attribution=false&access_token={self.api_token}"
        return self._request(url)

    def _extract_content_from_reponse(self, response: requests.models.Response) -> Union[Tuple[JpegImageFile, str], Tuple[None, None]]:
        """image extraction from API response

        Args:
            response (requests.models.Response): API response

        Returns:
            Union[Tuple[JpegImageFile, str], Tuple[None, None]]: image with its format
        """
        if response:
            content_type = response.headers["content-type"]
            format = content_type.split('/')[-1]
            image = Image.open(io.BytesIO(response.content))
            return image, format

        raise Exception("Récupération de l'image impossible") 
            
        
    
    def filter_mask(self, results : Results) -> Union[np.ndarray, None]:
        """keep the mask we're interested (central mask) 

        Args:
            results (Results): model prediction

        Returns:
            Union[np.ndarray, None]: Central mask 
        """
        masks = results.masks.data.cpu().numpy().astype(np.uint8)
        masks = (masks * 255)

        conf = results.boxes.conf.data.cpu().numpy().astype(np.float32) 
        center_mask_bool = masks[:, 256, 256] == 255

        filtered_conf = conf[center_mask_bool]

        # Comptez le nombre de masques centrés
        num_center_masks = np.sum(center_mask_bool)
        print(f'Nombre de masque centré : {num_center_masks}')

        if num_center_masks > 0:
            max_conf_index = np.argmax(filtered_conf)
            return masks[center_mask_bool, :, :][max_conf_index]
        else:
            return None
        
    def mask_cleaning(self, mask: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """mask cleaning and calculation of image coordinates 

        Args:
            mask (np.ndarray): central mask

        Returns:
            Union[np.ndarray, np.ndarray]: mask clean, polygon coordinates
        """
        # Création des contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #calcul d'air des contours
        areas = [cv2.contourArea(contour) for contour in contours]
        #récupération du contour le plus grand 
        max_index = np.argmax(areas)

        #création du nouveau mask
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, contours, max_index, 255, -1)
        
        epsilon = self.coef_seg * cv2.arcLength(contours[max_index], True)
        polygone = cv2.approxPolyDP(contours[max_index], epsilon, True)
        polygone_coor = polygone.reshape(-1,2)
        
        return new_mask, polygone_coor
    
    def transform_coordinate(self, contours: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        """transformation of parcel coordinates relative to the image into geographic coordinates 

        Args:
            contours (np.ndarray): parcel outline 
            coordinate (np.ndarray): box coordinates  

        Returns:
            np.ndarray: geographic coordinate of the polygon
        """
        contours = np.array([[0, 1] for i in range(np.shape(contours)[0])]) - contours/ self.pixel
        contours = np.abs(contours)

        poly_coordinate = coordinate[:2] + contours * self.offset

        return poly_coordinate

    def prediction(self, lon: str, lat: str) -> Union[str, str, np.array]:
        """Get prediction information about parcel contouring detection

        Args:
            lon (str): longitude
            lat (str): latitude

        Returns:
            Union[str, str, np.array]: image encode, image with contour encode, coordinate of the parcel
        """
        center = np.array([lon,lat], dtype=float)
        coord = self.create_box(center)

        response = self._request_image_box(coord[0], coord[1], coord[2], coord[3])

        image = self._extract_content_from_reponse(response)[0]

        #prédiction
        results = self.model(image)[0]

        centered_mask = self.filter_mask(results)

        if centered_mask is None:
            print("Aucune parcelle n'a été detecté ")
            return None, None
            
        #post processing mask
        clean_mask, contours = self.mask_cleaning(centered_mask)

        poly_coordinate = self.transform_coordinate(contours, coord)

        return poly_coordinate