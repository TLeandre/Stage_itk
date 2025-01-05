from typing import Union
import uvicorn
from fastapi import FastAPI

from Contouring import Contouring 
from Classifier import Classifier 

app = FastAPI()
contouring = Contouring()
#classifier = Classifier()

@app.get("/")
def activation_function():
    return {"Output" : "Mystere Egg"}

@app.get("/lon:{lon}/lat:{lat}")
def read_coordinates(lon: str, lat: str) -> dict:
    """get all element to display parcel informations

    Args:
        lon (str): longitude
        lat (str): latitude

    Returns:
        dict: image, image with contour, bool prairie, curve of ndvi
    """
    image_encode, image_contour_encode, polygone = contouring.prediction(lon, lat)

    if polygone is None:
        return {"image": image_encode, 
                "image_contour": None, 
                } #"prairie": None, "ndvi_curve": None
    
    #bool_prairie, ndvi_curve = classifier.prediction(polygone)

    return {"image": image_encode, 
            "image_contour": image_contour_encode, 
            } # "prairie": bool_prairie, "ndvi_curve": ndvi_curve

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
