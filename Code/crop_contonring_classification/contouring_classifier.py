from contouring import Contouring 
from classifier import Classifier 

def contouring_classification(lat, lon, api_token, project, start_date, end_date):
    contouring = Contouring(api_token=api_token)
    classifier = Classifier(project=project)

    polygone = contouring.prediction(lon, lat)

    prairie_prediction, crop_prediction = classifier.prediction(polygone, start_date=start_date, end_date=end_date)

    print(polygone)
    print(prairie_prediction)
    print(crop_prediction)

if __name__ =="__main__":
    contouring_classification(47.1294, 
                              1.2996, 
                              'pk.eyJ1IjoibGVhbmRyZTIwMjQiLCJhIjoiY2x2YXpwNHh1MDNmYjJscGJ1d21odXFjaSJ9.BQOCqEj6OHU-TnS7HeAg5Q', 
                              "ee-agroitk", 
                              '2022-01-01', 
                              '2022-12-31')