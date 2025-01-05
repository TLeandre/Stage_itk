import streamlit as st
from Contouring import Contouring 
from Classifier import Classifier 

contouring = Contouring()
classifier = Classifier()

st.markdown("# Crop segmentation and detection")
st.markdown("""Comment correctement utiliser cette platforme :  
            - Mettre votre point au centre de la parcelle voulue  
            - La parcelle doit être contenue dans un carré de 12 hectares  
            - L'entrainement et les données utilisé datent de 2022, cela signifie que la detection est réalisé pour 2022  
            """)

col1, col2 = st.columns(2)

with col1: 
    lon = st.number_input("Insert a longitude", value=1.2996, format="%f", placeholder="Type a number...")

with col2: 
    lat = st.number_input("Insert a latitude", value=47.1294, format="%f", placeholder="Type a number...")

st.write("The current number is ", lon, lat)

if st.button("Find parcelle"):
    if lon != None and lat != None :
        
        image_encode, image_contour_encode, polygone = contouring.prediction(lon, lat)

        st.pyplot(image_encode)
        st.pyplot(image_contour_encode)

        ndvi_curve, prairie_prediction, crop_prediction = classifier.prediction(polygone)

        st.markdown(f"### Prairie Classification : {prairie_prediction}")
        st.markdown(f"### Crop Classification : {crop_prediction}")

        st.pyplot(ndvi_curve)
    else : 
        st.write("No value for lon or lat ")
