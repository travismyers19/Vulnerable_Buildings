from addresses import Addresses
import requests
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras import preprocessing
import numpy as np
import pandas as pd
import io
from PIL import Image
from buildingclassifier import BuildingClassifier
from imagefunctions import load_image_for_prediction
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #If ternary classifier, set model2_filename=None
    #If binary classifier, model_filename = good vs bad model, model2_filename = soft vs non-soft model
    image_classifier = BuildingClassifier(model_filename='Models/trained_test_model.h5', model2_filename=None)
    
    st.title('Detect Soft-Story Buildings')
    header = st.empty()
    header.header('Enter the coordinates of the region to sample:')
    slots = [st.empty() for _ in range(5)]
    try:
        min_latitude = float(slots[0].text_input("Min Latitude", "37.631"))
        max_latitude = float(slots[1].text_input("Max Latitude", "37.8235"))
        min_lngitude = float(slots[2].text_input("Min Longitude", "-122.5209"))
        max_lngitude = float(slots[3].text_input("Max Longitude", "-122.173"))
        number_of_addresses = int(slots[4].text_input("Number of Addresses", "10"))
    except:
        st.write('Error in getting latitude and longitude and number of addresses')
        min_latitude = 37.631
        max_latitude = 37.8235
        min_lngitude = -122.5209
        max_lngitude = -122.173
        number_of_addresses = 10

    button = st.empty()

    if button.button('Enter'):
        button.empty()
        header.empty()
        for slot in slots:
            slot.empty()
        
        progress_text = st.empty()
        progress_text.text('Loading Model')
        progress_bar = st.empty()
        progress_bar = st.progress(0)
        addresses = Addresses()
        number_of_soft_story = 0
        number_of_non_soft_story = 0
        addresses_used = 0

<<<<<<< HEAD
        while addresses_used < number_of_addresses:
            progress_bar.progress(addresses_used/number_of_addresses)
            addresses.create_random_addresses(number_of_addresses=1, min_latitude=min_latitude, max_latitude=max_latitude, min_lngitude=min_lngitude, max_lngitude=max_lngitude)
            image = addresses.get_image(0)
            image = np.expand_dims(preprocessing.image.img_to_array(Image.open(io.BytesIO(image))), axis=0)
            if image_classifier.is_no_image(image):
                continue
            image = image/255.
            result = image_classifier.classify_image(image)
            progress_text.text('Processing')
            if result == 0:
                continue
            addresses_used += 1
            if result == 1:
                number_of_non_soft_story += 1
            if result == 2:
                number_of_soft_story += 1
=======
if button.button('Enter'):
    button.empty()
    for slot in slots:
        slot.empty()
    
    progress_text = st.empty()
    progress_text.text('Processing')
    progress_bar = st.empty()
    progress_bar = st.progress(0)
    model = load_model(model_filename)
    addresses = Addresses()
    addresses.create_random_addresses(number_of_addresses=number_of_addresses, min_latitude=min_latitude, max_latitude=max_latitude, min_lngitude=min_lngitude, max_lngitude=max_lngitude)
    number_of_soft_story = 0
    number_of_non_soft_story = 0
    number_of_bad_images = 0
    
    for address_index in range(number_of_addresses):
        progress_bar.progress(address_index/number_of_addresses)
        image = addresses.get_image(address_index)
        image = np.expand_dims(preprocessing.image.img_to_array(Image.open(io.BytesIO(image))), axis=0)
        image = image/255
        result = np.argmax(model.predict(image))
        if result == 0:
            number_of_bad_images += 1
        elif result == 1:
            number_of_non_soft_story += 1
        else:
            number_of_soft_story += 1
>>>>>>> 6ba2785555e39efda145a8fe1006732a213805e7

        my_circle=plt.Circle( (0,0), 0.7, color='white')
        plt.pie([number_of_soft_story, number_of_non_soft_story], labels=["Soft-Story: " + str(number_of_soft_story), "Non-Soft-Story: " + str(number_of_non_soft_story)], colors=['blue','green'])
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        st.pyplot()
        progress_text.empty()
        progress_bar.empty()
        header.header('Results')
        df = pd.DataFrame([
            [min_latitude, min_lngitude],
            [min_latitude, max_lngitude],
            [max_latitude, min_lngitude],
            [max_latitude, max_lngitude]],
            columns=['lat', 'lon'])
        st.map(df)