from Modules.addresses import Addresses
import requests
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras import preprocessing
import numpy as np
import pandas as pd
import io
from PIL import Image
from Modules.buildingclassifier import BuildingClassifier
from Modules.imagefunctions import load_image_for_prediction
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    #If ternary classifier, set model2_filename=None
    #If binary classifier, model_filename = good vs bad model, model2_filename = soft vs non-soft model
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--api_key_filename", type=str, default='api-key.txt',
        help = "The file location of a text file containing a Google API Key.  Default is 'api-key.txt'.")
    parser.add_argument(
        "--model_filename", type=str, default='Models/trained_model.h5',
        help = "The file location of the model to serve.  Default is 'Models/trained_model.h5'.")
    parser.add_argument(
        "--model2_filename", type=str, default='None',
        help = "If using a binary classifier, specify the file location of the second model.  If using a ternary classifier, set to 'None'.  Default is 'None'.")
    flags = parser.parse_args()
    api_key_filename = flags.api_key_filename
    model_filename = flags.model_filename
    model2_filename = flags.model2_filename
    if model2_filename == 'None':
        model2_filename = None
    image_classifier = BuildingClassifier(model_filename=model_filename, model2_filename=model2_filename)
    
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
        addresses = Addresses(api_key_filename)
        number_of_soft_story = 0
        number_of_non_soft_story = 0
        addresses_used = 0

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