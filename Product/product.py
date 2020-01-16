from addresses import Addresses
import requests
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras import preprocessing
import numpy as np
import pandas as pd
import io
from PIL import Image

model_filename = '/home/ubuntu/Insight/Vulnerable_Buildings/Models/inception_model_trained.h5'
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
        print(image.shape)
        result = np.argmax(model.predict(image))
        if result == 0:
            number_of_bad_images += 1
        elif result == 1:
            number_of_non_soft_story += 1
        else:
            number_of_soft_story += 1

    progress_text.empty()
    progress_bar.empty()
    st.write('Number of soft story buildings: ' + str(number_of_soft_story))
    st.write('Number of non-soft story buildings: ' + str(number_of_non_soft_story))
    st.write('Number of bad images: ' + str(number_of_bad_images))
    df = pd.DataFrame([
        [min_latitude, min_lngitude],
        [min_latitude, max_lngitude],
        [max_latitude, min_lngitude],
        [max_latitude, max_lngitude]],
        columns=['lat', 'lon'])
    st.map(df)