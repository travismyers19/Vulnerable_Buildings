# This script takes addresses from a csv file and gets the
# associated street view image and places it in the specified path
# in the folder "soft_story"

# the image is saved as "row.jpg", where row is the row number
# from the csv file

# When running the script, include the google api as the first argument

import sys
import io
import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv

image_folder = '../data/preprocessed/'
csv_filename = 'Soft-Story_Properties.csv'
address_column = 4
image_size = 299
image_pitch = 10
start_row = 0
end_row = 100
if len(sys.argv) < 2:
    print('Please provide a google api key.')
    sys.exit()
api_key = sys.argv[1]

def get_image(address):
    try:
        begin_url = 'https://maps.googleapis.com/maps/api/streetview?source=outdoor'
        url = '{}&size={}x{}&pitch={}&key={}&location={}'.format(begin_url, image_size, image_size, image_pitch, api_key, address)
        response = requests.get(url)
        image = response.content
    except:
        print('Error getting image')
        image = None
    return image

with open(csv_filename) as csv_file:
    reader = csv.reader(csv_file)
    row_number = 0

    for row in reader:
        row_number += 1
        if row_number < start_row:
            continue
        if row_number > end_row:
            break
        address = row[address_column]
        image = get_image(address)
        if image is None:
            continue
        np_image = Image.open(io.BytesIO(image))
        np_image = np.asarray(np_image)
        with open(image_folder + 'soft_story/' + str(row_number) + '.jpg', 'wb') as jpg_file:
            jpg_file.write(image)