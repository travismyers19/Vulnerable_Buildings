# This script takes addresses from a csv file and gets the
# associated street view image and displays it for the user.
# If the user presses 'y', the image is stored in the specified path
# in the folder "non_soft_story"; if the user presses 'u' the image
# is stored in the folder "bad_images"

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
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--api_key_filename", type=str, default='api-key.txt',
        help = "The file location of a text file containing a Google API Key.  Default is 'api-key.txt'.")
    parser.add_argument(
        "--image_folder", type=str, default='Data',
        help = "The directory in which to save the labeled images.  Default is 'Data")
    parser.add_argument(
        "--addresses_filename", type=str, default='Addresses/random_addresses.csv',
        help = "The file location of the csv file containing the addresses.  Default is 'Addresses/random_addresses.csv")
    parser.add_argument(
        "--address_column", type=int, default=0,
        help = "The column in the csv file which corresponds to the addresses.  Default is 0.")
    parser.add_argument(
        "--start_row", type=int, default=0,
        help = "The row of the csv file to start at.  Default is 0.")
    parser.add_argument(
        "--end_row", type=int, default=5000,
        help = "The row of the csv file to end at.  Default is 5000.")
    parser.add_argument(
        "--image_size", type=int, default=299,
        help = "The desired width and height of the images (the images will always be square images).  Default is 299.")
    parser.add_argument(
        "--image_pitch", type=float, default=10.,
        help = "The desired pitch of the image.  Default is 10.")
    flags = parser.parse_args()
    api_key_filename = flags.api_key-filename
    image_folder = flags.image_folder
    csv_filename = flags.addresses_filename
    address_column = flags.address_column
    image_size = flags.image_size
    image_pitch = flags.image_pitch
    start_row = flags.start_row
    end_row = flags.end_row
    with open(api_key_filename) as api_key_file:
            api_key = api_key_file.read()

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
            def on_press(event):
                plt.close()
                if event.key == 'y':
                    file_location = 
                    with open(os.path.join(image_folder, 'non_soft_story/', str(row_number) + '.jpg'), 'wb') as jpg_file:
                        jpg_file.write(image)
                elif event.key == 'u':
                    with open(os.path.join(image_folder, 'bad_images/', str(row_number) + '.jpg'), 'wb') as jpg_file:
                        jpg_file.write(image)
            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', on_press)
            ax.imshow(np_image)
            plt.show(block=True)