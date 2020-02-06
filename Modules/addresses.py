import requests
import numpy as np
import csv
from PIL import Image
import io
from tensorflow.keras import preprocessing

class Addresses:
    def __init__(self, api_key_filename='/home/ubuntu/Insight/google_api_key.txt'):
        self.addresses = None

        with open(api_key_filename) as api_key_file:
            self.api_key = api_key_file.read()

    def get_image(self, address_index, image_size=299, image_pitch=10):
        try:
            address = self.addresses[address_index]
            begin_url = 'https://maps.googleapis.com/maps/api/streetview?source=outdoor'
            url = '{}&size={}x{}&pitch={}&key={}&location={}'.format(begin_url, image_size, image_size, image_pitch, self.api_key, address)
            response = requests.get(url)
            image = response.content
        except:
            print('Error getting image from address')
            image = None
        return image

    def get_address(self, latitude, lngitude):
        url = ('https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}'.format(latitude, lngitude, self.api_key))
        try:
            response = requests.get(url)
            resp_json_payload = response.json()
            address = resp_json_payload['results'][0]['formatted_address']
        except:
            print('Error getting address from latitude and longitude')
            address = None
        return address

    def create_random_addresses(self, number_of_addresses=100, min_latitude=37.631, max_latitude=37.8235, min_lngitude=-122.5209, max_lngitude=-122.173):
        self.addresses = [None]*number_of_addresses

        for address_index in range(number_of_addresses):
            for _ in range(100):
                latitude = np.random.uniform(min_latitude, max_latitude)
                lngitude = np.random.uniform(min_lngitude, max_lngitude)
                address = self.get_address(latitude, lngitude)
                #It is assumed to be a valid address if it starts with a number
                if not(address == None) and address[0].isnumeric():
                    break

            self.addresses[address_index] = address

    def read_addresses_from_csv(self, csv_filename, address_column=0):
        with open(csv_filename, 'w') as csv_file:
            reader = csv.reader(csv_file)
            row_number = 0

            for row in reader:
                self.addresses[row_number] = row[address_column]

    def write_addresses_to_csv(self, csv_filename):
        with open(csv_filename, 'w') as csv_file:
            writer = csv.writer(csv_file)

            for address in self.addresses:
                writer.writerow([address])