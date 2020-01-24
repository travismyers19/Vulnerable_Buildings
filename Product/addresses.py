import requests
import numpy as np
import csv

class Addresses:
    def __init__(self, api_key):
        self.addresses = None
        self.api_key = api_key

    def get_image(self, address_index, image_size=299, image_pitch=10):
        try:
            address = self.addresses[address_index]
            begin_url = 'https://maps.googleapis.com/maps/api/streetview?source=outdoor'
            url = '{}&size={}x{}&pitch={}&key={}&location={}'.format(begin_url, image_size, image_size, image_pitch, self.api_key, address)
            response = requests.get(url)
            image = response.content
        except:
            print('Error getting image')
            image = None
        return image

    def get_address(self, latitude, lngitude):
        url = ('https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}'.format(latitude, lngitude, self.api_key))
        try:
            response = requests.get(url)
            resp_json_payload = response.json()
            address = resp_json_payload['results'][0]['formatted_address']
        except:
            print('ERROR')
            address = None
        return address

    def create_random_addresses(self, number_of_addresses=100, min_latitude=37.631, max_latitude=37.8235, min_lngitude=-122.5209, max_lngitude=-122.173):
        self.addresses = [None]*len(number_of_addresses)

        for address_index in range(number_of_addresses):
            while True:
                latitude = np.random.uniform(min_latitude, max_latitude)
                lngitude = np.random.uniform(min_lngitude, max_lngitude)
                address = self.get_address(latitude, lngitude)
                #It is assumed to be a valid address if it starts with a number
                if not(address == None) and address[0].isnumeric():
                    break

            self.addresses[address_index] = address

    def write_addresses_to_csv(self, csv_filename):
        with open(csv_filename) as file:
            writer = csv.writer(file)

            for address in self.addresses:
                writer.writerow([address])