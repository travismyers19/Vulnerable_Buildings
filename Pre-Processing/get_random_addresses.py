# This script generates random coordinates in a region centered around San Francisco
# and uses reverse geocoding to get the addresses and write them to a CSV file
# When running the script, include the google api key as the first argument

import csv
import numpy as np
import sys
import requests

number_of_addresses = 100
min_latitude = 37.631
min_lngitude = -122.5209
max_latitude = 37.8235
max_lngitude = -122.173
csv_file = open('random_addresses.csv', 'w')
if len(sys.argv) < 2:
    print('Please provide a google api key.')
    sys.exit()
api_key = sys.argv[1]

def get_address(api_key, latitude, longitude):
    url = ('https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}'.format(latitude, longitude, api_key))
    try:
        response = requests.get(url)
        resp_json_payload = response.json()
        address = resp_json_payload['results'][0]['formatted_address']
    except:
        print('ERROR')
        address = None
    return address

with csv_file:
    writer = csv.writer(csv_file)

    for _ in range(number_of_addresses):
        while True:
            latitude = np.random.uniform(min_latitude, max_latitude)
            lngitude = np.random.uniform(min_lngitude, max_lngitude)
            address = get_address(api_key, latitude, lngitude)
            #It is assumed to be a valid address if it starts with a number
            if not(address == None) and address[0].isnumeric():
                break

        writer.writerow([address])

csv_file.close()