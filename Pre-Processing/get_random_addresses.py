# This script generates random coordinates in a region centered around San Francisco
# and uses reverse geocoding to get the addresses and write them to a CSV file.

import sys
sys.path.insert(1, '../Product/addresses.py')
from addresses import Addresses

def get_random_addresses(csv_filename)
    addresses = Addresses()

    addresses.create_random_addresses()
    addresses.write_addresses_to_csv(csv_filename)

if __name__ == '__main__':
    csv_filename = 'random_addresses.csv'