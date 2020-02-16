# This script generates random coordinates in a region centered around San Francisco
# and uses reverse geocoding to get the addresses and write them to a CSV file.

from Modules.addresses import Addresses
import argparse

def get_random_addresses(csv_filename, api_key_filename, number_addresses)
    addresses = Addresses(api_key_filename)

    addresses.create_random_addresses(number_of_addresses=number_addresses)
    addresses.write_addresses_to_csv(csv_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--api_key_filename", type=str, default='api-key.txt',
        help = "The file location of a text file containing a Google API Key.  Default is 'api-key.txt'.")
    parser.add_argument(
        "--addresses_filename", type=str, default='Addresses/random_addresses.csv',
        help = "The file location to save the csv file containing the addresses.  Default is 'Addresses/random_addresses.csv")
    parser.add_argument(
        "--number_addresses", type=int, default=100,
        help = "The number of random addresses to generate.  Default is 100.")
    flags = parser.parse_args()
    api_key_filename = flags.api_key_filename
    csv_filename = flags.addresses_filename
    number_addresses = flags.number_addresses
    get_random_addresses(csv_filename, api_key_filename, number_addresses)