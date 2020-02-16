# This script takes addresses from a csv file and gets the
# associated street view image and places it in the specified folder.

# the image is saved as "row.jpg", where row is the row number
# from the csv file.

from Modules.addresses import Addresses
from Modules.imagefunctions import write_image_to_file
import os
import argparse

def get_soft_story_images(image_folder, csv_filename, address_column, start_index, end_index, api_key_filename)
    addresses = Addresses(api_key_filename)
    addresses.read_addresses_from_csv(csv_filename, address_column=address_column)

    for address_index in range(start_index, end_index):
        image = addresses.get_image(address_index)
        if image is None:
            continue
        write_image_to_file(image, os.path.join(image_folder, str(address_indeximage) + '.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--api_key_filename", type=str, default='api-key.txt',
        help = "The file location of a text file containing a Google API Key.  Default is 'api-key.txt'.")
    parser.add_argument(
        "--image_folder", type=str, default='Data',
        help = "The directory in which to save the labeled images.  Default is 'Data")
    parser.add_argument(
        "--addresses_filename", type=str, default='Addresses/Soft-Story-Properties.csv',
        help = "The file location of the csv file containing the addresses.  Default is 'Addresses/Soft-Story-Properties.csv")
    parser.add_argument(
        "--address_column", type=int, default=4,
        help = "The column in the csv file which corresponds to the addresses.  Default is 4.")
    parser.add_argument(
        "--start_row", type=int, default=0,
        help = "The row of the csv file to start at.  Default is 0.")
    parser.add_argument(
        "--end_row", type=int, default=5000,
        help = "The row of the csv file to end at.  Default is 5000.")
    flags = parser.parse_args()

    api_key_filename = flags.api_key_filename
    image_folder = flags.image_folder
    csv_filename = flags.addresses_filename
    address_column = flags.address_column
    start_index = flags.start_row
    end_index = flags.end_row
    get_soft_story_images(image_folder, csv_filename, address_column, start_index, end_index, api_key_filename)