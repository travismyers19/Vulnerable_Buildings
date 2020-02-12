# This script takes addresses from a csv file and gets the
# associated street view image and places it in the specified folder.

# the image is saved as "row.jpg", where row is the row number
# from the csv file.

from addresses import Addresses
from addresses import write_image_to_file

def get_soft_story_images(image_folder, csv_filename, address_column, start_index, end_index)
    addresses = Addresses()
    addresses.read_addresses_from_csv(csv_filename, address_column=address_column)

    for address_index in range(start_index, end_index):
        image = addresses.get_image(address_index)
        if image is None:
            continue
        write_image_to_file(image, image_folder + str(address_indeximage) + '.jpg')

if __name__ == '__main__':
    image_folder = ''
    csv_filename = 'Soft-Story_Properties.csv'
    address_column = 4
    start_index = 0
    end_index = 200
    get_soft_story_images(image_folder, csv_filename, address_column, start_index, end_index)