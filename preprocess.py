import glob
import sys
import os

from PIL import Image
from matplotlib import pyplot as plt

this_script_name = os.path.basename(sys.argv[0])

VIDEO_PATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
logo_filename = f'{VIDEO_PATH}\\netivei_logo.png'

images = glob.glob(f'{VIDEO_PATH}\\capture_*.png')

for image in images[1:]:
    img = Image.open(image)
    if not img.size == (1280, 1024):
        raise ValueError(f'{this_script_name}: image size is not (1280, 1024)!')

    # on desktop captures (1280, 1024) corner is (276, 298)
    crop_start_x = 276
    crop_start_y = 298
    crop_width = 800
    crop_height = 450
    img_cropped = img.crop((crop_start_x, crop_start_y, crop_start_x+crop_width, crop_start_y+crop_height))

    netivei_start_x = 285
    netivei_start_y = 314
    netivei_width = 177
    netivei_height = 49
    possible_logo = img.crop((netivei_start_x, netivei_start_y, netivei_start_x+netivei_width, netivei_start_y+netivei_height))

    if 0:  # show image and cropping
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_cropped)
        plt.figure()
        plt.imshow(possible_logo)

        plt.show()

    if 0:  # save logo
        possible_logo.save(logo_filename)

    if 1:  # test logo
        netiveiLogo = Image.open(logo_filename)
        netiveiLogo_BW = netiveiLogo.convert('L')
        possibleLogo_BW = possible_logo.convert('L')

plt.show()
