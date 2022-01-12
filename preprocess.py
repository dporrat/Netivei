import glob
import sys
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

this_script_name = os.path.basename(sys.argv[0])


def calc_correlation(array1, array2):
    m1 = np.mean(array1.astype(float))
    m2 = np.mean(array2.astype(float))
    c11 = np.sum(np.multiply((array1.astype(float)-m1), (array1.astype(float)-m1)))
    c22 = np.sum(np.multiply((array2.astype(float)-m2), (array2.astype(float)-m2)))
    c12 = np.sum(np.multiply((array1.astype(float)-m1), (array2.astype(float)-m2)))
    return c12 / np.sqrt(c11) / np.sqrt(c22)


VIDEO_PATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
logo_filename = f'{VIDEO_PATH}\\netivei_logo.png'

images = glob.glob(f'{VIDEO_PATH}\\capture_*.png')

for image in images:
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

    if 1:  # test logo
        netiveiLogo = Image.open(logo_filename)
        netiveiLogo_BW = np.array(netiveiLogo.convert('L'))
        possibleLogo_BW = np.array(possible_logo.convert('L'))
        correlation = calc_correlation(netiveiLogo_BW, possibleLogo_BW)

        print(f'{this_script_name}: logo correlation is {correlation:.3f}')

        if 1:  # show image and cropping
            plt.figure()
            plt.imshow(img)
            plt.figure()
            plt.imshow(img_cropped)
            plt.figure()
            plt.imshow(possible_logo)

            plt.show()
plt.show()
