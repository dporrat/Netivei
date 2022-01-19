import glob
import sys
import os
import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from constants import VIDEO_PATH, SCREEN_SIZE


RAD_PER_DEG = np.pi / 180

this_script_name = os.path.basename(sys.argv[0])
video_input_path = VIDEO_PATH
video_cropped_path = VIDEO_PATH + f'\\cropped_images'


def calc_correlation(array1, array2):
    m1 = np.mean(array1.astype(float))
    m2 = np.mean(array2.astype(float))
    c11 = np.sum(np.multiply((array1.astype(float)-m1), (array1.astype(float)-m1)))
    c22 = np.sum(np.multiply((array2.astype(float)-m2), (array2.astype(float)-m2)))
    c12 = np.sum(np.multiply((array1.astype(float)-m1), (array2.astype(float)-m2)))
    return c12 / np.sqrt(c11) / np.sqrt(c22)


def color_pixel(ax_, x_, y_):
    x__ = (x_ - 0.5) * np.ones((2, 2))
    x__[0, 1] = (x_ - 0.5) + 1
    x__[1, 1] = (x_ - 0.5) + 1
    y__ = (y_ - 0.5) * np.ones((2, 2))
    y__[1, 0] = (y_ - 0.5) + 1
    y__[1, 1] = (y_ - 0.5) + 1
    ax_.pcolormesh(x__, y__, np.ones((1, 1)), cmap='spring')


# crop location
if 1:
    # on desktop captures (1280, 1024) corner is (276, 298)
    crop_start_x = 276
    crop_start_y = 298
    crop_width = 800
    crop_height = 450

# Netivei logo
if 1:
    netivei_start_x = 285
    netivei_start_y = 314
    netivei_width = 177
    netivei_height = 49
    logo_filename = f'netivei_logo.png'
    netiveiLogo = Image.open(logo_filename)
    netiveiLogo_BW = np.array(netiveiLogo.convert('L'))

# offline wait circle
if 1:
    circle_start_x = crop_start_x + 363
    circle_start_y = crop_start_y + 186
    circle_width = 75
    circle_height = 75
    offline_circle_pixels = []
    center = (37, 37)
    delta_theta_deg = 0.5
    for radius in [32.5, 33.5, 34.5]:
        for theta in np.arange(0, 360, delta_theta_deg):
            x = round(center[0] + radius * np.cos(theta * RAD_PER_DEG))
            y = round(center[0] + radius * np.sin(theta * RAD_PER_DEG))
            if (x, y) not in offline_circle_pixels:
                offline_circle_pixels.append((x, y))


def preprocess_one_image(image_filename_):
    img = Image.open(image_filename_)
    if not img.size == SCREEN_SIZE:
        raise ValueError(f'{this_script_name}: image size is not (1280, 1024)!')

    img_cropped = img.crop((crop_start_x, crop_start_y, crop_start_x+crop_width, crop_start_y+crop_height))

    possible_logo = img.crop((netivei_start_x, netivei_start_y, netivei_start_x+netivei_width, netivei_start_y+netivei_height))

    possible_circle = img.crop(
        (circle_start_x, circle_start_y, circle_start_x + circle_width, circle_start_y + circle_height)).convert('L')

    # test logo
    if 1:
        image_ok = True
        possible_logo_bw = np.array(possible_logo.convert('L'))
        correlation = calc_correlation(netiveiLogo_BW, possible_logo_bw)
        if correlation > 0.9:
            found_netivei_logo = True
        else:
            found_netivei_logo = False
            image_ok = False

        print(f'{this_script_name}: logo correlation is {correlation:.3f}')

    # test circle and error
    if 1:
        possible_circle_np = np.array(possible_circle)
        sum_color = 0
        for x, y in offline_circle_pixels:
            sum_color += possible_circle_np[x, y]
        mean_color = sum_color / len(offline_circle_pixels)
        if mean_color > 199:
            offline = True
            image_ok = False
        else:
            offline = False

        if mean_color < 60:
            error = True
            image_ok = False
        else:
            error = False

    # show image and cropping
    if 0:
        plt.figure()
        plt.imshow(img_cropped)

        if image_ok:
            title_str = 'Good image, save.'
        else:
            if found_netivei_logo:
                title_str = ''
            else:
                title_str = 'Logo missing.'

            if offline:
                title_str += ' offline.'

            if error:
                title_str += ' error.'

        plt.title(title_str)

        # show cropping in middle
        if 0:
            plt.figure()
            plt.imshow(possible_circle, cmap='gray')
            plt.title(f'mean circle color is {mean_color:.1f}')

            # draw circle
            if 0:
                ax = plt.gca()
                color_pixel(ax, center[0],  center[1])
                for x, y in offline_circle_pixels:
                    color_pixel(ax, x, y)

        plt.show()

    if image_ok:
        return img_cropped
    else:
        return None


if __name__ == '__main__':
    while True:
        images = glob.glob(f'{video_input_path}\\capture_*.png')
        for image_filename in images[0:]:
            cropped_image = preprocess_one_image(image_filename)
            if cropped_image is not None:
                print(f'File {image_filename} has a good image')
                cropped_filename = video_cropped_path + f'\\' + os.path.basename(image_filename)
                cropped_image.save(cropped_filename)
                print(f'Saved file {cropped_filename}')
            os.remove(image_filename)
        time.sleep(60)


