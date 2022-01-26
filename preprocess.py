import glob
import sys
import os
import time

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
from matplotlib import pyplot as plt

from constants import VIDEO_BASEPATH, SCREEN_SIZE, OS_SEPARATOR, \
    CAMERAS, CAMERA_LIST, \
    CROP_START_X, CROP_START_Y, CROP_WIDTH, CROP_HEIGHT

RAD_PER_DEG = np.pi / 180

this_script_name = os.path.basename(sys.argv[0])
video_input_path = VIDEO_BASEPATH
video_cropped_paths = []
for camera_name in list(CAMERAS.index):
    video_cropped_paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'cropped_images')
    if not os.path.exists(video_cropped_paths[-1]):
        os.mkdir(video_cropped_paths[-1])


def calc_correlation(array1, array2):
    m1 = np.mean(array1.astype(float))
    m2 = np.mean(array2.astype(float))
    c11 = np.sum(np.multiply((array1.astype(float) - m1), (array1.astype(float) - m1)))
    c22 = np.sum(np.multiply((array2.astype(float) - m2), (array2.astype(float) - m2)))
    c12 = np.sum(np.multiply((array1.astype(float) - m1), (array2.astype(float) - m2)))
    return c12 / np.sqrt(c11) / np.sqrt(c22)


def color_pixel(ax_, x_, y_):
    color_size = 0.6
    x__ = (x_ - color_size/2) * np.ones((2, 2))
    x__[0, 1] = x_ + color_size/2
    x__[1, 1] = x_ + color_size/2
    y__ = (y_ - color_size/2) * np.ones((2, 2))
    y__[1, 0] = y_ + color_size/2
    y__[1, 1] = y_ + color_size/2
    ax_.pcolormesh(x__, y__, np.ones((1, 1)), cmap='spring')


def preprocess_one_image(image_filename_, camera_name_):
    logo_start_x = CAMERAS.loc[camera_name_, 'logo_start_x']
    logo_start_y = CAMERAS.loc[camera_name_, 'logo_start_y']

    try:
        img = Image.open(image_filename_)
    except UnidentifiedImageError:
        return None
    if not img.size == SCREEN_SIZE:
        raise ValueError(f'{this_script_name}: image size is not (1280, 1024)!')

    try:
        img_cropped = img.crop((CROP_START_X,
                                CROP_START_Y,
                                CROP_START_X + CROP_WIDTH,
                                CROP_START_Y + CROP_HEIGHT))
    except OSError:
        return None

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_cropped)
        plt.show()

    possible_logo = img_cropped.crop(
        (logo_start_x, logo_start_y, logo_start_x + netivei_width, logo_start_y + netivei_height))

    possible_circle = img_cropped.crop(
        (circle_start_x,
         circle_start_y,
         circle_start_x + circle_width,
         circle_start_y + circle_height)).convert('L')

    possible_circle = possible_circle.convert('L')

    possible_triangle = img_cropped.crop(
        (triangle_start_x,
         triangle_start_y,
         triangle_start_x + triangle_width,
         triangle_start_y + triangle_height)).convert('L')

    possible_triangle = possible_triangle.convert('L')

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(possible_logo)
        plt.figure()
        plt.imshow(possible_circle)
        plt.show()

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

        # print(f'{this_script_name}: logo correlation is {correlation:.3f}')

    # test circle and error
    if 1:
        possible_circle_np = np.array(possible_circle)
        sum_color_circle = 0
        for x_, y_ in offline_circle_pixels:
            sum_color_circle += possible_circle_np[x_, y_]
        mean_color_circle = sum_color_circle / len(offline_circle_pixels)
        if mean_color_circle > 185:
            offline = True
            image_ok = False
        else:
            offline = False

        if mean_color_circle < 60:
            error = True
            image_ok = False
        else:
            error = False

    # test triangle
    if 1:
        possible_triangle_np = np.array(possible_triangle)
        sum_color_triangle = 0
        for x_, y_ in play_triangle_pixels:
            sum_color_triangle += possible_triangle_np[x_, y_]
        mean_color_triangle = sum_color_triangle / len(play_triangle_pixels)
        if mean_color_triangle > 185:
            paused = True
            image_ok = False
        else:
            paused = False

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

            if paused:
                title_str += ' paused.'

            if error:
                title_str += ' error.'

        plt.title(title_str)

        # show circle in middle
        if 0:
            plt.figure()
            plt.imshow(possible_circle, cmap='gray')
            plt.title(f'mean circle color is {mean_color_circle:.1f}')

            # draw circle
            if 0:
                ax = plt.gca()
                color_pixel(ax, center[0], center[1])
                for x_, y_ in offline_circle_pixels:
                    color_pixel(ax, x_, y_)

        # show triangle in middle
        if 1:
            plt.figure()
            plt.imshow(possible_triangle, cmap='gray')

            # draw triangle
            if 1:
                ax = plt.gca()
                color_pixel(ax, center[0], center[1])
                for x_, y_ in play_triangle_pixels:
                    color_pixel(ax, x_, y_)

        plt.show()

    if image_ok:
        return img_cropped
    else:
        return None


# preparations
if 1:
    # Netivei logo
    if 1:
        # logo_start_x = CAMERAS.loc[camera_name, 'logo_start_x']
        # logo_start_y = CAMERAS.loc[camera_name, 'logo_start_y']
        netivei_width = 177
        netivei_height = 49
        logo_filename = f'netivei_logo.png'
        netiveiLogo = Image.open(logo_filename)
        netiveiLogo_BW = np.array(netiveiLogo.convert('L'))

    # offline wait circle
    if 1:
        circle_start_x = 363
        circle_start_y = 186
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

    # play triangle
    if 1:
        triangle_start_x = 365
        triangle_start_y = 186
        triangle_width = 80
        triangle_height = 80
        play_triangle_pixels = []
        for x in range(7, 81):
            for y in np.arange(int(x/2)-1, 78-int(x/2)):
                if (x, y) not in play_triangle_pixels:
                    play_triangle_pixels.append((x, y))

if __name__ == '__main__':
    Video_Paths = []
    for camera_name in CAMERA_LIST:
        Video_Paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name)

    # debug
    if 0:
        image_filename = '/media/dana/second local disk1/dana/Netivei/videos/Raanana_Merkaz/capture_2022_01_19_13_49_16_473626.png'
        cropped_image = preprocess_one_image(image_filename, camera_name)

    while True:
        for iiCamera, Video_Path in enumerate(Video_Paths):
            camera_name = CAMERA_LIST[iiCamera]
            video_cropped_path = video_cropped_paths[iiCamera]
            images = glob.glob(Video_Path + OS_SEPARATOR + 'capture_*.png')
            iiFile = 0
            for image_filename in images:
                if not os.path.exists(image_filename):
                    raise FileNotFoundError(f'cannot find {image_filename}')
                if 0:
                    image_filename = VIDEO_BASEPATH + OS_SEPARATOR + 'Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_01_26_12_55_33_762499.png'
                    image_filename = VIDEO_BASEPATH + OS_SEPARATOR + 'Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_01_26_12_56_03_865190.png'
                cropped_image = preprocess_one_image(image_filename, camera_name)
                if cropped_image is not None:
                    # print(f'File {image_filename} has a good image')
                    cropped_filename = video_cropped_path + OS_SEPARATOR + os.path.basename(image_filename)
                    cropped_image.save(cropped_filename)
                    # print(f'Saved file {cropped_filename}')
                os.remove(image_filename)
                print('.', end='')
                iiFile += 1
                if iiFile == 30:
                    print('')
                    iiFile = 0
                # print(f'Deleted file {image_filename}')
        print(' ')
        time.sleep(10)
