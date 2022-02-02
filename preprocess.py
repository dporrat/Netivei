from matplotlib import pyplot as plt

import glob
import inspect
import sys
import os
import time

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from constants import VIDEO_BASEPATH, CROP_DATA, OS_SEPARATOR, \
    CAMERAS, CAMERA_LIST, \
    DAY_START_UTC, DAY_END_UTC, \
    NETIVEI_WIDTH, NETIVEI_HEIGHT, \
    CIRCLE_START_X, CIRCLE_START_Y, CIRCLE_WIDTH,  CIRCLE_HEIGHT, \
    TRIANGLE_START_X, TRIANGLE_START_Y, TRIANGLE_WIDTH, TRIANGLE_HEIGHT, \
    NETIVEI_LOGO_BW, \
    OFFLINE_CIRCLE_PIXELS, PLAY_TRIANGLE_PIXELS, CENTER_PIXEL


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


def test_cropped_image(img_cropped, camera_name_, image_filename_=None, show_figures=False):
    logo_start_x = CAMERAS[camera_name_]['logo_start_x']
    logo_start_y = CAMERAS[camera_name_]['logo_start_y']

    possible_logo = img_cropped.crop(
        (logo_start_x, logo_start_y, logo_start_x + NETIVEI_WIDTH, logo_start_y + NETIVEI_HEIGHT))

    possible_circle = img_cropped.crop(
        (CIRCLE_START_X,
         CIRCLE_START_Y,
         CIRCLE_START_X + CIRCLE_WIDTH,
         CIRCLE_START_Y + CIRCLE_HEIGHT)).convert('L')

    possible_circle_bw = possible_circle.convert('L')

    possible_triangle = img_cropped.crop(
        (TRIANGLE_START_X,
         TRIANGLE_START_Y,
         TRIANGLE_START_X + TRIANGLE_WIDTH,
         TRIANGLE_START_Y + TRIANGLE_HEIGHT)).convert('L')

    if 0:
        plt.figure()
        plt.imshow(possible_logo)
        plt.figure()
        plt.imshow(possible_circle_bw, cmap='gray')
        plt.show()

    # test logo
    if 1:
        image_ok = True
        possible_logo_bw = np.array(possible_logo.convert('L'))
        correlation = calc_correlation(NETIVEI_LOGO_BW, possible_logo_bw)
        if correlation > 0.9:
            found_netivei_logo = True
        else:
            found_netivei_logo = False
            image_ok = False
        if 0:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(NETIVEI_LOGO_BW, cmap='gray', vmin=0, vmax=255)
            ax[0].set_title('Saved logo')
            ax[1].imshow(possible_logo_bw, cmap='gray', vmin=0, vmax=255)
            ax[1].set_title(f'From Current Image, corrlation is {correlation}')

        # print(f'{this_function_name}: logo correlation is {correlation:.3f}')

    # test circle and error
    if 1:
        possible_circle_np = np.array(possible_circle_bw)
        sum_color_circle = 0
        for x_, y_ in OFFLINE_CIRCLE_PIXELS:
            sum_color_circle += possible_circle_np[x_, y_]
        mean_color_circle = sum_color_circle / len(OFFLINE_CIRCLE_PIXELS)
        if mean_color_circle > 160:
            offline = True
            image_ok = False
        else:
            offline = False

        if mean_color_circle < 20:
            error = True
            image_ok = False
        else:
            error = False

        if 0:
            plt.figure()
            plt.imshow(possible_circle_np, cmap='gray')
            plt.title(f'mean_color_circle is {mean_color_circle}')
            plt.show()

    # test triangle
    if 1:
        possible_triangle_np = np.array(possible_triangle)
        sum_color_triangle = 0
        for x_, y_ in PLAY_TRIANGLE_PIXELS:
            sum_color_triangle += possible_triangle_np[x_, y_]
        mean_color_triangle = sum_color_triangle / len(PLAY_TRIANGLE_PIXELS)
        if mean_color_triangle > 160:
            paused = True
            image_ok = False
        else:
            paused = False

    # test time
    if 1:
        night = False
        if image_filename_ is not None:
            hour_utc = int(image_filename_.split(OS_SEPARATOR)[-1].split('_')[4])
            if hour_utc < DAY_START_UTC or hour_utc >= DAY_END_UTC:
                night = True
                image_ok = False
            else:
                night = False

    # show image and cropping
    if show_figures:
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

            if night:
                title_str += ' night.'

        plt.title(title_str)

        # show circle in middle
        if 1:
            plt.figure()
            plt.imshow(possible_circle_bw, cmap='gray')
            plt.title(f'mean circle color is {mean_color_circle:.1f}')

            # draw circle
            if 1:
                ax = plt.gca()
                color_pixel(ax, CENTER_PIXEL[0], CENTER_PIXEL[1])
                for x_, y_ in OFFLINE_CIRCLE_PIXELS:
                    color_pixel(ax, x_, y_)

        # show triangle in middle
        if show_figures:
            plt.figure()
            plt.imshow(possible_triangle, cmap='gray')
            plt.title(f'mean triangle color is {mean_color_triangle:.1f}')

            # draw triangle
            if 1:
                ax = plt.gca()
                color_pixel(ax, CENTER_PIXEL[0], CENTER_PIXEL[1])
                for x_, y_ in PLAY_TRIANGLE_PIXELS:
                    color_pixel(ax, x_, y_)

        plt.show()
    return image_ok, paused


def preprocess_one_image(image_filename_, camera_name_):
    this_function_name = inspect.currentframe().f_code.co_name

    try:
        img = Image.open(image_filename_)
    except UnidentifiedImageError:
        return None, None, None

    try:
        if img.size not in CROP_DATA.keys():
            raise ValueError(f'{this_function_name}: could not find data from screen of size {img.size}')
        img_cropped = img.crop((CROP_DATA[img.size]['crop_start_x'],
                                CROP_DATA[img.size]['crop_start_y'],
                                CROP_DATA[img.size]['crop_start_x'] + CROP_DATA[img.size]['crop_width'],
                                CROP_DATA[img.size]['crop_start_y'] + CROP_DATA[img.size]['crop_height']))
    except OSError:
        return None, None, None

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_cropped, interpolation=None)
        plt.show()

    image_ok, paused = test_cropped_image(img_cropped, camera_name_, image_filename_=image_filename_)
    if image_ok is None:
        return None, None, None
    else:
        return img_cropped, image_ok, paused


if __name__ == '__main__':
    this_script_name = os.path.basename(sys.argv[0])
    if 0:
        img_cropped = Image.open(r"C:\Users\dporrat\Desktop\Netivei\videos\Aluf_Sadeh\cropped_images\capture_2022_02_02_11_30_28_831763.png")
        camera_name_ = 'Aluf_Sadeh'
        test_cropped_image(img_cropped, camera_name_, show_figures=True)

    if 0:
        camera_name = 'Aluf_Sadeh'
        image_filename = VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'capture_2022_02_02_12_17_32_733220.png'
        print(os.path.exists(image_filename))
        cropped_image, image_ok, paused = preprocess_one_image(image_filename, camera_name)

    video_input_path = VIDEO_BASEPATH
    video_cropped_paths = []
    for camera_name in list(CAMERA_LIST):
        video_cropped_paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'cropped_images')
        if not os.path.exists(video_cropped_paths[-1]):
            os.mkdir(video_cropped_paths[-1])

    Video_Paths = []
    for camera_name in CAMERA_LIST:
        Video_Paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name)

    # debug
    if 0:
        image_filename = '/media/dana/second local disk1/dana/Netivei/videos/Raanana_Merkaz/capture_2022_01_19_13_49_16_473626.png'
        cropped_image, image_ok, paused = preprocess_one_image(image_filename, camera_name)

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
                cropped_image, stam, stam = preprocess_one_image(image_filename, camera_name)
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
