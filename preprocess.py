from matplotlib import pyplot as plt

import glob
import inspect
import sys
import os
import copy

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from constants import VIDEO_BASEPATH, CROP_DATA, OS_SEPARATOR, \
    CAMERAS, CAMERA_LIST, \
    DAY_START_UTC, DAY_END_UTC, \
    CIRCLE_START_X, CIRCLE_START_Y, CIRCLE_WIDTH, CIRCLE_HEIGHT, \
    TRIANGLE_START_X, TRIANGLE_START_Y, TRIANGLE_WIDTH, TRIANGLE_HEIGHT, \
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
    x__ = (x_ - color_size / 2) * np.ones((2, 2))
    x__[0, 1] = x_ + color_size / 2
    x__[1, 1] = x_ + color_size / 2
    y__ = (y_ - color_size / 2) * np.ones((2, 2))
    y__[1, 0] = y_ + color_size / 2
    y__[1, 1] = y_ + color_size / 2
    ax_.pcolormesh(x__, y__, np.ones((1, 1)), cmap='spring')


def test_cropped_image(img_cropped_, camera_name_, screen_size, image_filename_=None, show_figures=False):
    logo_start_x = CAMERAS[camera_name_]['logo_start_x']
    logo_start_y = CAMERAS[camera_name_]['logo_start_y']

    netivei_logo_width = CROP_DATA[screen_size]['netivei_logo_width']
    netivei_logo_height = CROP_DATA[screen_size]['netivei_logo_height']

    possible_logo = img_cropped_.crop(
        (logo_start_x, logo_start_y, logo_start_x + netivei_logo_width, logo_start_y + netivei_logo_height))

    possible_circle = img_cropped_.crop(
        (CIRCLE_START_X,
         CIRCLE_START_Y,
         CIRCLE_START_X + CIRCLE_WIDTH,
         CIRCLE_START_Y + CIRCLE_HEIGHT)).convert('L')

    possible_circle_bw = possible_circle.convert('L')

    possible_triangle_bw = img_cropped_.crop(
        (TRIANGLE_START_X,
         TRIANGLE_START_Y,
         TRIANGLE_START_X + TRIANGLE_WIDTH,
         TRIANGLE_START_Y + TRIANGLE_HEIGHT)).convert('L')

    top_right_corner = img_cropped_.crop(
        (CROP_DATA[screen_size]['crop_width'] - CROP_DATA[screen_size]['top_right_width'],
         0,
         CROP_DATA[screen_size]['crop_width'],
         CROP_DATA[screen_size]['top_right_height']))

    if 0:
        plt.figure()
        plt.imshow(img_cropped_)

        plt.figure()
        plt.imshow(possible_logo)

        plt.figure()
        plt.imshow(possible_circle_bw, cmap='gray')

        plt.figure()
        plt.imshow(top_right_corner)
        plt.title('Top Right')

        plt.show()

    # test logo
    if 1:
        image_ok_ = True
        possible_logo_bw = np.array(possible_logo.convert('L'))

        netivei_logo = Image.open(CROP_DATA[screen_size]['netivei_logo_filename'])
        netivei_logo_bw = np.array(netivei_logo.convert('L'))

        correlation = calc_correlation(netivei_logo_bw, possible_logo_bw)
        if correlation > 0.9:
            found_netivei_logo = True
        else:
            found_netivei_logo = False
            image_ok_ = False
        if 0:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(netivei_logo_bw, cmap='gray', vmin=0, vmax=255)
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
            waiting = True
            image_ok_ = False
        else:
            waiting = False

        if mean_color_circle < 20:
            error = True
            image_ok_ = False
        else:
            error = False

        if 0:
            plt.figure()
            plt.imshow(possible_circle_np, cmap='gray')
            plt.title(f'mean_color_circle is {mean_color_circle}')
            plt.show()

    # test triangle
    if 1:
        possible_triangle_np = np.array(possible_triangle_bw)
        sum_color_triangle = 0
        for x_, y_ in PLAY_TRIANGLE_PIXELS:
            sum_color_triangle += possible_triangle_np[x_, y_]
        mean_color_triangle = sum_color_triangle / len(PLAY_TRIANGLE_PIXELS)
        if mean_color_triangle > 160:
            paused_ = True
            image_ok_ = False
        else:
            paused_ = False

    # test time
    if 1:
        night = False
        if image_filename_ is not None:
            hour_utc = int(image_filename_.split(OS_SEPARATOR)[-1].split('_')[4])
            if hour_utc < DAY_START_UTC or hour_utc >= DAY_END_UTC:
                night = True
                image_ok_ = False
            else:
                night = False

    # test text in top right corner
    if 1:
        red = np.array(top_right_corner.getchannel('R'))
        green = np.array(top_right_corner.getchannel('G'))
        blue = np.array(top_right_corner.getchannel('B'))

        if 1:
            bands = top_right_corner.getbands()
            gray = np.array(top_right_corner.convert('L'))

            im_array = np.array(top_right_corner)
            if 'A' in bands:
                im_array = im_array[:, :, :3]
            top_right_norm_array = np.array(im_array).astype(float)

            for i in range(top_right_norm_array.shape[0]):
                for j in range(top_right_norm_array.shape[1]):
                    color_vec = im_array[i, j]
                    top_right_norm_array[i, j] = color_vec/np.linalg.norm(color_vec)

            ref_red = 235
            ref_green = 249
            ref_blue = 80
            ref_color = np.array((ref_red, ref_green, ref_blue))
            ref_color = ref_color/np.linalg.norm(ref_color)
            yellow_corr = np.dot(top_right_norm_array, ref_color)

            yellow_corr_threshold = copy.deepcopy(yellow_corr)

            yellow_corr_threshold[yellow_corr < 0.96] = 0
            yellow_corr_threshold[yellow_corr >= 0.96] = 1

        yellow = red.astype('int') + green.astype('int') - blue.astype('int')
        yellow_threshold = copy.deepcopy(yellow)
        yellow_threshold[yellow < 180] = 0
        yellow_threshold = np.round(yellow_threshold.astype(float) / np.max(yellow_threshold) * 255).astype('uint8')

        if 0:  # save as referenc3
            yellow_img = Image.fromarray(yellow)
            yellow_img.save('/home/user1/Dana Porrat/Netivei/Code/Raanana_Merkaz_130_12.png')

        if 0:
            plt.figure()
            ax0 = plt.subplot(511)
            ax0.imshow(top_right_corner)

            ax = plt.subplot(512, sharex=ax0, sharey=ax0)
            ax.imshow(yellow_corr, cmap='gray')
            ax.set_title('yellow_corr')

            ax = plt.subplot(513, sharex=ax0, sharey=ax0)
            ax.imshow(yellow_corr_threshold, cmap='gray')
            ax.set_title('yellow_corr threshold')

            ax = plt.subplot(514, sharex=ax0, sharey=ax0)
            ax.imshow(yellow, cmap='gray')
            ax.set_title('yellow2')

            ax = plt.subplot(515, sharex=ax0, sharey=ax0)
            ax.imshow(yellow_threshold, cmap='gray')
            ax.set_title('yellow2_threshold')

            plt.show()

        yellow_reference = np.array(Image.open(CAMERAS[camera_name_]['camera_name_file']))

        yellow_threshold[yellow_reference == 0] = 0
        if np.all(yellow_threshold == 0):
            yellow_covariance = 0
        else:
            yellow_covariance = calc_correlation(yellow_threshold, yellow_reference)
        if yellow_covariance > 0.9:
            camera_name_ok = True
        else:
            camera_name_ok = False
            image_ok_ = False

        if 0:
            plt.figure()
            plt.imshow(top_right_corner)

            plt.figure()
            ax0 = plt.subplot(311)
            ax0.imshow(yellow, cmap='gray')
            ax0.set_title('yellow')

            ax0 = plt.subplot(312)
            ax0.imshow(yellow_threshold, cmap='gray')
            ax0.set_title(f'yellow_threshold, covariance is {yellow_covariance:.2f}')

            ax = plt.subplot(313, sharex=ax0, sharey=ax0)
            ax.imshow(yellow_reference, cmap='gray')
            ax.set_title(f'yellow reference')

    # show image and cropping
    if show_figures:
        plt.figure()
        plt.imshow(img_cropped_)

        if image_ok_:
            title_str = 'Good image, save.'
        else:
            if found_netivei_logo:
                title_str = ''
            else:
                title_str = 'Logo missing.'

            if waiting:
                title_str += ' offline.'

            if paused_:
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
        if 1:
            plt.figure()
            plt.imshow(possible_triangle_bw, cmap='gray')
            plt.title(f'mean triangle color is {mean_color_triangle:.1f}')

            # draw triangle
            if 1:
                ax = plt.gca()
                color_pixel(ax, CENTER_PIXEL[0], CENTER_PIXEL[1])
                for x_, y_ in PLAY_TRIANGLE_PIXELS:
                    color_pixel(ax, x_, y_)

        # show top right corner
        if 1:
            plt.figure()
            plt.imshow(top_right_corner)
            plt.title('Top Right Corner')

        plt.show()
    image_status_ = {'paused': paused_,
                     'found_netivei_logo': found_netivei_logo,
                     'waiting': waiting,
                     'error': error,
                     'night': night,
                     'camera_name_ok': camera_name_ok,
                     }
    return image_ok_, image_status_


def preprocess_one_image(image_filename_, camera_name_, skip_time_test=False):
    this_function_name = inspect.currentframe().f_code.co_name

    try:
        img = Image.open(image_filename_)
    except UnidentifiedImageError:
        return None, None, None

    try:
        if img.size not in CROP_DATA.keys():
            raise ValueError(f'{this_function_name}: could not find data for screen of size {img.size}')
        img_cropped_ = img.crop((CROP_DATA[img.size]['crop_start_x'],
                                CROP_DATA[img.size]['crop_start_y'],
                                CROP_DATA[img.size]['crop_start_x'] + CROP_DATA[img.size]['crop_width'],
                                CROP_DATA[img.size]['crop_start_y'] + CROP_DATA[img.size]['crop_height']))
    except OSError:
        return None, None, None

    netivei_logo_filename = None
    if 0:
        netivei_logo_filename = CROP_DATA[img.size]['netivei_logo_filename']

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_cropped_, interpolation=None)
        plt.show()

    if skip_time_test:
        image_ok_, image_status_ = test_cropped_image(img_cropped_, camera_name_,
                                                      img.size)
    else:
        image_ok_, image_status_ = test_cropped_image(img_cropped_, camera_name_,
                                                      img.size,
                                                      image_filename_=image_filename_)
    if image_ok_ is None:
        return None, None, None
    else:
        return img_cropped_, image_ok_, image_status_


if __name__ == '__main__':
    this_script_name = os.path.basename(sys.argv[0])

    if 0:
        camera_name = 'Aluf_Sadeh'
        image_filename = VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'cropped_images' + OS_SEPARATOR + 'capture_2022_02_02_11_30_28_831763.png'
        camera_name = 'Raanana_Merkaz'
        image_filename = VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'cropped_images' + OS_SEPARATOR + 'capture_2022_02_02_11_10_07_503876.png'

        print(os.path.exists(image_filename))
        img_cropped = Image.open(image_filename)
        test_cropped_image(img_cropped, camera_name, show_figures=True)

    if 0:
        camera_name = 'Aluf_Sadeh'
        image_filename = VIDEO_BASEPATH + OS_SEPARATOR + camera_name + OS_SEPARATOR + 'capture_2022_02_02_12_17_32_733220.png'
        print(os.path.exists(image_filename))
        cropped_image, image_ok, image_status = preprocess_one_image(image_filename, camera_name)

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
        cropped_image, image_ok, image_status = preprocess_one_image(image_filename, camera_name)

    iiFile = 0
    while True:
        for iiCamera, Video_Path in enumerate(Video_Paths):
            camera_name = CAMERA_LIST[iiCamera]
            video_cropped_path = video_cropped_paths[iiCamera]
            images = glob.glob(Video_Path + OS_SEPARATOR + 'capture_*.png')
            for image_filename in images:
                if not os.path.exists(image_filename):
                    raise FileNotFoundError(f'cannot find {image_filename}')
                if 0:
                    image_filename = VIDEO_BASEPATH + OS_SEPARATOR + 'Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_01_26_12_55_33_762499.png'
                    image_filename = VIDEO_BASEPATH + OS_SEPARATOR + 'Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_01_26_12_56_03_865190.png'
                cropped_image, image_ok, image_status = preprocess_one_image(image_filename, camera_name)
                if cropped_image is not None:
                    if image_ok:
                        # print(f'File {image_filename} has a good image')
                        cropped_filename = video_cropped_path + OS_SEPARATOR + os.path.basename(image_filename)
                        cropped_image.save(cropped_filename)
                        # print(f'Saved file {cropped_filename}')
                os.remove(image_filename)
                print('.', end='')
                iiFile += 1
                if iiFile >= 30:
                    print(' ')
                    iiFile = 0
                # print(f'Deleted file {image_filename}')
        # time.sleep(10)
