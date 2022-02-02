from matplotlib import pyplot as plt

if 1:
    import glob
    import inspect
    import sys
    import os
    import time

    import numpy as np
    from PIL import Image
    from PIL import UnidentifiedImageError

if 1:
    from constants import VIDEO_BASEPATH, CROP_DATA, OS_SEPARATOR, \
        CAMERAS, CAMERA_LIST, \
        DAY_START_UTC, DAY_END_UTC, \
        NETIVEI_WIDTH, NETIVEI_HEIGHT, \
        CIRCLE_START_X, CIRCLE_START_Y, CIRCLE_WIDTH,  CIRCLE_HEIGHT, \
        TRIANGLE_START_X, TRIANGLE_START_Y, TRIANGLE_WIDTH, TRIANGLE_HEIGHT, \
        NETIVEI_LOGO_BW, \
        OFFLINE_CIRCLE_PIXELS, PLAY_TRIANGLE_PIXELS


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
    this_function_name = inspect.currentframe().f_code.co_name

    logo_start_x = CAMERAS[camera_name_]['logo_start_x']
    logo_start_y = CAMERAS[camera_name_]['logo_start_y']

    try:
        img = Image.open(image_filename_)
    except UnidentifiedImageError:
        return None, None
    # if not img.size == SCREEN_SIZE:
    #     raise ValueError(f'{this_function_name}: image size is not (1280, 1024)!')
    # if img.size not in CROP_DATA.index.Values:
    #     raise ValueError(f'{this_function_name}: image size {img.size} not found in list')

    try:
        if img.size not in CROP_DATA.keys():
            raise ValueError(f'{this_function_name}: could not find data from screen of size {img.size}')
        img_cropped = img.crop((CROP_DATA[img.size]['crop_start_x'],
                                CROP_DATA[img.size]['crop_start_y'],
                                CROP_DATA[img.size]['crop_start_x'] + CROP_DATA[img.size]['crop_width'],
                                CROP_DATA[img.size]['crop_start_y'] + CROP_DATA[img.size]['crop_height']))
    except OSError:
        return None, None

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_cropped, interpolation=None)
        plt.show()

    possible_logo = img_cropped.crop(
        (logo_start_x, logo_start_y, logo_start_x + NETIVEI_WIDTH, logo_start_y + NETIVEI_HEIGHT))

    possible_circle = img_cropped.crop(
        (CIRCLE_START_X,
         CIRCLE_START_Y,
         CIRCLE_START_X + CIRCLE_WIDTH,
         CIRCLE_START_Y + CIRCLE_HEIGHT)).convert('L')

    possible_circle = possible_circle.convert('L')

    possible_triangle = img_cropped.crop(
        (TRIANGLE_START_X,
         TRIANGLE_START_Y,
         TRIANGLE_START_X + TRIANGLE_WIDTH,
         TRIANGLE_START_Y + TRIANGLE_HEIGHT)).convert('L')

    if 0:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(possible_logo)
        plt.figure()
        plt.imshow(possible_circle, cmap='gray')
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
        possible_circle_np = np.array(possible_circle)
        sum_color_circle = 0
        for x_, y_ in OFFLINE_CIRCLE_PIXELS:
            sum_color_circle += possible_circle_np[x_, y_]
        mean_color_circle = sum_color_circle / len(OFFLINE_CIRCLE_PIXELS)
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
        for x_, y_ in PLAY_TRIANGLE_PIXELS:
            sum_color_triangle += possible_triangle_np[x_, y_]
        mean_color_triangle = sum_color_triangle / len(PLAY_TRIANGLE_PIXELS)
        if mean_color_triangle > 185:
            paused = True
            image_ok = False
        else:
            paused = False

    # test time
    if 1:
        hour_utc = int(image_filename_.split(OS_SEPARATOR)[-1].split('_')[4])
        if hour_utc < DAY_START_UTC or hour_utc >= DAY_END_UTC:
            night = True
            image_ok = False
        else:
            night = False

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

            if night:
                title_str += ' night.'

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
        return img_cropped, paused
    else:
        return None, None


if __name__ == '__main__':

    this_script_name = os.path.basename(sys.argv[0])
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
                cropped_image, stam = preprocess_one_image(image_filename, camera_name)
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
