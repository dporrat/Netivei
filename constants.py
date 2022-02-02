import os
import numpy as np
from PIL import Image

if 1:
    SCREEN_SIZE = (1920, 1080)
    VIDEO_BASEPATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
    OS_SEPARATOR = f'\\'
    if not os.path.exists(VIDEO_BASEPATH):
        SCREEN_SIZE = (1280, 1024)
        VIDEO_BASEPATH = '/media/dana/second local disk1/dana/Netivei/videos'
        OS_SEPARATOR = f'/'

        # selenium on linux
        SCREEN_SIZE = (1207, 883)

if 1:
    CROP_DATA = {}
    CROP_DATA[(1207, 883)] = {'screen_name': 'selenium on linux',
                              'crop_start_x': 204,
                              'crop_start_y': 158,
                              'crop_width': 800,
                              'crop_height': 450}
    CROP_DATA[(1920, 923)] = {'screen_name': 'selenium on linux',
                              'crop_start_x': 560,
                              'crop_start_y': 180,
                              'crop_width': 800,
                              'crop_height': 450}
if 0:
    if 1:
        CROP_DATA = pd.DataFrame(
            columns=['screen_width', 'screen_height', 'screen_name', 'crop_start_x', 'crop_start_y', 'crop_width',
                     'crop_height'])

    new_row = pd.DataFrame({'screen_width': [1207],
                            'screen_height': [883],
                            'screen_name': ['selenium on linux'],
                            'crop_start_x': [204],
                            'crop_start_y': [158],
                            'crop_width': [800],
                            'crop_height': [450]})
    CROP_DATA = pd.concat((CROP_DATA, new_row))

    new_row = pd.DataFrame({'screen_width': [1920],
                            'screen_height': [923],
                            'screen_name': ['selenium on laptop'],
                            'crop_start_x': [204],
                            'crop_start_y': [158],
                            'crop_width': [800],
                            'crop_height': [450]})
    CROP_DATA = pd.concat((CROP_DATA, new_row))
    CROP_DATA.index = [0, 1]

    # CROP_DATA.set_index(['screen_width', 'screen_height'])

if 1:
    if 0:
        # crop positions
        if 1:
            CROP_START_X = 276
            CROP_START_Y = 298
            CROP_WIDTH = 800
            CROP_HEIGHT = 450

            # Raanana Merkaz from selenium communication problem
            if 0:
                CROP_START_X = 215
                CROP_START_Y = 158
                CROP_WIDTH = 779
                CROP_HEIGHT = 450

            # Aluf Sadeh from selenium good image
            CROP_START_X = 204
            CROP_START_Y = 158

            if 0:
                # Aluf Sadeh from selenium with play arrow
                CROP_START_X = 211
                CROP_START_Y = 163
                CROP_WIDTH = 786
                CROP_HEIGHT = 442

    DAY_START_UTC = 4
    DAY_END_UTC = 17

    TIME_BETWEEN_FRAMES_SEC = 0.2

if 0:
    CAMERAS = pd.DataFrame({'name': ['Raanana_Merkaz', 'Aluf_Sadeh', 'Abu_Gosh'],
                            'search_string': ['רעננה מרכז', 'אלוף שדה', 'אבו גוש'],
                            'logo_start_x': [0, 0, 9],
                            'logo_start_y': [0, 0, 16],
                            })
    CAMERAS.set_index('name', inplace=True)
    CAMERAS.sort_index(inplace=True)

if 1:
    CAMERAS = {}
    CAMERAS['Abu_Gosh'] = {'search_string': 'אבו גוש',
                           'logo_start_x': 9,
                           'logo_start_y': 16}
    CAMERAS['Aluf_Sadeh'] = {'search_string': 'אלוף שדה',
                             'logo_start_x': 0,
                             'logo_start_y': 0}
    CAMERAS['Raanana_Merkaz'] = {'search_string': 'רעננה מרכז',
                                 'logo_start_x': 0,
                                 'logo_start_y': 0}

if 1:
    CAMERA_LIST = ['Aluf_Sadeh', 'Raanana_Merkaz']

    CAMERA_LIST = ['Aluf_Sadeh']

    CAMERA_URL = 'https://www.iroads.co.il/%D7%AA%D7%99%D7%A7%D7%99%D7%99%D7%AA-%D7%9E%D7%A6%D7%9C%D7%9E%D7%95%D7%AA/'

    VIDEO_RESET_MIN = 15

    RAD_PER_DEG = np.pi / 180

# Netivei logo
if 1:
    NETIVEI_WIDTH = 177
    NETIVEI_HEIGHT = 49
    LOGO_FILENAME = f'netivei_logo.png'
    NETIVEI_LOGO = Image.open(LOGO_FILENAME)
    NETIVEI_LOGO_BW = np.array(NETIVEI_LOGO.convert('L'))

# offline wait circle
if 1:
    CIRCLE_START_X = 363
    CIRCLE_START_Y = 186
    CIRCLE_WIDTH = 75
    CIRCLE_HEIGHT = 75
    OFFLINE_CIRCLE_PIXELS = []
    CENTER_PIXEL = (37, 37)
    delta_theta_deg = 0.5
    for radius in [32.5, 33.5, 34.5]:
        for theta in np.arange(0, 360, delta_theta_deg):
            x = round(CENTER_PIXEL[0] + radius * np.cos(theta * RAD_PER_DEG))
            y = round(CENTER_PIXEL[0] + radius * np.sin(theta * RAD_PER_DEG))
            if (x, y) not in OFFLINE_CIRCLE_PIXELS:
                OFFLINE_CIRCLE_PIXELS.append((x, y))

# play triangle
if 1:
    TRIANGLE_START_X = 365
    TRIANGLE_START_Y = 186
    TRIANGLE_WIDTH = 80
    TRIANGLE_HEIGHT = 80
    PLAY_TRIANGLE_PIXELS = []
    for x in range(7, 81):
        for y in np.arange(int(x / 2) - 1, 78 - int(x / 2)):
            if (x, y) not in PLAY_TRIANGLE_PIXELS:
                PLAY_TRIANGLE_PIXELS.append((x, y))

