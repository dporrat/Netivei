import os
import numpy as np
from PIL import Image

if 1:
    VIDEO_BASEPATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
    OS_SEPARATOR = f'\\'
    if not os.path.exists(VIDEO_BASEPATH):
        VIDEO_BASEPATH = '/media/dana/second local disk1/dana/Netivei/videos'
        VIDEO_BASEPATH = '/home/user1/Dana Porrat/Netivei/videos'
        OS_SEPARATOR = f'/'

if 1:
    CROP_DATA = {}
    CROP_DATA[(1207, 883)] = {'screen_name': 'selenium on linux',
                              'crop_start_x': 204,
                              'crop_start_y': 158,
                              'crop_width': 800,
                              'crop_height': 450,
                              'netivei_logo_width': 177,
                              'netivei_logo_height': 49,
                              'netivei_logo_filename': 'netivei_logo_177_49.png',
                              'top_right_width': 130,
                              'top_right_height': 12,
                              }
    CROP_DATA[(1920, 923)] = {'screen_name': 'selenium on linux',
                              'crop_start_x': 560,
                              'crop_start_y': 180,
                              'crop_width': 800,
                              'crop_height': 450,
                              'netivei_logo_width': 177,
                              'netivei_logo_height': 49,
                              'netivei_logo_filename': 'netivei_logo_177_49.png',
                              'top_right_width': 130,
                              'top_right_height': 12,
                              }
    CROP_DATA[(1848, 896)] = {'screen_name': 'selenium on linux',
                              'crop_start_x': 604,
                              'crop_start_y': 221,
                              'crop_width': 640,
                              'crop_height': 360,
                              'netivei_logo_width': 142,
                              'netivei_logo_height': 40,
                              'netivei_logo_filename': 'netivei_logo_142_40.png',
                              'top_right_width': 130,
                              'top_right_height': 12,
                              }

if 1:
    DAY_START_UTC = 4
    DAY_END_UTC = 17

    TIME_BETWEEN_FRAMES_SEC = 0.2

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
                                 'logo_start_y': 0,
                                 'camera_name_file': f'Raanana_Merkaz_130_12.png'}

if 1:
    CAMERA_LIST = ['Aluf_Sadeh', 'Raanana_Merkaz']
    CAMERA_LIST = ['Raanana_Merkaz']

    CAMERA_URL = 'https://www.iroads.co.il/%D7%AA%D7%99%D7%A7%D7%99%D7%99%D7%AA-%D7%9E%D7%A6%D7%9C%D7%9E%D7%95%D7%AA/'

    VIDEO_RESET_MIN = 15

    RAD_PER_DEG = np.pi / 180

# Netivei logo - moved to per screen size
if 0:
    NETIVEI_LOGO_WIDTH = 177
    NETIVEI_LOGO_HEIGHT = 49
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

# top right corner - moved to per screen size
if 0:
    TOP_RIGHT_START_X = 670
    TOP_RIGHT_START_Y = 0
    TOP_RIGHT_WIDTH = 130
    TOP_RIGHT_HEIGHT = 12
