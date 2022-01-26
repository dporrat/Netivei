import os
import pandas as pd

SCREEN_SIZE = (1920, 1080)
VIDEO_BASEPATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
OS_SEPARATOR = f'\\'
if not os.path.exists(VIDEO_BASEPATH):
    SCREEN_SIZE = (1280, 1024)
    VIDEO_BASEPATH = '/media/dana/second local disk1/dana/Netivei/videos'
    OS_SEPARATOR = f'/'

TIME_BETWEEN_FRAMES_SEC = 0.2

CAMERAS = pd.DataFrame({'name': ['Raanana_Merkaz', 'Aluf_Sadeh', 'Abu_Gosh'],
                        'search_string': ['רעננה מרכז', 'אלוף שדה', 'אבו גוש'],
                        'logo_start_x': [0, 0, 9],
                        'logo_start_y': [0, 0, 16],
                        })

CAMERA_LIST = ['Raanana_Merkaz', 'Aluf_Sadeh']

CAMERAS.set_index('name', inplace=True)

CAMERA_URL = 'https://www.iroads.co.il/%D7%AA%D7%99%D7%A7%D7%99%D7%99%D7%AA-%D7%9E%D7%A6%D7%9C%D7%9E%D7%95%D7%AA/'

VIDEO_RESET_MIN = 0.5
