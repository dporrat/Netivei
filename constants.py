import os

SCREEN_SIZE = (1920, 1080)
VIDEO_BASEPATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
OS_SEPARATOR = f'\\'
if not os.path.exists(VIDEO_BASEPATH):
    SCREEN_SIZE = (1280, 1024)
    VIDEO_BASEPATH = '/media/dana/second local disk1/dana/Netivei/videos'
    OS_SEPARATOR = f'/'

TIME_BETWEEN_FRAMES_SEC = 0.2

CAMERA_LINKS = {'Abu_Gosh': 'https://www.iroads.co.il/%D7%AA%D7%99%D7%A7%D7%99%D7%99%D7%AA-%D7%9E%D7%A6%D7%9C%D7%9E'
                            '%D7%95%D7%AA/#camera1Modal275-381',
                'Raanana_Merkaz': 'https://www.iroads.co.il/%D7%AA%D7%99%D7%A7%D7%99%D7%99%D7%AA-%D7%9E%D7%A6%D7%9C'
                                  '%D7%9E%D7%95%D7%AA/#camera1Modal188-43',
                }

CAMERA = 'Abu_Gosh'
