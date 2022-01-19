import os

SCREEN_SIZE = (1920, 1080)
VIDEO_PATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
if not os.path.exists(VIDEO_PATH):
    SCREEN_SIZE = (1280, 1024)
    VIDEO_PATH = '/media/dana/second local disk1/dana/Netivei/videos'

TIME_BETWEEN_FRAMES_SEC = 0.2
