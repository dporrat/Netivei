import cv2
import numpy as np
import pyautogui
import datetime
import time


if __name__ == '__main__':
    SCREEN_SIZE = (1920, 1080)
    VIDEO_PATH = f'C:\\Users\\dporrat\\Desktop\\Netivei\\videos'
    SECTION_LENGTH_SEC = 60
    FRAMES_PER_SEC = 10

    while True:
        section_start_time_sec = time.time()
        fource = cv2.VideoWriter_fourcc(*'XVID')

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S')
        filename = f'{VIDEO_PATH}\\capture_{now_str}.mp4'
        out = cv2.VideoWriter(filename, fource, 20.0, (SCREEN_SIZE))

        prev_time_sec = time.time()
        while time.time() - section_start_time_sec <= SECTION_LENGTH_SEC:
            time_elapsed_sec = time.time() - prev_time_sec
            img = pyautogui.screenshot()
            if time_elapsed_sec > 1.0 / FRAMES_PER_SEC:
                prev_time_sec = time.time()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)
            # cv2.waitKey(1000)
