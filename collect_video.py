import pyautogui
import datetime
import time

from constants import TIME_BETWEEN_FRAMES_SEC, VIDEO_PATH


if __name__ == '__main__':
    SECTION_LENGTH_SEC = 60

    if 0:
        screenshot = pyautogui.screenshot()
        screenshot.save(f'{VIDEO_PATH}/filename.png')

    if 1:  # save images
        prev_time_sec = time.time()
        while True:
            time_elapsed_sec = time.time() - prev_time_sec
            img = pyautogui.screenshot()
            if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
                filename = f'{VIDEO_PATH}/capture_{now_str}.png'
                img.save(filename)
    else:  # save video
        while True:
            section_start_time_sec = time.time()
            fource = cv2.VideoWriter_fourcc(*'XVID')

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S')
            filename = f'{VIDEO_PATH}\\capture_{now_str}.mp4'
            out = cv2.VideoWriter(filename, fource, 20.0, SCREEN_SIZE)

            prev_time_sec = time.time()
            while time.time() - section_start_time_sec <= SECTION_LENGTH_SEC:
                time_elapsed_sec = time.time() - prev_time_sec
                img = pyautogui.screenshot()
                if time_elapsed_sec > 1.0 / FRAMES_PER_SEC:
                    prev_time_sec = time.time()
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out.write(frame)
                    print(f'saved file {filename}')
                # cv2.waitKey(1000)
