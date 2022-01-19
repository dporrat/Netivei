import pyautogui
import datetime
import time
import os

from selenium import webdriver
from selenium.webdriver.common.by import By

from constants import TIME_BETWEEN_FRAMES_SEC, VIDEO_BASEPATH, OS_SEPARATOR, CAMERA, CAMERA_LINKS


if __name__ == '__main__':

    # open url - doesn't work
    if 0:
        driver = webdriver.Firefox()
        # driver = webdriver.Chrome()
        driver.maximize_window()
        # driver.get("https://www.google.com/")
        # driver.find_element_by_name("q").send_keys("javatpoint")

        url = CAMERA_LINKS[CAMERA]
        driver.get(url)

        search_element = driver.find_element(By.CLASS_NAME, 'searchBar')

        search_element.click()
        search_element.send_keys("stam")

    Video_Path = VIDEO_BASEPATH + OS_SEPARATOR + CAMERA
    if not os.path.exists(Video_Path):
        os.mkdir(Video_Path)
    SECTION_LENGTH_SEC = 60

    if 0:
        screenshot = pyautogui.screenshot()
        screenshot.save(f'{Video_Path}/filename.png')

    if 1:  # save images
        prev_time_sec = time.time()
        while True:
            time_elapsed_sec = time.time() - prev_time_sec
            img = pyautogui.screenshot()
            if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
                filename = f'{Video_Path}/capture_{now_str}.png'
                img.save(filename)
                print(f'saved file {filename}')
    else:  # save video
        while True:
            section_start_time_sec = time.time()
            fource = cv2.VideoWriter_fourcc(*'XVID')

            now_utc = datetime.datetime.now(datetime.timezone.utc)
            now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S')
            filename = f'{Video_Path}\\capture_{now_str}.mp4'
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
