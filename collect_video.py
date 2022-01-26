import datetime
import time
import os
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By

from constants import TIME_BETWEEN_FRAMES_SEC, VIDEO_BASEPATH, OS_SEPARATOR, CAMERAS, CAMERA_LIST, CAMERA_URL, VIDEO_RESET_MIN


def locate_element(driver_, xpath_string):
    ii_try = 0
    while ii_try < 5:
        try:
            search_element_ = driver_.find_element(By.XPATH, xpath_string)
            return search_element_
        except:
            ii_try += 1
            time.sleep(10)
    return None


if __name__ == '__main__':

    if 1:
        # Selenium: https://www.simplilearn.com/tutorials/python-tutorial/selenium-with-python#selenium_webdriver_methods
        # Selenium: https://pythonspot.com/selenium/

        temp_image_name = 'screenshot.png'
        drivers = []
        for camera_name in CAMERA_LIST:
            drivers.append(webdriver.Firefox())
            # driver = webdriver.Chrome()
            drivers[-1].maximize_window()
            # driver.get("https://www.google.com/")
            # driver.find_element_by_name("q").send_keys("javatpoint")

            url = CAMERA_URL
            drivers[-1].get(url)

        start_time = time.time()
        while True:
            while time.time() < start_time + VIDEO_RESET_MIN * 60:
                for iiCamera, driver in enumerate(drivers):
                    camera_name = CAMERA_LIST[iiCamera]
                    driver.refresh()

                    search_element = locate_element(driver, "//input[@class='searchInput']")
                    if search_element is None:
                        raise ValueError('Cannot find search link')
                    search_element.click()
                    search_element.send_keys(CAMERAS.loc[camera_name, 'search_string'])

                time.sleep(30)
                for iiCamera, driver in enumerate(drivers):
                    camera_name = CAMERA_LIST[iiCamera]
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
                    camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
                    if camera_element is None:
                        raise ValueError(f'Cannot find camera link for {camera_name}')

                    # camera_element = driver.find_element(By.XPATH, "//li[@class='col-md-4 cam-item']")
                    camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
                    if camera_element is None:
                        raise ValueError('Cannot find camera link')
                    camera_element.click()
                    # time.sleep(1)

                time.sleep(3)
                Video_Paths = []
                for iiCamera, driver in enumerate(drivers):
                    camera_name = CAMERA_LIST[iiCamera]
                    try:
                        modal = driver.find_element(By.CLASS_NAME, 'modal-backdrop fade show')
                        print(f'{camera_name}: found modal-backdrop')
                        modal.click()
                    except:
                        print(f'{camera_name}: did not find modal-backdrop')
                        stam = 0

                    Video_Paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name)
                    if not os.path.exists(Video_Paths[-1]):
                        os.mkdir(Video_Paths[-1])

                # save images
                if 1:
                    prev_time_sec = time.time()
                    while time.time() < start_time + VIDEO_RESET_MIN * 60:
                        time_elapsed_sec = time.time() - prev_time_sec
                        # img = pyautogui.screenshot()
                        iiFile = 0
                        for iiCamera, driver in enumerate(drivers):
                            img = driver.save_screenshot(temp_image_name)
                            if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
                                now_utc = datetime.datetime.now(datetime.timezone.utc)
                                now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
                                filename = f'{Video_Paths[iiCamera]}/capture_{now_str}.png'
                                # img.save(filename)
                                shutil.move(temp_image_name, filename)
                                # print(f'saved file {filename}')
                                print('.', end='')
                                iiFile += 1
                                if iiFile >30:
                                    print('')
                                    iiFile = 0
                    print(' ')

            start_time = time.time()
