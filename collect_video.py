import datetime
import time
import os
import shutil
from multiprocessing import Process

from selenium import webdriver
from selenium.webdriver.common.by import By

from constants import TIME_BETWEEN_FRAMES_SEC, VIDEO_BASEPATH, OS_SEPARATOR, CAMERAS, \
    CAMERA_LIST, CAMERA_URL, VIDEO_RESET_MIN
from preprocess import preprocess_one_image


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


def collect_one_camera(camera_name_):
    temp_image_name = f'screenshot_{camera_name_}.png'
    driver = None
    while True:
        driver_error = False

        if driver is not None:
            try:
                driver.close()
            except:
                stam=0

        # driver = webdriver.Firefox()

        if 0:
            options = webdriver.ChromeOptions()
            options.add_argument("headless")
            driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver', chrome_options=options)
        else:
            driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver')

        driver.maximize_window()
        try:
            driver.get(CAMERA_URL)
        except:
            driver_error = True

        start_time = time.time()
        while not driver_error:
            while (time.time() < start_time + VIDEO_RESET_MIN * 60) and (not driver_error):
                try:
                    driver.refresh()
                except:
                    driver_error = True

                # if 0 and not driver_error:
                if not driver_error:
                    search_element = locate_element(driver, "//input[@class='searchInput']")
                    if search_element is None:
                        raise ValueError(f'Cannot find search link for {camera_name_}')
                    try:
                        search_element.click()
                        search_element.send_keys(CAMERAS[camera_name_]['search_string'])
                    except:
                        driver_error = True

                if not driver_error:
                    time.sleep(30)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
                    camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
                    if camera_element is None:
                        driver_error = True

                    if not driver_error:
                        try:
                            camera_element.click()
                        except:
                            driver_error = True

                        if not driver_error:
                            time.sleep(10)
                            video_path = VIDEO_BASEPATH + OS_SEPARATOR + camera_name_
                            if not os.path.exists(video_path):
                                os.mkdir(video_path)

                            # save images
                            try:
                                first_image_checked = False
                                ii_file = 0
                                restart_browser = False
                                prev_time_sec = time.time()
                                while (time.time() < start_time + VIDEO_RESET_MIN * 60) and (not restart_browser):
                                    time_elapsed_sec = time.time() - prev_time_sec
                                    if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
                                        img = driver.save_screenshot(temp_image_name)
                                        now_utc = datetime.datetime.now(datetime.timezone.utc)
                                        now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
                                        filename = video_path + OS_SEPARATOR + f'capture_{now_str}.png'
                                        shutil.copyfile(temp_image_name, filename)
                                        print('.', end='')
                                        ii_file += 1
                                        if ii_file > 30:
                                            print('')
                                            ii_file = 0
                                        if not first_image_checked:
                                            if 0:
                                                filename = r'/media/dana/second local disk1/dana/Netivei/videos/Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_02_02_12_17_32_733200.png'
                                                camera_name_ = 'Aluf_Sadeh'
                                            stam, image_ok, image_status = preprocess_one_image(temp_image_name, camera_name_, skip_time_test=True)
                                            first_image_checked = True
                                            restart_browser = image_status['paused'] or image_status['error'] or not image_status['camera_name_ok']

                                print(' ')
                            except:
                                driver_error = True

                start_time = time.time()


if __name__ == '__main__':
    if 1:
        # Selenium: https://www.simplilearn.com/tutorials/python-tutorial/selenium-with-python#selenium_webdriver_methods
        # Selenium: https://pythonspot.com/selenium/

        # for camera_name in CAMERA_LIST:
        #     collect_one_camera(camera_name)

        if 1 and len(CAMERA_LIST) > 1:  # parallel cameras
            procs = []
            for camera_name in CAMERA_LIST:
                # print(name)
                proc = Process(target=collect_one_camera, args=(camera_name,))
                procs.append(proc)
                proc.start()

                # complete the processes
            for proc in procs:
                proc.join()
        else:  # one camera
            collect_one_camera(CAMERA_LIST[0])
