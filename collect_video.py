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
        if driver is not None:
            driver.close()

        driver = webdriver.Firefox()
        driver.maximize_window()
        driver.get(CAMERA_URL)
        driver_error = False
        start_time = time.time()
        while not driver_error:
            while (time.time() < start_time + VIDEO_RESET_MIN * 60) and (not driver_error):
                try:
                    driver.refresh()
                except:
                    driver_error = True

                if not driver_error:
                    search_element = locate_element(driver, "//input[@class='searchInput']")
                    if search_element is None:
                        raise ValueError(f'Cannot find search link for {camera_name_}')
                    search_element.click()
                    search_element.send_keys(CAMERAS[camera_name_]['search_string'])

                if not driver_error:
                    time.sleep(30)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
                    camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
                    if camera_element is None:
                        raise ValueError(f'Cannot find camera link for {camera_name_}')

                    camera_element.click()

                    time.sleep(3)
                    video_path = VIDEO_BASEPATH + OS_SEPARATOR + camera_name_
                    if not os.path.exists(video_path):
                        os.mkdir(video_path)

                    # save images
                    if 1:
                        first_image_checked = False
                        ii_file = 0
                        paused = False
                        prev_time_sec = time.time()
                        while (time.time() < start_time + VIDEO_RESET_MIN * 60) and (not paused):
                            time_elapsed_sec = time.time() - prev_time_sec
                            if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
                                img = driver.save_screenshot(temp_image_name)
                                now_utc = datetime.datetime.now(datetime.timezone.utc)
                                now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
                                filename = video_path + OS_SEPARATOR + f'capture_{now_str}.png'
                                shutil.move(temp_image_name, filename)
                                print('.', end='')
                                ii_file += 1
                                if ii_file > 30:
                                    print('')
                                    ii_file = 0
                                if not first_image_checked:
                                    if 0:
                                        filename = r'/media/dana/second local disk1/dana/Netivei/videos/Aluf_Sadeh' + OS_SEPARATOR + 'capture_2022_02_02_12_06_51_567975.png'
                                        camera_name_ = 'Aluf_Sadeh'
                                    stam, paused = preprocess_one_image(filename, camera_name_)
                                    first_image_checked = True

                        print(' ')

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

        # drivers = []
        # while True:
        #     driver_error = False
        #     if len(drivers) > 0:
        #         for driver in drivers:
        #             driver.close()
        #     drivers = []
        #     for camera_name in CAMERA_LIST:
        #         drivers.append(webdriver.Firefox())
        #         drivers[-1].maximize_window()
        #         # driver = webdriver.Chrome()
        #         # driver.get("https://www.google.com/")
        #         # driver.find_element_by_name("q").send_keys("javatpoint")
        #
        #         url = CAMERA_URL
        #         drivers[-1].get(url)
        #
        #     start_time = time.time()
        #     while not driver_error:
        #         while (time.time() < start_time + VIDEO_RESET_MIN * 60) and (not driver_error):
        #             for iiCamera, driver in enumerate(drivers):
        #                 camera_name = CAMERA_LIST[iiCamera]
        #                 try:
        #                     driver.refresh()
        #                 except:
        #                     driver_error = True
        #
        #                 if not driver_error:
        #                     search_element = locate_element(driver, "//input[@class='searchInput']")
        #                     if search_element is None:
        #                         raise ValueError(f'Cannot find search link for {camera_name}')
        #                     search_element.click()
        #                     search_element.send_keys(CAMERAS.loc[camera_name, 'search_string'])
        #
        #             if not driver_error:
        #                 time.sleep(30)
        #                 for iiCamera, driver in enumerate(drivers):
        #                     camera_name = CAMERA_LIST[iiCamera]
        #                     driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2)")
        #                     camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
        #                     if camera_element is None:
        #                         raise ValueError(f'Cannot find camera link for {camera_name}')
        #
        #                     # camera_element = driver.find_element(By.XPATH, "//li[@class='col-md-4 cam-item']")
        #                     # camera_element = locate_element(driver, "//li[@class='col-md-4 cam-item']")
        #                     # if camera_element is None:
        #                     #     raise ValueError('Cannot find camera link')
        #                     camera_element.click()
        #                     # time.sleep(1)
        #
        #                 time.sleep(3)
        #                 Video_Paths = []
        #                 for iiCamera, driver in enumerate(drivers):
        #                     camera_name = CAMERA_LIST[iiCamera]
        #                     if 0:
        #                         try:
        #                             modal = driver.find_element(By.CLASS_NAME, 'modal-backdrop fade show')
        #                             print(f'{camera_name}: found modal-backdrop')
        #                             modal.click()
        #                         except:
        #                             print(f'{camera_name}: Error! did not find modal-backdrop')
        #
        #                     Video_Paths.append(VIDEO_BASEPATH + OS_SEPARATOR + camera_name)
        #                     if not os.path.exists(Video_Paths[-1]):
        #                         os.mkdir(Video_Paths[-1])
        #
        #                 # save images
        #                 if 1:
        #                     first_image_checked = np.zeros(len(drivers))
        #                     iiFile = 0
        #                     prev_time_sec = time.time()
        #                     while time.time() < start_time + VIDEO_RESET_MIN * 60:
        #                         time_elapsed_sec = time.time() - prev_time_sec
        #                         if time_elapsed_sec >= TIME_BETWEEN_FRAMES_SEC:
        #                             for iiCamera, driver in enumerate(drivers):
        #                                 camera_name = CAMERA_LIST[iiCamera]
        #                                 img = driver.save_screenshot(temp_image_name)
        #                                 now_utc = datetime.datetime.now(datetime.timezone.utc)
        #                                 now_str = now_utc.strftime('%Y_%m_%d_%H_%M_%S_%f')
        #                                 filename = f'{Video_Paths[iiCamera]}/capture_{now_str}.png'
        #                                 shutil.move(temp_image_name, filename)
        #                                 print('.', end='')
        #                                 if iiCamera == 0:
        #                                     iiFile += 1
        #                                     if iiFile > 30:
        #                                         print('')
        #                                         iiFile = 0
        #                                 if not first_image_checked[iiCamera]:
        #                                     stam, paused = preprocess_one_image(filename, camera_name)
        #                                     # if paused:
        #
        #
        #                     print(' ')
        #
        #         start_time = time.time()
