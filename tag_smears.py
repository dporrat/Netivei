import os
import glob
import shutil

from matplotlib import pyplot as plt
from PIL import Image

from constants import VIDEO_BASEPATH, OS_SEPARATOR

input_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                   OS_SEPARATOR + '2022_02_16'
reverse = True


def on_press(event):
    if event.key == 'd':
        shutil.copy(file, smeared_folder + OS_SEPARATOR + os.path.basename(file))
        print(f'File {os.path.basename(file)} copied to smeared folder')
        plt.close()
    elif event.key == 'o':
        shutil.copy(file, ok_folder + OS_SEPARATOR + os.path.basename(file))
        print(f'File {os.path.basename(file)} copied to ok folder')
        plt.close()
    else:
        print('press d or o')
    if 0:
        sys.stdout.flush()
        if event.key == 'x':
            visible = xl.get_visible()
            xl.set_visible(not visible)
            fig.canvas.draw()


ok_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                   OS_SEPARATOR + 'find_smears' + OS_SEPARATOR + 'images_ok'
smeared_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                   OS_SEPARATOR + 'find_smears' + OS_SEPARATOR + 'images_smeared'

all_files = glob.glob(input_folder + OS_SEPARATOR + '*.png')
files = []
for file in all_files:
    if not os.path.exists(ok_folder + OS_SEPARATOR + os.path.basename(file)) and \
            not os.path.exists(smeared_folder + OS_SEPARATOR + os.path.basename(file)):
        files.append(file)

files.sort(reverse=reverse)
for file in files:
    print(file)
    fig = plt.figure(figsize=(12, 12))
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.imshow(Image.open(file))
    plt.title(os.path.basename(file))
    plt.show()
    print(' ')

