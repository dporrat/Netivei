import glob
import os
import shutil

import random

from constants import VIDEO_BASEPATH, OS_SEPARATOR

num_validate_smeared_images = 50

find_smears_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                     OS_SEPARATOR + 'find_smears'

images_ok_folder = find_smears_folder + OS_SEPARATOR + 'images_ok'
images_smeared_folder = find_smears_folder + OS_SEPARATOR + 'images_smeared'

images_ok_list = glob.glob(images_ok_folder + OS_SEPARATOR + '*.png')
images_smeared_list = glob.glob(images_smeared_folder + OS_SEPARATOR + '*.png')

num_images_ok = len(images_ok_list)
num_images_smeared = len(images_smeared_list)

if num_images_ok < num_images_smeared:
    raise ValueError(f'Too few ok images in folder {images_ok_folder}')

if num_images_smeared <= num_validate_smeared_images:
    raise ValueError(f'Too few smeared images in folder {images_smeared_folder}')

num_train_smeared_images = num_images_smeared - num_validate_smeared_images
num_train_ok_images = num_train_smeared_images
num_validate_ok_images = num_validate_smeared_images

random.shuffle(images_ok_list)
random.shuffle(images_smeared_list)

train_folder = find_smears_folder + OS_SEPARATOR + 'train'
validate_folder = find_smears_folder + OS_SEPARATOR + 'validate'


if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
os.mkdir(train_folder)

if os.path.exists(validate_folder):
    shutil.rmtree(validate_folder)
os.mkdir(validate_folder)

train_folder_ok = train_folder + OS_SEPARATOR + 'ok'
os.mkdir(train_folder_ok)
train_folder_smeared = train_folder + OS_SEPARATOR + 'smeared'
os.mkdir(train_folder_smeared)
validate_folder_ok = validate_folder + OS_SEPARATOR + 'ok'
os.mkdir(validate_folder_ok)
validate_folder_smeared = validate_folder + OS_SEPARATOR + 'smeared'
os.mkdir(validate_folder_smeared)

for file in images_ok_list[:num_train_ok_images]:
    shutil.copy(file, train_folder_ok + OS_SEPARATOR)
for file in images_ok_list[num_train_ok_images:num_train_ok_images + num_validate_ok_images]:
    shutil.copy(file, validate_folder_ok + OS_SEPARATOR)
for file in images_smeared_list[:num_train_smeared_images]:
    shutil.copy(file, train_folder_smeared + OS_SEPARATOR)
for file in images_smeared_list[num_train_smeared_images:num_train_smeared_images + num_validate_smeared_images]:
    shutil.copy(file, validate_folder_smeared + OS_SEPARATOR)

