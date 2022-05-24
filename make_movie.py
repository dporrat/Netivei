import glob
from constants import VIDEO_BASEPATH

start_time = '2022_03_08_10'
png_folder = VIDEO_BASEPATH + f'/Raanana_Merkaz/cropped_images/2022_03_08'
png_list = glob.glob(png_folder + f'/capture_' + start_time + '*.png')
print(len(png_list))
print(png_list[0])

