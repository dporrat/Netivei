import os
import glob
import cv2

from constants import VIDEO_BASEPATH

start_time = '2022_03_08_10'
minutes = ['00', '01']
video_name = VIDEO_BASEPATH + f'/Raanana_Merkaz/2022_03_08_10_video.avi'

png_folder = VIDEO_BASEPATH + f'/Raanana_Merkaz/cropped_images/2022_03_08'
png_list = []
for minute in minutes:
    png_list += glob.glob(png_folder + f'/capture_' + start_time + '_' + minute + '*.png')
# print(len(png_list))
# print(png_list[0])

frame = cv2.imread(os.path.join(png_folder, png_list[0]))
height, width, layers = frame.shape

fps = 5
video = cv2.VideoWriter(video_name, 0, fps, (width, height))
# video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for png_file in png_list:
    video.write(cv2.imread(os.path.join(png_folder, png_file)))

cv2.destroyAllWindows()
video.release()

