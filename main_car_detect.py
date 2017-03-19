import cv2
import os
from car_detect import car_detect, find_car_multiscale
from matplotlib import pyplot as plt
import imageio
import pickle
import numpy as np

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    target = "project"

    # output_dir = "output_images/" + target + "_video_result"
    # ensure_dir(output_dir)

    video_file = target + '_video.mp4'
    # vidcap = cv2.VideoCapture()
    vid = imageio.get_reader(video_file, 'ffmpeg')
    num_img = len(vid)

    with open('output_images/car_clf.pkl', 'rb') as f:
        [clf, X_scaler, config] = pickle.load(f)


    box_list = []
    for i, img in enumerate(vid):
        print('Frame %i / %i' % (i+1, num_img))

        img = img.astype(np.float32) / 255
        # print(np.max(img))

        #hot_window_img = car_detect(img, [clf, X_scaler, config])
        hot_windows, hot_window_img = find_car_multiscale(img, [clf, X_scaler, config])

        # result = 255*cv2.cvtColor(hot_window_img, cv2.COLOR_RGB2BGR)
        # result = result.astype(np.uint8)
        # cv2.imwrite("output_images/" + target + "_video_result/{:04d}.png".format(i),result)

        box_list.append(hot_windows)

    with open('output_images/car_box.pkl', 'wb') as f:
        pickle.dump(box_list, f)