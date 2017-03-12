import cv2
import os
from car_detect import car_detect
from matplotlib import pyplot as plt
import imageio

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    target = "project"
    output_dir = "output_images/" + target + "_video_result"
    ensure_dir(output_dir)

    video_file = target + '_video.mp4'
    # vidcap = cv2.VideoCapture()
    vid = imageio.get_reader(video_file, 'ffmpeg')
    num_img = len(vid)
    for i, img in enumerate(vid):
        print('Frame %i / %i' % (i+1, num_img))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("output_images/" + target + "_video_result.avi", fourcc, fps, (width, height))

        # img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = car_detect(img)

        cv2.imwrite("output_images/" + target + "_video_result/{:04d}.jpg".format(i),result)
        # out.write(result)
        # count += 1
    #
    # print("width: ", width, ", height: ", height)
    # print("frame: ", length)
    # print("fps: ", fps)
    # print("count:", count)
    # vidcap.release()
    # out.release()