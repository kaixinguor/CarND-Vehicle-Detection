import cv2
import os
from car_detect import car_detect
from matplotlib import pyplot as plt

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    target = "project"
    output_dir = "output_images/" + target + "_video_result"
    ensure_dir(output_dir)

    vidcap = cv2.VideoCapture("./test_video.mp4")
    print(vidcap)

    print(target + "_video.mp4")
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    print("width: ", width, ", height: ", height)
    print("frame: ", length)
    print("fps: ", fps)
    exit(0)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_images/" + target + "_video_result.avi", fourcc, fps, (width, height))

    count = 0
    while True:
        success, image = vidcap.read()
        if success is False:
            break

        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = car_detect(img)

        cv2.imwrite("output_images/" + target + "_video_result/{:04d}.jpg".format(count),result)
        out.write(result)
        count += 1

    print("width: ", width, ", height: ", height)
    print("frame: ", length)
    print("fps: ", fps)
    print("count:", count)
    vidcap.release()
    out.release()