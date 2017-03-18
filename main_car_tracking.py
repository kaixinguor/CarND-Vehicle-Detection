import pickle
import imageio
import os
from car_detect import draw_boxes
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


if __name__ == '__main__':
    target = "project"

    output_dir_det = "output_images/" + target + "_video_result_det/"
    ensure_dir(output_dir_det)
    output_dir_trac = "output_images/" + target + "_video_result_trac/"
    ensure_dir(output_dir_trac)

    video_file = target + '_video.mp4'
    vid = imageio.get_reader(video_file, 'ffmpeg')
    num_img = len(vid)

    fps = vid.get_meta_data()['fps']
    writer = imageio.get_writer("output_images/" + target + "_video_result.mp4", fps=fps)

    with open('output_images/car_box.pkl', 'rb') as f:
        box_list = pickle.load(f)

    idx = -1
    for i, img in enumerate(vid):
        idx += 1

        print('Frame %i / %i' % (i+1, num_img))

        hot_windows = box_list[idx]


        # visualize detection
        # hot_window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
        # result = cv2.cvtColor(hot_window_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(output_dir_det + "{:04d}.png".format(i),result)

        # plt.imshow(hot_window_img)
        # plt.show()

        # for tracking
        hot_windows = sum(box_list[idx:idx+20],[])
        # print(hot_windows)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 5)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        # print("labels", labels)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        writer.append_data(draw_img)

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()
        # plt.savefig(output_dir_trac + "{:04d}.png".format(i))
        # # plt.show()
        # plt.close()

    writer.close()