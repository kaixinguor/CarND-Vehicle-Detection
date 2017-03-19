import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from car_train import get_features_from_image
import time
from ft_extract import convert_color, get_hog_features,bin_spatial,color_hist

def car_detect(img,classifier):

    # multi-scale sliding window
    window_sizes = [80,96,112,128]
    window_overlap = 0.25
    y_start_stops = [[400, 500], [400, 600], [400, 657], [400, 657]]

    num_scale = len(window_sizes)
    ms_windows = []
    for iscale in range(num_scale):
        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stops[iscale],
                               xy_window=(window_sizes[iscale], window_sizes[iscale]), xy_overlap=(window_overlap, window_overlap))
        ms_windows += windows

    hot_windows = search_windows(img, ms_windows,classifier)

    window_img = draw_boxes(img, ms_windows, color=(0, 0, 1), thick=6)
    plt.imshow(window_img)
    plt.savefig('output_images/detect_multiscale_window.png')
    # plt.show()

    hot_window_img = draw_boxes(img, hot_windows, color=(0, 0, 1), thick=6)
    return hot_windows, hot_window_img


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    # Calculate each window position
    # Append window position to list
    imsize = (img.shape[1], img.shape[0])

    x_start = 0
    x_end = imsize[0] - xy_window[0] + 1
    if x_start_stop[0] is not None:
        x_start = max(x_start,x_start_stop[0])
    if x_start_stop[1] is not None:
        x_end = min(x_end,x_start_stop[1])
    x_start_stop = (x_start, x_end)
    x_step = int(xy_window[0] * xy_overlap[0])

    y_start = 0
    y_end = imsize[1] - xy_window[1] + 1
    if y_start_stop[0] is not None:
        y_start = max(y_start,y_start_stop[0])
    if y_start_stop[1] is not None:
        y_end = min(y_end,y_start_stop[1])
    y_start_stop = (y_start, y_end)
    y_step = int(xy_window[1] * xy_overlap[1])

    for x in range(x_start_stop[0], x_start_stop[1], x_step):
        for y in range(y_start_stop[0], y_start_stop[1], y_step):
            window = ((x, y), (x + xy_window[0], y + xy_window[1]))
            window_list.append(window)

    # Return the list of windows
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, classifier):

    [clf, X_scaler, config] = classifier

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:

        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        #4) Extract features for that window using single_img_features()
        features = get_features_from_image(test_img, config)

        #5) Scale extracted features to be fed to classifier
        test_features = X_scaler.transform(np.array(features).reshape(1, -1))

        #6) Predict using your classifier
        prediction = clf.predict(test_features)

        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def find_car(img, classifier,ystart=400,ystop=657,scale=1.5):
    [clf, X_scaler, config] = classifier
    colorspace = config['colorspace']
    orient = config['orient']
    pix_per_cell = config['pix_per_cell']
    cell_per_block = config['cell_per_block']
    hog_channel = config['hog_channel']
    spatial_feat = config['spatial_feat']
    hist_feat = config['hist_feat']
    hog_feat = config['hog_feat']
    spatial_size = config['spatial_size']
    hist_bins = config['hist_bins']

    # ystart = 400
    # ystop = 657
    # scale = 1.5

    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    ctrans_tosearch = convert_color(img_tosearch, color_space=colorspace)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    #print("debug:",ctrans_tosearch.shape)
    hog_channels = []
    if hog_feat == True:
        if hog_channel == 'ALL':
            for channel in range(ctrans_tosearch.shape[2]):
                #print(channel)
                hog_channels.append(get_hog_features(ctrans_tosearch[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,feature_vec=False))
        else:
            hog_channels.append(get_hog_features(ctrans_tosearch[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, feature_vec=False))

    hot_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            img_features = []

            # Get color features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                img_features.append(spatial_features)

            if hist_feat == True:
                hist_features,_,_ = color_hist(subimg, nbins=hist_bins,color_space=colorspace)
                img_features.append(hist_features)

            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(ctrans_tosearch.shape[2]):
                    hog_features.extend(hog_channels[channel][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
            else:
                hog_features.extend(
                    hog_channels[channel][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
            img_features.append(hog_features)

            # Scale features and make a prediction
            # test_features = X_scaler.transform(
            #     np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_features = X_scaler.transform(np.concatenate(img_features).reshape(1,-1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 1), 6)
                box = ((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart))
                hot_windows.append(box)

    hot_window_img = draw_boxes(img, hot_windows, color=(0, 0, 1), thick=6)
    return hot_windows, hot_window_img


def find_car_multiscale(img, classifier):

    y_start_stops = [[400, 500], [400, 600], [400, 657], [400, 657]]
    scales = [1.25,1.5,1.75,2.0]

    hot_window_list = []
    for iscale in range(len(scales)):
        scale = scales[iscale]
        ystart = y_start_stops[iscale][0]
        ystop = y_start_stops[iscale][1]
        hot_windows, _ = find_car(img, classifier, ystart, ystop, scale)
        hot_window_list += hot_windows
    hot_window_img = draw_boxes(img, hot_window_list, color=(0, 0, 1), thick=6)
    return hot_window_list, hot_window_img


if __name__ == '__main__':
    with open('output_images/car_clf.pkl', 'rb') as f:
        [clf, X_scaler,config] = pickle.load(f)

    img = mpimg.imread('test_images/test4.jpg')
    img = img.astype(np.float32) / 255
    print(img.shape)
    print(np.max(img))


    t = time.time()
    hot_windows0, hot_window_img = car_detect(img, [clf, X_scaler,config])
    print(round(time.time() - t, 2), 'Seconds to detect car ...')

    plt.imshow(hot_window_img)
    plt.savefig('output_images/detect_hot_window1.png')
    # plt.show()

    t = time.time()
    hot_windows1, hot_window_img = find_car(img, [clf, X_scaler,config])
    print(round(time.time() - t, 2), 'Seconds to detect car 2...')
    plt.imshow(hot_window_img)
    plt.savefig('output_images/detect_hot_window2.png')
    # plt.show()

    t = time.time()
    hot_windows2, hot_window_img = find_car_multiscale(img, [clf, X_scaler,config])
    print(round(time.time() - t, 2), 'Seconds to detect car 3...')
    plt.imshow(hot_window_img)
    plt.savefig('output_images/detect_hot_window3.png')

    print(len(hot_windows0))
    print(len(hot_windows1))
    print(len(hot_windows2))
    # plt.show()