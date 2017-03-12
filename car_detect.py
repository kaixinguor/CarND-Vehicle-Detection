import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ft_extract import single_img_features

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
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()

        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows





def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

#
# # Define a single function that can extract features using hog sub-sampling and make predictions
# def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
#     draw_img = np.copy(img)
#     img = img.astype(np.float32) / 255
#
#     img_tosearch = img[ystart:ystop, :, :]
#     ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
#
#     ch1 = ctrans_tosearch[:, :, 0]
#     ch2 = ctrans_tosearch[:, :, 1]
#     ch3 = ctrans_tosearch[:, :, 2]
#
#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell) - 1
#     nyblocks = (ch1.shape[0] // pix_per_cell) - 1
#     nfeat_per_block = orient * cell_per_block ** 2
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell) - 1
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
#
#     # Compute individual channel HOG features for the entire image
#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
#
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb * cells_per_step
#             xpos = xb * cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
#             print(hog_features.shape)
#
#             xleft = xpos * pix_per_cell
#             ytop = ypos * pix_per_cell
#
#             # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
#
#             # Get color features
#             spatial_features = bin_spatial(subimg, size=spatial_size)
#             print(spatial_features.shape)
#             hist_features = color_hist(subimg, nbins=hist_bins)
#             print(hist_features.shape)
#
#             # Scale features and make a prediction
#             test_features = X_scaler.transform(
#                 np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
#             # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
#             test_prediction = svc.predict(test_features)
#
#             if test_prediction == 1:
#                 xbox_left = np.int(xleft * scale)
#                 ytop_draw = np.int(ytop * scale)
#                 win_draw = np.int(window * scale)
#                 cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
#                               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
#
#     return draw_img
#

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


def car_detect(img):
    with open('output_images/car_clf.pkl', 'rb') as f:
        [svc, X_scaler] = pickle.load(f)

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = False
    hist_feat = False
    hog_feat = True


    window_sizes = [64,128,256,512]
    window_overlaps = [0.5,0.4,0.3,0.2]
    num_scale = len(window_sizes)
    ms_windows = []
    for iscale in range(num_scale):
        window_size = window_sizes[iscale]
        window_overlap = window_overlaps[iscale]

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[400,None],
                               xy_window=(window_size, window_size), xy_overlap=(window_overlap, window_overlap))
        ms_windows += windows

    hot_windows = search_windows(img, ms_windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel,
                                 spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    # window_img = draw_boxes(img, ms_windows, color=(0, 0, 255), thick=6)
    # plt.imshow(window_img)
    # plt.savefig('output_images/detect_multiscale_window.png')

    hot_window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
    return hot_window_img

if __name__ == '__main__':
    with open('output_images/car_clf.pkl', 'rb') as f:
        [svc, X_scaler] = pickle.load(f)

    img = mpimg.imread('test_images/test1.jpg')
    # image = image.astype(np.float32) / 255

    hot_window_img = car_detect(img)
    plt.imshow(hot_window_img)
    plt.savefig('output_images/detect_hot_window.png')
    plt.show()

    # ystart = 400
    # ystop = 656
    # scale = 1.5
    #
    #
    # orient = 9  # HOG orientations
    # pix_per_cell = 4  # HOG pixels per cell
    # cell_per_block = 4  # HOG cells per block
    # spatial_size = (32,32)  # Spatial binning dimensions
    # hist_bins = 32  # Number of histogram bins
    #
    #
    # out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    #
    # plt.imshow(out_img)