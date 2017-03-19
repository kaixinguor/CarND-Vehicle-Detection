import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D
import cv2


colors = ['RGB','HSV','LUV','HLS','YUV','YCrCb']
color_ranges = {'RGB': [(0, 1), (0, 1), (0, 1)],
          'HSV': [(0, 360), (0, 1), (0, 1)],
          'LUV': [(0, 100), (-134, 220), (-140, 122)],
          'HLS': [(0, 360), (0, 1), (0, 1)],
          'YUV': [(0, 1), (0, 1), (0, 1)],
          'YCrCb': [(0, 1), (0, 1), (0, 1)]}


def convert_color(img,  color_space='RGB'):

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32,color_space='RGB'):    #bins_range=(0, 256)

    range = color_ranges[color_space]
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=range[0])
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=range[1])
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=range[2])

    bin_centers = (channel1_hist[1][1:] + channel1_hist[1][0:-1]) / 2

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features, [channel1_hist[0],channel2_hist[0],channel3_hist[0]],bin_centers


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    #1) Define an empty list to receive features
    img_features = []

    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)

    #3) Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    #4) Compute histogram features if flag is set
    if hist_feat is True:
        hist_features,_,_ = color_hist(feature_image, nbins=hist_bins,color_space=color_space)
        img_features.append(hist_features)

    #5) Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    #6) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_from_image(image, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    img_features = single_img_features(image, cspace, spatial_size,
                    hist_bins, orient,pix_per_cell, cell_per_block, hog_channel,
                    spatial_feat, hist_feat, hog_feat)

    return img_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    # Create a list to append feature vectors to
    features = []
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file) # png image range [0,1]!!!
        # print(np.max(image))
        img_features = single_img_features(image, cspace, spatial_size,
                        hist_bins, orient,pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat)
        features.append(img_features)

    return features


def plot3d(ax,pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # # Create figure and 3D axes
    # fig = plt.figure(figsize=(8, 8))
    #ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def display_color(img):
    # here image should be float type

    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)

    fig = plt.figure(figsize=(24, 8))
    idx = 0

    for color in colors:
        # Convert subsampled image to desired color space(s)
        img_small_color = convert_color(img_small, color_space=color)
        print("color", color, np.max(img_small_color))

        # Plot and show
        # Create figure and 3D axes
        idx += 1
        ax = fig.add_subplot(2, 3, idx, projection='3d')

        if color != 'YCrCb':
            axis_labels = list(color)
        else:
            axis_labels = ['Y','Cr','Cb']

        plot3d(ax,img_small_color, img_small, axis_labels = axis_labels, axis_limits = color_ranges[color])

        ax.set_title(color, fontsize=20)


def explore_color(vis1, vis2, example_vehicle, example_nonvehicle, num_example):
    # display image example (concatenated)
    f, axarr = plt.subplots(1, 2, figsize=(10, 9))
    axarr[0].imshow(vis1)
    axarr[0].set_title("Vehicle", fontsize=20)
    axarr[1].imshow(vis2)
    axarr[1].set_title("Non-vehicle", fontsize=20)
    plt.savefig('output_images/example_img.png')

    # display image example in color space
    display_color(vis1)
    plt.savefig('output_images/3d_vehicle.png')

    display_color(vis2)
    plt.savefig('output_images/3d_nonvehicle.png')

    for color in colors:

        f, axarr = plt.subplots(3, 3, figsize=(20, 12))
        for i in range(num_example):
            img_car = convert_color(example_vehicle[i], color_space=color)
            vehicle_color = bin_spatial(img_car,size=(32, 32))
            # Plot features
            for j in range(3):
                axarr[i, j].plot(vehicle_color[j * 1024:(j + 1) * 1024]) # 32*32=1024
                axarr[i, j].set_title('Vehicle {0:d} '.format(i) + color + ' spatial channel {0:d}'.format(j+1))
        plt.savefig('output_images/color_spatial_' + color + '_vehicle.png')

        f, axarr = plt.subplots(3, 3, figsize=(20, 12))
        for i in range(num_example):
            img_norcar = convert_color(example_nonvehicle[i], color_space=color)
            nonvehicle_color =  bin_spatial(img_norcar,size=(32, 32))
            # Plot features
            for j in range(3):
                axarr[i, j].plot(nonvehicle_color[j * 1024:(j + 1) * 1024])
                axarr[i, j].set_title('Non-vehicle {0:d} '.format(i) + color + ' spatial channel {0:d}'.format(j+1))
        plt.savefig('output_images/color_spatial_' + color + '_nonvehicle.png')

        # display color histogram
        f, axarr = plt.subplots(3, 3, figsize=(20, 12))
        for i in range(num_example):
            img_car = convert_color(example_vehicle[i], color_space=color)
            feature_vec, hists, bincen = color_hist(img_car, nbins=32,color_space = color)

            # Plot a figure with all three bar charts
            if hists[0] is not None:
                for ichannel in range(3):
                    axarr[i, ichannel].bar(bincen, hists[ichannel],width=(bincen[1] - bincen[0]) / 2.)
                    axarr[i, ichannel].set_title('Vehicle {0:d} '.format(i) + color + ' histogram channel {0:d}'.format(ichannel+1))
            else:
                print('Your function is returning None for at least one variable...')
        plt.savefig('output_images/color_hist_' + color + '_vehicle.png')

        f, axarr = plt.subplots(3, 3, figsize=(20, 12))
        for i in range(num_example):
            img_norcar = convert_color(example_nonvehicle[i], color_space=color)
            feature_vec, hists, bincen = color_hist(img_norcar, nbins=32,color_space = color)

            # Plot a figure with all three bar charts
            if hists[0] is not None:
                for ichannel in range(3):
                    axarr[i, ichannel].bar(bincen, hists[ichannel],width=(bincen[1] - bincen[0]) / 2.)
                    axarr[i, ichannel].set_title('Vehicle {0:d} '.format(i) + color + ' histogram channel {0:d}'.format(ichannel+1))
            else:
                print('Your function is returning None for at least one variable...')
        plt.savefig('output_images/color_hist_' + color + '_nonvehicle.png')


def explore_hog(vis1,vis2):

    for color in colors:

        # car
        img_car = convert_color(vis1, color_space=color)

        f, ax_arr = plt.subplots(2, 3, figsize=(6, 9))
        for ichannel in range(3):

            features, hog_image = hog(img_car[:, :, ichannel], visualise=True, feature_vector=False)

            ax_arr[0,ichannel].imshow(img_car[:, :, ichannel],cmap='gray')
            ax_arr[0,ichannel].set_title(color + "image", fontsize=20)

            ax_arr[1,ichannel].imshow(hog_image, cmap='gray')
            ax_arr[1,ichannel].set_title("hog feature", fontsize=20)
        plt.savefig('output_images/hog_' + color + '_car.png')

        # not car
        img_notcar = convert_color(vis2, color_space=color)

        f, ax_arr = plt.subplots(2, 3, figsize=(6, 9))
        for ichannel in range(3):
            features, hog_image = hog(img_notcar[:, :, ichannel], visualise=True, feature_vector=False)

            ax_arr[0,ichannel].imshow(img_notcar[:, :, ichannel],cmap='gray')
            ax_arr[0,ichannel].set_title(color + "image", fontsize=20)

            ax_arr[1,ichannel].imshow(hog_image, cmap='gray')
            ax_arr[1,ichannel].set_title("hog feature", fontsize=20)
        plt.savefig('output_images/hog_' + color + '_notcar.png')


if __name__ == '__main__':


    path_vehicle = 'data/vehicles/KITTI_extracted/'
    path_nonvehicle= 'data/non-vehicles/Extras/'


    file_vehicle = glob.glob(path_vehicle+"*.png")
    file_nonvehicle = glob.glob(path_nonvehicle+"*.png")


    # display image example (single)
    num_ex = 3
    example_vehicle = []
    example_nonvehicle = []
    for i in range(num_ex):

        im_vehicle = mpimg.imread(file_vehicle[i])
        print(np.max(im_vehicle))
        im_nonvehicle = mpimg.imread(file_nonvehicle[i])

        example_vehicle.append(im_vehicle)
        example_nonvehicle.append(im_nonvehicle)

        f, axarr = plt.subplots(1, 2, figsize=(24, 9))
        axarr[0].imshow(im_vehicle)
        axarr[0].set_title("Vehicle",fontsize=40)
        axarr[1].imshow(im_nonvehicle)
        axarr[1].set_title("Non-vehicle",fontsize=40)

        plt.savefig('output_images/example_img{:d}.png'.format(i))

    vis1 = np.concatenate(example_vehicle, axis=0)
    vis2 = np.concatenate(example_nonvehicle, axis=0)


    # # explore color space
    explore_color(vis1, vis2, example_vehicle, example_nonvehicle, num_ex)

    exit(0)


    # # explore HOG features in different color space
    explore_hog(vis1,vis2)


    # explore HOG features for different parameters
    orients = [6, 8, 9, 10, 12]
    pix_per_cell = 8
    cell_per_block = 2
    f, axarr = plt.subplots(1, len(orients)+1, figsize=(24, 9))
    axarr[0].imshow(vis1)
    axarr[0].set_title("original image", fontsize=20)
    channel_gray = cv2.cvtColor(vis1, cv2.COLOR_RGB2GRAY)
    for i in range(len(orients)):

        orient = orients[i]
        features,hog_image = get_hog_features(channel_gray,orient,pix_per_cell,cell_per_block,vis=True)
        axarr[i+1].imshow(hog_image, cmap='gray')
        axarr[i+1].set_title("orient:{0:d}".format(orient), fontsize=20)
    plt.savefig('output_images/hog_para_orients.png')


    orient = 9
    cell_per_block = 2
    pix_per_cells = [4, 6, 8, 10, 12]
    f, axarr = plt.subplots(1, len(pix_per_cells)+1, figsize=(24, 9))
    axarr[0].imshow(vis1)
    axarr[0].set_title("original image", fontsize=20)
    channel_gray = cv2.cvtColor(vis1, cv2.COLOR_RGB2GRAY)
    for i in range(len(pix_per_cells)):

        pix_per_cell = pix_per_cells[i]
        features,hog_image = get_hog_features(channel_gray,orient,pix_per_cell,cell_per_block,vis=True)
        axarr[i+1].imshow(hog_image, cmap='gray')
        axarr[i+1].set_title("pix_per_cell:{0:d}".format(pix_per_cell), fontsize=20)
    plt.savefig('output_images/hog_para_pix_per_cell.png')


    orient = 9
    cell_per_blocks = [1,2,3,4,5]
    pix_per_cell = 8
    f, axarr = plt.subplots(1, len(cell_per_blocks)+1, figsize=(24, 9))
    axarr[0].imshow(vis1)
    axarr[0].set_title("original image", fontsize=20)
    channel_gray = cv2.cvtColor(vis1, cv2.COLOR_RGB2GRAY)
    for i in range(len(cell_per_blocks)):

        cell_per_block = cell_per_blocks[i]
        features,hog_image = get_hog_features(channel_gray,orient,pix_per_cell,cell_per_block,vis=True)
        axarr[i+1].imshow(hog_image, cmap='gray')
        axarr[i+1].set_title("cell_per_block:{0:d}".format(cell_per_block), fontsize=20)
        print(features.shape)
        print(hog_image.shape)
    plt.savefig('output_images/hog_para_cell_per_block.png')
