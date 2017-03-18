import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage import img_as_ubyte



def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


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


def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


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
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
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
    # Return list of feature vectors
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
    # Return list of feature vectors
    return features


def hog_plot(img):
    # explore HOG features in different color space
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print("image shape: ", img.shape)
    features, hog_image = hog(img, visualise=True, feature_vector=False)

    print("hog image shape: ", hog_image.shape)
    print("hog feature shape: ", features.shape)

    f, (ax1,ax2) = plt.subplots(1,2,figsize=(9,9))
    ax1.imshow(img,cmap='gray')
    ax1.set_title("gray image",fontsize=20)
    ax2.imshow(hog_image,cmap='gray')
    ax2.set_title("hog feature",fontsize=20)


def bin_spatial_suv(img,size=(32,32)):
    color_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color_LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    color = np.dstack((color_HSV[:, :, 1], color_LUV[:, :, 1], color_LUV[:, :, 2]))

    #normalize
    color[:, :, 0] = 1/1.   *(color[:, :, 0])
    color[:, :, 1] = 1/354. *(color[:, :, 1] + 134.)
    color[:, :, 2] = 1/262. *(color[:, :, 2] + 140.)

    feature = bin_spatial(color, size=(32, 32))
    return color, feature


# def color_hist(img, nbins=32, bins_range=(0, 256)):
#     img = img_as_ubyte(img)
#
#     # Compute the histogram of the RGB channels separately
#     rhist = np.histogram(img[:,:,0],bins=nbins,range=bins_range)
#     ghist = np.histogram(img[:,:,1],bins=nbins,range=bins_range)
#     bhist = np.histogram(img[:,:,2],bins=nbins,range=bins_range)
#     # Generating bin centers
#     bin_centers = (rhist[1][1:]+rhist[1][0:-1])/2
#     # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((rhist[0],ghist[0],bhist[0]))
#     # Return the individual histograms, bin_centers and feature vector
#     return rhist, ghist, bhist, bin_centers, hist_features


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
    img = img_as_ubyte(img)

    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    img_small_RGB = img_small # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)

    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

    # Plot and show
    # Create figure and 3D axes
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot3d(ax1,img_small_RGB, img_small_rgb)
    ax1.set_title("RGB space", fontsize=20)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot3d(ax2,img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    ax2.set_title("HSV space", fontsize=20)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot3d(ax3,img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
    ax3.set_title("LUV space", fontsize=20)


def explore_color(vis1,vis2,ex_vehicle,ex_nonvehicle,num_ex):
    # display image example (concatenated)
    f, axarr = plt.subplots(1, 2, figsize=(10, 9))
    axarr[0].imshow(vis1)
    axarr[0].set_title("Vehicle", fontsize=20)
    axarr[1].imshow(vis2)
    axarr[1].set_title("Non-vehicle", fontsize=20)
    plt.savefig('output_images/ex_img.png')

    #display image example in color space
    display_color(vis1)
    plt.savefig('output_images/color_vehicle.png')

    display_color(vis2)
    plt.savefig('output_images/color_nonvehicle.png')


    # explore color space S U V

    # S [0,1] V [-134, 220] V [-140, 122]
    f, axarr = plt.subplots(3,3,figsize=(20, 12))
    str = 'USV'
    for i in range(num_ex):

        vehicle_color, vehicle_ft_color = bin_spatial_suv(ex_vehicle[i])
        # Plot features
        for j in range(3):
            axarr[i,j].plot(vehicle_ft_color[j*1024:(j+1)*1024])
            axarr[i,j].set_title('Vehicle{0:d} Spatial Feature {1:s} Channel '.format(i,str[j]))
    plt.savefig('output_images/ft_color_spatial_vehicle.png')


    f, axarr = plt.subplots(3, 3, figsize=(20, 12))
    for i in range(num_ex):

        nonvehicle_color, nonvehicle_ft_color = bin_spatial_suv(ex_nonvehicle[i])

        # Plot features
        for j in range(3):
            axarr[i, j].plot(nonvehicle_ft_color[j * 1024:(j + 1) * 1024])
            axarr[i, j].set_title('Non-vehicle{0:d} Spatial Feature {1:s} Channel '.format(i, str[j]))
    plt.savefig('output_images/ft_color_spatial_nonvehicle.png')


    # display color histogram
    f, axarr = plt.subplots(3, 3, figsize=(20, 12))
    for i in range(num_ex):
        vehicle_color, vehicle_ft_color = bin_spatial_suv(ex_vehicle[i])
        rh, gh, bh, bincen, feature_vec = color_hist(vehicle_color, nbins=32, bins_range=(0, 256))

        # Plot a figure with all three bar charts
        if rh is not None:

            axarr[i,0].bar(bincen, rh[0])
            axarr[i,0].set_title('Vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[0]))
            axarr[i,1].bar(bincen, gh[0])
            axarr[i,1].set_title('Vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[1]))
            axarr[i,2].bar(bincen, bh[0])
            axarr[i,2].set_title('Vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[2]))

        else:
            print('Your function is returning None for at least one variable...')
    plt.savefig('output_images/ft_color_hist_vehicle.png')

    f, axarr = plt.subplots(3, 3, figsize=(20, 12))
    for i in range(num_ex):
        nonvehicle_color, nonvehicle_ft_color = bin_spatial_suv(ex_nonvehicle[i])
        rh, gh, bh, bincen, feature_vec = color_hist(nonvehicle_color, nbins=32, bins_range=(0, 256))

        # Plot a figure with all three bar charts
        if rh is not None:

            axarr[i,0].bar(bincen, rh[0])
            axarr[i,0].set_title('Non-vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[0]))
            axarr[i,1].bar(bincen, gh[0])
            axarr[i,1].set_title('Non-vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[1]))
            axarr[i,2].bar(bincen, bh[0])
            axarr[i,2].set_title('Non-vehicle{0:d} Color Histogram {1:s} Channel '.format(i, str[2]))

        else:
            print('Your function is returning None for at least one variable...')
    plt.savefig('output_images/ft_color_hist_nonvehicle.png')


if __name__ == '__main__':

    path_vehicle = 'data/vehicles/KITTI_extracted/'
    path_nonvehicle= 'data/non-vehicles/Extras/'
    # print(path_vehicle+"*.png")

    file_vehicle = glob.glob(path_vehicle+"*.png")
    file_nonvehicle = glob.glob(path_nonvehicle+"*.png")
    print(file_vehicle)
    print(file_nonvehicle)

    # display image example (single)
    num_ex = 3
    ex_vehicle = []
    ex_nonvehicle = []
    for i in range(num_ex):

        im_vehicle = mpimg.imread(file_vehicle[i])
        im_nonvehicle = mpimg.imread(file_nonvehicle[i])

        ex_vehicle.append(im_vehicle)
        ex_nonvehicle.append(im_nonvehicle)

        # f, axarr = plt.subplots(1, 2, figsize=(24, 9))
        # axarr[0].imshow(im_vehicle)
        # axarr[0].set_title("Vehicle",fontsize=40)
        # axarr[1].imshow(im_nonvehicle)
        # axarr[1].set_title("Non-vehicle",fontsize=40)
        #
        # plt.savefig('output_images/ex_img{:d}.png'.format(i))

    vis1 = np.concatenate(ex_vehicle, axis=0)
    vis2 = np.concatenate(ex_nonvehicle, axis=0)

    # explore_color(vis1, vis2, ex_vehicle, ex_nonvehicle, num_ex)



    # default parameter
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    # explore HOG features in different color space
    # hog_plot(vis1)
    # plt.savefig('output_images/ft_hog_vehicle.png')
    # plt.show()
    # hog_plot(vis2)
    # plt.savefig('output_images/ft_hog_nonvehicle.png')
    # plt.show()


    # explore HOG features for different parameters


    # features,hog_image = get_hog_features(vis1,orient,pix_per_cell,cell_per_block,vis=True)
    # print(features.shape)
    # print(hog_image.shape)
    # f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    # ax1.imshow(gray,cmap='gray')
    # ax1.set_title("gray image",fontsize=40)
    # ax2.imshow(hog_image,cmap='gray')
    # ax2.set_title("hog feature",fontsize=40)
    # plt.show()


    # orients = [6, 8, 9, 10, 12]
    # pix_per_cell = 8
    # cell_per_block = 2
    # f, axarr = plt.subplots(1, len(orients)+1, figsize=(24, 9))
    # axarr[0].imshow(vis1)
    # axarr[0].set_title("original image", fontsize=20)
    # channel_gray = cv2.cvtColor(vis1, cv2.COLOR_RGB2GRAY)
    # for i in range(len(orients)):
    #
    #     orient = orients[i]
    #     features,hog_image = get_hog_features(channel_gray,orient,pix_per_cell,cell_per_block,vis=True)
    #     axarr[i+1].imshow(hog_image, cmap='gray')
    #     axarr[i+1].set_title("orient:{0:d}".format(orient), fontsize=20)
    #     print(features.shape)
    #     print(hog_image)
    # plt.savefig('output_images/ft_hog_orients.png')
    # plt.show()


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
        # print(features.shape)
        # print(hog_image)
    plt.savefig('output_images/ft_hog_pix_per_cell.png')
    plt.show()

    # orient = 9
    # cell_per_blocks = [1,2,3,4,5]
    # pix_per_cell = 8
    # f, axarr = plt.subplots(1, len(cell_per_blocks)+1, figsize=(24, 9))
    # axarr[0].imshow(vis1)
    # axarr[0].set_title("original image", fontsize=20)
    # channel_gray = cv2.cvtColor(vis1, cv2.COLOR_RGB2GRAY)
    # for i in range(len(cell_per_blocks)):
    #
    #     cell_per_block = cell_per_blocks[i]
    #     features,hog_image = get_hog_features(channel_gray,orient,pix_per_cell,cell_per_block,vis=True)
    #     axarr[i+1].imshow(hog_image, cmap='gray')
    #     axarr[i+1].set_title("cell_per_block:{0:d}".format(cell_per_block), fontsize=20)
    #     print(features.shape)
    #     print(hog_image.shape)
    # plt.savefig('output_images/ft_hog_cell_per_block.png')
    # plt.show()
