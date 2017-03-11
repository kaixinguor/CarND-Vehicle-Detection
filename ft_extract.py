import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage import img_as_ubyte

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    # feature pixel spatial distribution
    features = cv2.resize(img, size).ravel(order='F')
    return features


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


def color_hist(img, nbins=32, bins_range=(0, 256)):
    img = img_as_ubyte(img)

    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0],bins=nbins,range=bins_range)
    ghist = np.histogram(img[:,:,1],bins=nbins,range=bins_range)
    bhist = np.histogram(img[:,:,2],bins=nbins,range=bins_range)
    # Generating bin centers
    bin_centers = (rhist[1][1:]+rhist[1][0:-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0],ghist[0],bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


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
        f, axarr = plt.subplots(1, 2, figsize=(24, 9))
        im_vehicle = mpimg.imread(file_vehicle[i])
        im_nonvehicle = mpimg.imread(file_nonvehicle[i])

        ex_vehicle.append(im_vehicle)
        ex_nonvehicle.append(im_nonvehicle)

        axarr[0].imshow(im_vehicle)
        axarr[0].set_title("Vehicle",fontsize=40)
        axarr[1].imshow(im_nonvehicle)
        axarr[1].set_title("Non-vehicle",fontsize=40)

        plt.savefig('output_images/ex_img{:d}.png'.format(i))
    vis1 = np.concatenate(ex_vehicle, axis=0)
    vis2 = np.concatenate(ex_nonvehicle, axis=0)

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



    # # default parameter
    # orient = 9
    # pix_per_cell = 8
    # cell_per_block = 2
    #
    # # explore HOG features in different color space
    # img = im_vehicle[:,:,0]
    # print(img.shape)
    # features, hog_image = hog(img, visualise=True, feature_vector=False)
    # print(features.shape)
    # print(hog_image.shape)
    # f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    # ax1.imshow(img,cmap='gray')
    # ax1.set_title("channel image",fontsize=40)
    # ax2.imshow(hog_image,cmap='gray')
    # ax2.set_title("hog feature",fontsize=40)
    # plt.show()

    # explore HOG features for different parameters


    # gray = cv2.cvtColor(im_vehicle,cv2.COLOR_RGB2GRAY)
    # features,hog_image = get_hog_features(gray,orient,pix_per_cell,cell_per_block,vis=True)
    # print(features.shape)
    # print(hog_image.shape)
    # f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    # ax1.imshow(gray,cmap='gray')
    # ax1.set_title("gray image",fontsize=40)
    # ax2.imshow(hog_image,cmap='gray')
    # ax2.set_title("hog feature",fontsize=40)
    # plt.show()

    # gray = cv2.cvtColor(im_vehicle, cv2.COLOR_RGB2GRAY)
    #
    # orients = [6, 9, 12, 15, 18]
    # f, axarr = plt.subplots(1, len(orients)+1, figsize=(24, 9))
    # axarr[0].imshow(gray, cmap='gray')
    # axarr[0].set_title("gray image", fontsize=20)
    #
    # for i in range(len(orients)):
    #
    #     orient = orients[i]
    #     features,hog_image = get_hog_features(gray,orient,pix_per_cell,cell_per_block,vis=True)
    #     axarr[i+1].imshow(hog_image, cmap='gray')
    #     axarr[i+1].set_title("orient:{0:d}".format(orient), fontsize=20)
    #     print(features.shape)
    #     print(hog_image)
    #
    # plt.show()