import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import cv2

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


if __name__ == '__main__':

    path_vehicle = 'data/vehicles/KITTI_extracted/'
    path_nonvehicle= 'data/non-vehicles/Extras/'
    # print(path_vehicle+"*.png")

    file_vehicle = glob.glob(path_vehicle+"*.png")
    file_nonvehicle = glob.glob(path_nonvehicle+"*.png")
    # print(file_vehicle)
    # print(file_nonvehicle)

    # show image
    # for i in range(3):
    #     im_vehicle = mpimg.imread(file_vehicle[i])
    #     im_nonvehicle = mpimg.imread(file_nonvehicle[i])
    #
    #     f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    #
    #     ax1.imshow(im_vehicle)
    #     ax1.set_title("Vehicle",fontsize=40)
    #     ax2.imshow(im_nonvehicle)
    #     ax2.set_title("Non-vehicle",fontsize=40)
    #
    #     plt.savefig('output_images/ex_img{:d}.png'.format(i))
    #     plt.show()

    # extract HOG feature
    im_idx = 0
    im_vehicle = mpimg.imread(file_vehicle[im_idx])
    im_nonvehicle = mpimg.imread(file_nonvehicle[im_idx])

    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    gray = cv2.cvtColor(im_vehicle,cv2.COLOR_RGB2GRAY)
    features,hog_image = get_hog_features(gray,orient,pix_per_cell,cell_per_block,vis=True)
    print(features.shape)
    print(hog_image.shape)