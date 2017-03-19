import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import cv2
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from ft_extract import get_hog_features
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
from ft_extract import extract_features, extract_features_from_image


def get_image_list(nsample=500):

    # path_vehicle = 'data/vehicles/KITTI_extracted/'
    folders_vehicle = ['data/vehicles/GTI_Far/', 'data/vehicles/GTI_MiddleClose/', 'data/vehicles/GTI_Left/',
                       'data/vehicles/GTI_Right/', 'data/vehicles/KITTI_extracted/']
    folders_nonvehicle = ['data/non-vehicles/GTI/','data/non-vehicles/Extras/']
    # print(path_vehicle+"*.png")

    cars = []
    for f in folders_vehicle:
        cars += glob.glob(f + "*.png")
    notcars = []
    for f in folders_nonvehicle:
        notcars += glob.glob(f + "*.png")

    np.random.seed(123456789)
    sample_cars = np.random.choice(cars,nsample,replace=False)
    sample_notcars = np.random.choice(notcars,nsample,replace=False)

    return sample_cars, sample_notcars


def get_features(imgs,config):

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

    features = extract_features(imgs, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    return features

def get_features_from_image(image,config):

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

    features = extract_features_from_image(image, cspace=colorspace, spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    return features


def normalize_data(car_features,notcar_features):

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))

    return X_scaler, X_train, X_test, y_train, y_test


def train_clf(X_train, X_test, y_train, y_test):

    # Use a linear SVC
    clf = LinearSVC()

    # Check the training time for the classifier
    t = time.time()
    clf.fit(X_train, y_train)
    print(round(time.time() - t, 2), 'Seconds to train classifier...')

    # Check the score of the classifier
    print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My classifier predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    print(round(time.time() - t, 5), 'Seconds to predict', n_predict, 'labels with classifier')

    return clf


if __name__ == '__main__':

    t = time.time()
    cars, notcars = get_image_list(500)
    print(round(time.time() - t, 2), 'Seconds to load data ...')

    config = {'colorspace':'YCrCb','spatial_size':(32, 32),'hist_bins':32,
              'orient':9,'pix_per_cell':8,'cell_per_block': 2,
              'hog_channel':'ALL','spatial_feat':True,'hist_feat':True,'hog_feat':True}

    print('Using:',config['colorspace'], 'colorspace,', config['spatial_size'], 'spatial_size,', config['hist_bins'],'hist_bins',
          config['orient'],'orientations,',config['pix_per_cell'],
        'pixels per cell and', config['cell_per_block'],'cells per block')

    t = time.time()
    car_features = get_features(cars,config)
    notcar_features = get_features(notcars, config)
    print(round(time.time()-t, 2), 'Seconds to extract HOG features...')

    t = time.time()
    print(len(car_features))
    print(len(notcar_features))
    X_scaler, X_train, X_test, y_train, y_test = normalize_data(car_features,notcar_features)
    print(round(time.time() - t, 2), 'Seconds to normalize features...')

    t = time.time()
    clf = train_clf(X_train, X_test, y_train, y_test)
    print(round(time.time() - t, 2), 'Seconds to train and test...')

    with open('output_images/car_clf.pkl', 'wb') as f:
        pickle.dump([clf,X_scaler,config], f)