##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)


[image1_1]: ./output_images/ex_img.png
[image1_2]: ./output_images/color_vehicle.png
[image1_3]: ./output_images/color_nonvehicle.png
[image1_4_1]: ./output_images/ft_color_spatial_vehicle.png
[image1_4_2]: ./output_images/ft_color_spatial_nonvehicle.png
[image1_5_1]: ./output_images/ft_color_hist_vehicle.png
[image1_5_2]: ./output_images/ft_color_hist_nonvehicle.png
[image1_6_1]: ./output_images/ft_hog_vehicle.png
[image1_6_2]: ./output_images/ft_hog_nonvehicle.png
[image1_7_1]: ./output_images/ft_hog_orients.png
[image1_7_2]: ./output_images/ft_hog_pix_per_cell.png
[image2_1]: ./output_images/detect_multiscale_window.png
[image2_2]: ./output_images/detect_hot_window.png


[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines # through # of the file called `ft_extract.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is some examples of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1_1]

I plot the three samples in different color space: RGB, HSV and LUV, to get a feel of the distribution of  color.

The distribution of vehicle pixels in different color space looks like:

![alt text][image1_2]

And the distribution of non-vehicle pixels in different color space looks like:

![alt text][image1_3]

It seems that the vehicle and non-nonvehicle samples are more separated in Satuation channel and in UV-space.

Then I choose these three colors and continue to explore spatial features and histogram features.
The spatial features for vehicle samples like this


![alt text][image1_4_1]

The spatial features for non-vehicle samples like this

![alt text][image1_4_2]

The histogram features for vehicle samples like this

![alt text][image1_5_1]

The histogram features for non-vehicle samples like this
![alt text][image1_5_2]


These figures show that spatial color is not very robust to position variation (comparing different samples). And U and V channel values are centered so that they are not very informative for discriminating car versus non-cars.

The conclusion is, satuation color histgram may be disriminative for car and non-car.

----

Next thing I explore is the parameter of HOG features. Since HOG is about the graident, I first convert the image into gray scale　using `cv2.cvtColor`.

HOG for vehicle

![alt text][image1_6_1]

HOG for non-vehicle

![alt text][image1_6_2]


####2. Explain how you settled on your final choice of HOG parameters.

I visualized HOG image of using different `orientations`:
![alt text][image1_7_1]

and different`pixels_per_cell`:

![alt text][image1_7_2]

Seems that `orientations` does not look sensitive and  `pixels_per_cell=(8, 8)` is a potential good choice.
another parameter `cells_per_block=(2, 2)` is for normalization, I just use the value used in the lecture.

Finally, I chose gray scale color space and set HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

Except for the performance, these parameters also affect feature length, thus cause different usage of memory and time computation.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features. The code is in the file `./car_train.py`.

I read `car` images and `notcar` images, extract HOG features from R,G,B channels respectively, and concatenate them together to get feature vectors.

Then I split data into 80% training and 20% test by `sklearn.cross_validation.train_test_split()`. `sklearn.svm.LinearSVC` is used for training and test. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use a multi-scale sliding window search to be the candidate windows. The code is in the file `detect_car.py`


![alt text][image2_1]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2_2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

