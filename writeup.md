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


[image1_1]: ./output_images/example_img.png
[image1_2]: ./output_images/3d_vehicle.png
[image1_3]: ./output_images/3d_nonvehicle.png
[image1_4_1]: ./output_images/color_spatial_YCrCb_vehicle.png
[image1_4_2]: ./output_images/color_spatial_YCrCb_nonvehicle.png
[image1_5_1]: ./output_images/color_hist_HLS_vehicle.png
[image1_5_2]: ./output_images/color_hist_HLS_nonvehicle.png
[image1_6_1]: ./output_images/hog_YCrCb_car.png
[image1_6_2]: ./output_images/hog_YCrCb_notcar.png
[image1_7_1]: ./output_images/hog_para_orients.png
[image1_7_2]: ./output_images/hog_para_pix_per_cell.png
[image1_7_3]: ./output_images/hog_para_cell_per_block.png
[image2_1]: ./output_images/detect_multiscale_window.png
[image2_2]: ./output_images/detect_hot_window1.png
[image2_3]: ./output_images/detect_hot_window2.png

[image3_0]: ./output_images/detect_single_frame/0970.png
[image3_1]: ./output_images/detect_single_frame/0971.png
[image3_2]: ./output_images/detect_single_frame/0972.png
[image3_3]: ./output_images/detect_single_frame/0973.png
[image3_4]: ./output_images/detect_single_frame/0974.png

[image4_1]: ./output_images/0974.png


[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_vfideo.mp4


###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code about feature extraction is in the file `./ft_extract.py`.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is some examples of `vehicle` and `non-vehicle` classes:

![alt text][image1_1]

I plot these samples in different color space: `RGB`, `HSV`, `LUV`, `HLS`, `YUV`, `YCrCb` to get a feel of the distribution of  color.

I use the default parameter settings in the lecture to set color spatial feature and color histogram features.
With binning technology, the color information is less sensitive to  position variation (comparing different samples). 

The distribution of vehicle pixels in different color spaces looks like:

![alt text][image1_2]

And the distribution of non-vehicle pixels in different color spaces looks like:

![alt text][image1_3]

It seems that non-vehicle images are more homogeneously distributed, as shown in the sample images.

It's not clear which color space is better, so I continue to explore spatial features and histogram features.
In the following, I use YCrCb color space as an example.

The spatial features for vehicle samples like this


![alt text][image1_4_1]

The spatial features for non-vehicle samples like this

![alt text][image1_4_2]

The histogram features for vehicle samples like this

![alt text][image1_5_1]

The histogram features for non-vehicle samples like this
![alt text][image1_5_2]

This seems to show some patterns which are different between car and non-car classes while similar in the samples in the same class. A hint is that color provide some useful information.

----

The next thing I explore is the parameter of HOG features. The implementation of getting HOG feature is in lines 37 through 54 in the file `./ft_extract.py`.

Since HOG is about the graident, I first convert the image into `YCrCb` space using `cv2.cvtColor`. The HOG for each channel is like this


![alt text][image1_6_1]

HOG for non-vehicle is like this

![alt text][image1_6_2]


####2. Explain how you settled on your final choice of HOG parameters.

I visualized HOG image of using different `orientations`:
![alt text][image1_7_1]

Small orientation does not give enough information while big one describe more detailed orientation information but could be very sensitive. So choose `9` as compliment number.


The HOG image with different`pixels_per_cell` looks like this:

![alt text][image1_7_2]

The image looks good with `8 pixels_per_cell`.

The HOG image with different`cell_per_block` look like this.

![alt text][image1_7_3]

This parameter will influence normalization. To make the HOG more robust, I choose `2` as the parameter used in the exepriments.


In conclusion, the parameters for HOG features are selected as follows:
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
The idea is to provide as much as information as possible while be more robust.
 
So in the following experiments, I use these parameter setting and combine HOG, color spatial feature and color histogram feature as the feature vector to learn a classifier.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features. The code is in the file `./car_train.py`.

I read `car` images and `notcar` images, extract HOG features from YCrCb color space, and concatenate them together to get feature vectors.

Then I split data into 80% training and 20% test by `sklearn.model_selection.train_test_split()`. `sklearn.svm.LinearSVC` is used for training and test.

For training, I only use 500 car samples and 500 non-car samples for the sake of time. The feature length is 8460. Test accuracy is 98%.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use a multi-scale sliding window search to find candidate windows. The code is in lines 10 through 37 in the function `car_detect()` in the file `./car_detect.py`.

![alt text][image2_1]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The output of `car_detect` looks like this:

![alt text][image2_2]

The result looks good, because the car is found with multiple boxes with different scales. Then I found using this method is relatively slow: it cost around 1.5 seconds to process one image.

Then I tried the method of computing HOG on the whole image and then get hog for search window size, as taught in the lecture. The result looks like this.

![alt text][image2_3]

The result has slightly changed since the scale is different. But the result is still good enough at this stage and it only costs 0.5 seconds to process one image


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code that processes each frame of the video and gets detections on each frame is in the file `./main_car_detect.py`. 
 
I recorded the positions of positive detections in each frame of the video, and save it in the file `./output_images/car_box.pkl` for later use.

From the single frame detection result, we can see duplicate detections and false positives. Apparently duplicate detections appear much more frequent than false positives in consecutive frames.

### Here are five frames and their corresponding heatmaps:

![alt text][image3_0]
![alt text][image3_1]
![alt text][image3_2]
![alt text][image3_3]
![alt text][image3_4]


So for every frame, I take detections in previous 20 frames (including current frame) and accumulate the boxes in each frame, then I build a heatmap to indicate how many times each pixel is identified as a car. I mark those pixels which are identified at least 5 times as car pixel, then use `scipy.ndimage.measurements.label()` identify individual blobs in the heatmap.  Then I assume each blob correspond to a vehicle. 

The accumulated heatmap and the final bounding box is like this:

![alt text][image4_1]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


* color space
The color space I chose is somewhat intuitive. It deserves to explore more deeply in the future which color is really better for the car-detection and whether combine from different color spaces will help.
 
* dataset choice

In the beginning I use only KITTI dataset and found the detector get quite high test accuracy while the result on test image is bad. After I add GTI dataset, the result is better. Then I looked into the image and found that KITTI dataset is more different from the test image compared to GTI dataset. So training with similar images is important.

* png and jpg scale

One nees to be careful about the range of the image. opencv function `cv2.cvtColor()` will have different behavior on different type of image.

 

* sliding window speed
In the experiments, avoid computing HOG features for every candidate window saved around 65% time. The similar thing could be done for other features.
Detection rate would be very important for the application.
