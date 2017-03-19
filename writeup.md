## Vehicle Detection Project

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
[image2_2]: ./output_images/test_hot_windows.png
[image2_3]: ./output_images/detect_hot_window2.png
[image2_4]: ./output_images/detect_hot_window3.png

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

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is some examples of `vehicle` and `non-vehicle` classes:

![alt text][image1_1]



###Histogram of Oriented Gradients (HOG)

The code for this section is in the file `./ft_extract.py`.

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I use `skimage.feature.hog()`to extract HOG features. The code is in the function `get_hog_features()` in lines 37 through 54 in the file `./ft_extract.py`.

The way how I use it is described in lines 108 through 118 in the function `single_img_features()` in `./ft_extract.py`.  
I first convert an image into some color space using `cv2.cvtColor()`, then I feed one channel image into  `get_hog_features()`. 

The following image shows examples of HOGs computed in respective channels in `YCrCb` space for cars:

![alt text][image1_6_1]

And the following image shows examples of HOGs for non-cars:

![alt text][image1_6_2]




####2. Explain how you settled on your final choice of HOG parameters.

I explored three parameters `orientations`, `pixels_per_cell` and `cells_per_block` for computing HOG features. The code is in the function `explore_hog()` in lines 318 through 366 in `./ft_extract.py`.

I convert image into gray scale and feed it into hog computation.

I visualize HOG images of different `orientations`:
![alt text][image1_7_1]

Small value of this parameter (e.g. `orientations=6 or 8`) does not contain enough bins to capture specific orientation information of car gradients (e.g., horizontal gradients go to bins in which gradients with obvious angle are dominant).

When `orientations>=9`, the gradient looks reasonable. But when the value goes bigger, the feature may however become sensitive to in-plain rotation (e.g. when the camera is not exactly horizontal) . So I set `orientations=9` to keep enough information of gradient orientation while more invariant to slight rotation.

The HOG images with different`pixels_per_cell` look like this:

![alt text][image1_7_2]

This parameter influences the robustness of local statistics of orientation. Small value gives more detailed configuration on spatial distribution of gradients, while it makes the feature less robust. And it will also increase fast the size of the features thus decrease feature extraction speed. To get a balance, I choose `pixels_per_cell = 8`.

The HOG images with different`cell_per_block` look like this.

![alt text][image1_7_3]

This parameter will influence normalization. I choose `cell_per_block=2` so that the orientation values are normalized in every 2-by-2 block.

In conclusion, the parameters for HOG features are selected as follows:
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
The idea is to provide as much information as possible while being robust.
Applying on a `64*64` image, this setting will give `8*8` cells, every cell gives `9` dimention feature vector, and every neighboring `2*2` cells are normalized. So the final featue vector has `7*7*2*2*9 = 1764` dimension vector.




####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


I trained a linear SVM using HOG features combined with color spatial feature and color histogram feature (the analysis of color feature is at the end of this document).

The code is in the file `./car_train.py`. The code for training pipeline is in lines 123 through 158. 

I first set parameters for extracting features. 

Then I read `car` images and `notcar` images. For saving computation time, I only use 500 car images and 500 non-car images. They are randomly sampled from the respective dataset.

Then I extract features from all channels of YCrCb color space, and concatenate them together to get feature vectors.

The data are splitted into 80% training and 20% test by `sklearn.model_selection.train_test_split()`.

The data are normalized in the function `normalize_data()`. 

 `sklearn.svm.LinearSVC` is used for training and test.

 The total feature length is 8460 (Each channel contains `1764` HOG features, `32*32=1024` color spatial features and `32` color histogram features. So three channels contain in total `(1764+1024+32)*3=8460`). Test accuracy is 98%.

Finally, the trained classifer, normalization information and parameters are saved in the file `./output_images/car_clf.pkl` for later use.



###Sliding Window Search


The code for this part is in the file `./car_detect.py`.


####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


The sliding window search is implemented in the function `slide_window()` in the file `./car_detect.py` in lines 39 through 79. The search process will generate a list of box by moving a window (the size of the window is set in the parameter `xy_window`) from left to right, from top to bottom. The step of moving is specified by the parameter `xy_overlap`. An overlap of `0.5` means `50%` overlapping; `0.25` means `75%` overlapping between neighboring windows.
 
This function can limit searching within a part of the image depending on the parameters of `x_start_stop` and `y_start_stop`. 

In the experiment the overlapping is set to `75%`.



In order to capture cars with different sizes, I use a multi-scale sliding window search to find candidate windows. The code is in lines 11 through 33 in the function `car_detect()` in the file `./car_detect.py`. 

I use four kinds of window size `[80,96,112,128]` by trial and test. I plot different sizes of box in the test images and compare them with different sizes of cars.

In order to reduce the computation of sliding window and reject potentially false positives, I limit the search area in the parameter `y_start_stops`. The idea is that the cars appeared in the middle of the image tend to be smaller than those appeared in the bottom of the image. So we the effective searching area needs to be adaptive to the window size. 


The multi-scale windows I used look like this:
 
![alt text][image2_1]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The output of detection on some test images looks like this:

![alt text][image2_2]

The result looks good, because most of the time the bounding boxes are surrounding the cars instead of the background. 

But there are several problems:

** There are multiple boxes on the same car **

** There are sometimes false positives on the background **

** The computation is very slow: close to 6 seconds to process one image with  `75%` overlap **

The first two problem will be handled in the next section when we process a video. In the following I discuss about the computation.

It will takes around 1.5 seconds to process one image with `50%` overlap, but when the candidate windows become sparse, the performance of detection may drop, i.e. car is more likely missed if the box is not align well with the car.
So reducing the number of searching windows may not be a good idea.

Since computing the HOG feature could be one bottleneck for the computation, I tried another searching method provided by the lecture: the idea is to compute HOG image on the whole image only once, and then the hog features for each searching window are computed directly from HOG image. 

The code is in the function `find_car()` in lines 127 through 227 in `./car_detect.py`.

The result of using this method in a single scale (equivalent with using window size `96`) looks like this.

![alt text][image2_3]


A multi-scale search can be done using this techniques and save a lot of time. 
I implement this part in `find_car_multiscale()` in lines 230 through 243 in `./car_detect.py`. With the same setting of `car_detect()` which computesHOG for every individual searching window. This fucntion cost only 1 second (compared to 6 seconds) to process one image. The results look like this

![alt text][image2_4]
The results of these two methods are very similar but not exactly the same due to the different downsampling strategy.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline which processes each frame of the video and gets detections on each frame is implemented in `./main_car_detect.py`. 
 
I recorded the positions of positive detections in each frame of the video, and save it in the file `./output_images/car_box.pkl` for later use.

From the example detection result shown in previous section, we can see duplicate detections and false positives. Apparently duplicate detections appear much more frequent than false positives in consecutive frames.

So I compute a heatmap for every frame (see function `add_heat()` in lines 15 through 23 in `./main_car_tracking.py`).
The result of this shows that how many times that every pixel is identified as car inside one bounding box.

### Here are five frames and their corresponding heatmaps:

![alt text][image3_0]
![alt text][image3_1]
![alt text][image3_2]
![alt text][image3_3]
![alt text][image3_4]

Since single frame detection is not very robust, I accumulate heat across consecutive images in one video, so that the detection knowledge is accumulated over time.

More specifically, for every frame, I take detection boxes from previous 20 frames (including the current frame) and accumulate the heat, then I apply a threshold filtering in the function `apply_threshold()` in lines 26 through 30 in `./main_car_tracking.py`. I consider those pixels which are identified at least 5 times as true car pixel. Then  `scipy.ndimage.measurements.label()` is applied to identify individual blobs in the heatmap.  I assume each blob correspond to one vehicle. This procedure is in lines 102 through 118 in  `./main_car_tracking.py`.

The accumulated heatmap and the final bounding box look like this:

![alt text][image4_1]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


* Color space 
The color space I chose is somewhat intuitive. It deserves to explore more deeply in the future which color is really better for the car-detection and whether combine from different color spaces will help.
 
* Dataset choice

In the beginning I use only KITTI dataset and found the detector get quite high test accuracy while the result on test image is bad. After I add GTI dataset, the result looks better. Then I looked into the image and found that KITTI dataset is more different from the test image compared to GTI dataset. So training with similar images (information distribution) is important especially when the dataset is not large.

* png and jpg scale

One nees to be careful about the range of the image. opencv function `cv2.cvtColor()` may have different behaviors on different types of image.
 

* Sliding window speed
In the experiments, avoid computing HOG features for every candidate window saved more than 80% time. The similar thing could be done for other features. Besides, one of the state-of-the-arts detector ACF (P. Dollar, R. Appel, S. Belongie and P. Perona, "Fast Feature Pyramids for Object Detection," PAMI 2014.) involves the simiar idea to approximate HOG features in different scales instead of compute them from images in order to achieve 30fps detection rate. Detection rate would be very important for the later tracking process too.

* By looking at the project video result, two typical failure cases could be found:

One is accasionally false positives due to consistent false detection over time. As the parameter of filters are heuristic.

Another cass is when two cars are very close or even occluded, this technology will group two cars together into one large bounding box. Constraints on the final bounding box size could be helpful to get more refined results. 


### Appendix

### Color information

####1. Here is some exploration about color features from the training images.

The code about feature extraction is in the file `./ft_extract.py`.

I plot data samples in different color space: `RGB`, `HSV`, `LUV`, `HLS`, `YUV`, `YCrCb` to get a feel of the distribution of  color.

The distribution of vehicle pixels in different color spaces looks like:

![alt text][image1_2]

And the distribution of non-vehicle pixels in different color spaces looks like:

![alt text][image1_3]

It seems that non-vehicle images are more homogeneously distributed, as shown in the sample images.

It's not clear which color space is better, so I continue to explore spatial features and histogram features.

The code to compute color spatial features is in the function `bin_spatial()` in lines 57 through 63 in `./ft_extract.py`.
The code to compute color histogram features is in the function `color_hist()` in lines 66 through 80 in `./ft_extract.py`.

I use the default parameter settings in the lecture to set color spatial feature and color histogram features.
With binning technology, the color information is less sensitive to  position variation (by comparing different samples). 


In the following, I use YCrCb color space as an example.

The spatial features for vehicle samples like this


![alt text][image1_4_1]

The spatial features for non-vehicle samples like this

![alt text][image1_4_2]

The histogram features for vehicle samples like this

![alt text][image1_5_1]

The histogram features for non-vehicle samples like this
![alt text][image1_5_2]

The above figures show some patterns which are different between car and non-car classes while similar for the samples in the same class. A hint is that colors provide some useful information. However more investigation is needed.

----
