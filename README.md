## Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

## Writeup Report / README

This is the project 5 Vehicle Detection and Tracking in Self-Driving Car Nanodegree course by Udacity. The goal is to write a software pipeline to detect vehicles in a video.
For the original assignments can be found in [the project repository](https://github.com/udacity/CarND-Vehicle-Detection).

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
[image1]: ./examples/car_and_not_car.png
[image2]: ./examples/feature_extraction.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/heat_map.png
[image5]: ./examples/threshold_heatmap.png
[video1]: ./project_video_output.mp4

---
### Writeup / README
All the necessary files are included in this [Github folder](https://github.com/duongquangduc/Udacity-CarND-Vehicle-Detection)

### Histogram of Oriented Gradients (HOG)

I started by exploring some information of the data set. The set includes 8972 cars and 8968 non-car images. Here is an example of one of each of the `car` and `non-car` classes:

![alt text][image1]

To build a classifier, I used HOG feature and a `linear SVC`.  The model reaches over 99% of accuracy.
Here is the following codes for executing extracting HOG features and training the model.

* Extracting HOG features
`
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 32
pix_per_cell = 16
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t=time.time()
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
not_car_features = extract_features(not_cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
`

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=32`, `pixels_per_cell=16` and `cells_per_block=2`.
Alternative options is also listed in the pipeline.

* Training a classifier
I tried with different classifier such as a `LinearSVC`, a `MLPC_Classifier`...and they all reach very high accuracy.
The sample code for a `LinearSVC` is as below.

`
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)
svc = LinearSVC()
svc.fit(X_train, y_train)
`

To see what the feature extracted looks like, please see as below.

![alt text][image2]


### Sliding Window Search

The sliding window search implementation can be found in the code cell #8. I used an overlap of 50%, 96x96 windows for 400 <= y <= 656, 128x128 windows for 380 <= y <= 720 and, 
assuming it is known that the car is already on the most left lane, only searched for x values between 600 and 1280. The image below shows those windows (96x96 in blue and 128x128 in red):

![alt text][image3]

![alt text][image4]

To detect the false positive, first I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map 
to identify vehicle positions and discard the false positives. To smooth the detections I kept a sum of the heatmaps from the last 10 frames. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. Here are the example images with the positive detections, their corresponding heat maps and their resulting bounding boxes:

![alt text][image5]


### Video Implementation

#### Provide a link to your final video output. 
I combined the vehicle detection pipeline with a lane_detection which is implemented in project 4 for the same project.
Here's a [link to my video result](./project_video_output.mp4). Or you can watch it by clicking the below thumbnail.

[![Alt text](http://img.youtube.com/vi/kjy1slZQOAY/0.jpg)](https://youtu.be/kjy1slZQOAY)

---

### Discussion

Though the final output video showed that vehicles are detected, it takes lots of effort to implement the algorithm, especially in the feature engineering phase.
Anyway, I learnt a lot in this project to implement a machine learning technique from beginning to end. For mastering these machine techniques, I consider the following things to do with
this project;
* Implement the pipeline to different sets of images
* Try on different videos/lanes
* Research and implement with Deep Learning approach, such as using the YOLO or SSD, which are very popular in deep learning model for computer vision.


