00## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/image240.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/search_windows.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/image0067.png
[image10]: ./output_images/1RGB2HLS.png
[image11]: ./output_images/1RGB2HSV.png
[image12]: ./output_images/1RGB2LUV.png
[image13]: ./output_images/1RGB2YCrCb.png
[image14]: ./output_images/1RGB2YUV.png
[image15]: ./output_images/2RGB2HLS.png
[image16]: ./output_images/2RGB2HSV.png
[image17]: ./output_images/2RGB2LUV.png
[image18]: ./output_images/2RGB2YCrCb.png
[image19]: ./output_images/2RGB2YUV.png
[image20]: ./output_images/3RGB2HLS.png
[image21]: ./output_images/3RGB2HSV.png
[image22]: ./output_images/3RGB2LUV.png
[image23]: ./output_images/3RGB2YCrCb.png
[image24]: ./output_images/3RGB2YUV.png
[image25]: ./output_images/4RGB2HLS.png
[image26]: ./output_images/4RGB2HSV.png
[image27]: ./output_images/4RGB2LUV.png
[image28]: ./output_images/4RGB2YCrCb.png
[image29]: ./output_images/4RGB2YUV.png
[image30]: ./output_images/5RGB2HLS.png
[image31]: ./output_images/5RGB2HSV.png
[image32]: ./output_images/5RGB2LUV.png
[image33]: ./output_images/5RGB2YCrCb.png
[image34]: ./output_images/5RGB2YUV.png
[image35]: ./output_images/6RGB2HLS.png
[image36]: ./output_images/6RGB2HSV.png
[image37]: ./output_images/6RGB2LUV.png
[image38]: ./output_images/6RGB2YCrCb.png
[image39]: ./output_images/6RGB2YUV.png
[video1]: ./project_video.mp4
[video2]: ./output_images/result_output.avi

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it - you're reading it now! :)

All code refered to in this document can be found in the accompanying IPython Notebook `car-detection.ipynb`

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in my IPython notebook, under the heading `Explore the data and see what we have`

Several datasets were supplied for our use, I ran several iterations of exploring data, choosing HOG and features parameters and checking data again. Ultimately, I decided to use all data provided except for the KITTI dataset as this seemed to reduce my HOG accuracy. This could of course indicate that I have an overfitting problem. This is something I would explore had I had more time to do so.

Examples of the raw data I used for both car and non-car data:

Example Non-car:
![alt text][image1]

Example Car:
![alt text][image8]

Next up, I decided to explore images looking at changing colorspaces, and looking at each image channel seperately. The end goal in my opinion is to be able to select a small a set of features as possible for detecting cars in a video stream. If we have too many features, the algorithm will run slowly, but we want to make a decision quickly, this is to be used in a car for safety applications after all!

Below is an example whereby I plotted car and non-car images channel by channel in different colorspaces. I also computer the HOG of each channel and plotted that too. I am looking for a colorspace and channel that shows nice clean identifiable car / non-car images. If I can find that, I have reduced my HOG data and computations by a third right off the bat...

##### Car images:

###### Car 1

HLS Colorspace
![alt text][image10]
HSV Colorspace
![alt text][image11]
LUV Colorspace
![alt text][image12]
YCrCb Colorspace
![alt text][image13]
YUV Colorspace
![alt text][image14]

###### Car 2

HLS Colorspace
![alt text][image15]
HSV Colorspace
![alt text][image16]
LUV Colorspace
![alt text][image17]
YCrCb Colorspace
![alt text][image18]
YUV Colorspace
![alt text][image19]

###### Car 3

HLS Colorspace
![alt text][image20]
HSV Colorspace
![alt text][image21]
LUV Colorspace
![alt text][image22]
YCrCb Colorspace
![alt text][image23]
YUV Colorspace
![alt text][image24]

##### Non-car images:

###### Non-Car 1

HLS Colorspace
![alt text][image25]
HSV Colorspace
![alt text][image26]
LUV Colorspace
![alt text][image27]
YCrCb Colorspace
![alt text][image28]
YUV Colorspace
![alt text][image29]

###### Non-Car 2

HLS Colorspace
![alt text][image30]
HSV Colorspace
![alt text][image31]
LUV Colorspace
![alt text][image32]
YCrCb Colorspace
![alt text][image33]
YUV Colorspace
![alt text][image34]

###### Non-Car 3

HLS Colorspace
![alt text][image35]
HSV Colorspace
![alt text][image36]
LUV Colorspace
![alt text][image37]
YCrCb Colorspace
![alt text][image38]
YUV Colorspace
![alt text][image39]


#### 2. Explain how you settled on your final choice of HOG parameters.

I played with many different color spaces and HOG feature parameters. Ultimately, I found that using the L channel of the LUV colorspace for the HOG, with spatial features at (24,24) resulted in a good accuracy vs small feature set tradeoff.

I checked the accuracy by testing my classifier with a 80% / 20% train / test split of my training data.

My final parameters were set as follows:

```
_color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
_orient = 9  # HOG orientations
_pix_per_cell = 8 # HOG pixels per cell
_cell_per_block = 8 # HOG cells per block
_hog_channel = 0 # Can be 0, 1, 2, or "ALL"
_spatial_size = (24, 24) # Spatial binning dimensions
_hist_bins = 24    # Number of histogram bins
_spatial_feat = True # Spatial features on or off
_hist_feat = False # Histogram features on or off
_hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In my code, I made a Feature_manager() class that encapsulates all the HOG training and classifier fitting code. I have also made a `TrainHog` function that instatiates the Feature_manager() and can either load a pre-trained classifier and HOG, or be used to rebuild and fit a new classifier. All parameters are kept in this class so it can be used for training as well as runtime without having to remember all the parameter settings throughout the code base.

At the head of the class, I can set the parameters, whether to use color histograms, spatial features and HOG channels. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In order to maximize speed, I take the following steps in my code to perform my search, the code for this is contained in the class I made called `hog_search_window_manager()`

1. Take target frame and clip off areas that should never contain cars (the top and bottom of the image) - let's throw as many pixels away as possible
2. Next, I created three search areas(defined by my class `SearchArea()`), one where we we expect to see small cars (distant cars), one for medium sized cars, and one for large (cars up close). The larger car area covers a larger area of the screen but the small area is more centered on the approximate horizon.
3. As a frame is ready for processing, I calculate the HOG on the clipped frame (I only do this once, and for only the channels used per the Feature_manager() to save on CPU). The results are cached so we only need to do the HOG once.
4. I now walk through each `SearchArea()` breaking it into tiles (as depicted in the image below). For each tile, I scale up the tile to match the HOG size, then extract the color and spatial features if required. Note the HOG does not need to be re-computed as it was already done in step 3.
5. Now I compare each tile with the classifier to see if we have a car in the tile, logging that tile if we do.

Image showing the small, medium and large search windows:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/result_output.avi)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I realize that the pipeline implemented here could fail in a nuumber of places. I would expect issues if the road angle increases or decreases sharply moving the cars above or below the search area. Also, cars crossing infront of me (side on) would likely not be detected as most training images were back-on.

Another issue I would expect to encounter is cars that move quickly through the scene (cars changing lanes quickly, passing us quickly, or us passing them quickly). This is in part due to the filtering I have put in place to filter out false positives.

With enough given time, I would investigate addressing such problems by altering my false positive elimination scheme, and my possibly having a seperate classifier for front or side on images of cars.

Another avenue to explore that will likely hold the best hope is to investigate using Deep Learning to solve this problem. Architectures such as SSD Multi-box, YOLO and Capsule Nets are possible avenues that would likely result in better detections and faster processing times.

