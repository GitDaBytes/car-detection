## Writeup Template
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
[image3]: ./output_images/search_windows.png
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
[image40]: ./output_images/test1.png
[image41]: ./output_images/test2.png
[image42]: ./output_images/test3.png
[image43]: ./output_images/test4.png
[image44]: ./output_images/test5.png
[image45]: ./output_images/test6.png
[image46]: ./output_images/heat1.png
[image47]: ./output_images/heat2.png
[image48]: ./output_images/heat3.png
[image49]: ./output_images/heat4.png
[image50]: ./output_images/heat5.png
[image51]: ./output_images/heat6.png
[image52]: ./output_images/heat7.png
[image53]: ./output_images/heat8.png
[image54]: ./output_images/heat9.png
[image55]: ./output_images/heat10.png
[image56]: ./output_images/heat_thresh.png
[image57]: ./output_images/heat_labels.png
[image58]: ./output_images/boximage.jpg
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

Below is an example whereby I plotted car and non-car images channel by channel in different colorspaces. I also computed the HOG of each channel and plotted that too. I am looking for a colorspace and channel that shows nice clean identifiable car / non-car images. If I can find that, I have reduced my HOG data and computations by a third right off the bat...

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

1. Take target frame and clip off areas that should never contain cars (the top and bottom of the image) - lets throw as many pixels away as possible
2. Next, I created three search areas (defined by my class `SearchArea()`), one where we expect to see small cars (distant cars), one for medium sized cars, and one for large (cars up close). The larger car area covers a larger area of the screen but the small area is more centered on the approximate horizon.
3. As a frame is ready for processing, I calculate the HOG on the clipped frame (I only do this once, and for only the channels used per the Feature_manager() to save on CPU). The results are cached so we only need to do the HOG once.
4. I now walk through each `SearchArea()` breaking it into tiles (as depicted in the image below). For each tile, I scale up the tile to match the HOG size, then extract the color and spatial features if required. Note the HOG does not need to be re-computed as it was already done in step 3.
5. Now I compare each tile with the classifier to see if we have a car in the tile, logging that tile if we do.

Image showing the small, medium and large search windows:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As above, I settled on LUV color space, using the L channel for the HOG, and used spatial binning with 24,24 dimensions, at three different scales.  Here are some example images of detections (some correct and some mismatches too):



![alt text][image40]
![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]
![alt text][image45]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/result_output.avi)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to filter out as many false postives as possible, I played with several possible options. Initially I looked at using a heatmap technique that assign a higher probability of a detection based on the number of search tiles that overlap on a given area. While this works if you group together tiles very tightly, I wanted fast processing, and therefore wanted to limit the number of tiles I tested at runtime.

What I came up with was a class called `heatmap_filter()`. This is how it works:

After processing each frame of the video, I have a list of bounding boxes of suspected vehicle locations in my image. I then create a mask frame, and in that mask frame, I set each pixel value to one that falls within any bounding box I listed for that frame. I then take that mask and put it into a dequeue structure to act as a FIFO buffer. I set my buffer to keep a history of the last 10 mask frames from the video.

Now, I create a second mask frame (my aggregate), and add up each pixel value from those in my history queue. Finally, all the pixels in my aggregate are set to zero if the pixel value is <= 6 (threshold), or 1 otherwise.

At this point I have an aggregate view of stable tracks over a period of time, it allows for short dropout of stable tracks, and only lets through strong signals. I apply `scipy.ndimage.measurements.label` to this aggregate to identify clumps of hot pixels, where each detected 'clump' is taken to be a vehicle. I then draw my final bounding boxes around each label, this is what is shown on my final video.

### Here are the heatmap masks from the final 10 frames of the video. White boxes indicate possible detections:

![alt text][image46]
![alt text][image47]
![alt text][image48]
![alt text][image49]
![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]

### Here is the output of the ten frames after threshold:
![alt text][image56]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all ten frames:
![alt text][image57]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image58]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I realize that the pipeline implemented here could fail in a nuumber of places. I would expect issues if the road angle increases or decreases sharply moving the cars above or below the search area. Also, cars crossing infront of me (side on) would likely not be detected as most training images were back-on.

Another issue I would expect to encounter is cars that move quickly through the scene (cars changing lanes quickly, passing us quickly, or us passing them quickly). This is in part due to the filtering I have put in place to filter out false positives.

With enough given time, I would investigate addressing such problems by altering my false positive elimination scheme, and my possibly having a seperate classifier for front or side on images of cars.

Another avenue to explore that will likely hold the best hope is to investigate using Deep Learning to solve this problem. Architectures such as SSD Multi-box, YOLO and Capsule Nets are possible avenues that would likely result in better detections and faster processing times.

