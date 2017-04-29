## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product to create is a detailed writeup of the project. 

The Project
---

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
6. Warp the detected lane boundaries back onto the original image.
7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation. 
 
 ### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


The code for this step is contained in the first code cell of the following IPython notebook: *"./P4-Camera Calib.ipynb"*.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The following is the result of the corners detection:

<img src=./output_images/corner_detection.png  width="1200">

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (second cell of the same notebook).  I then save the camera calibration in a pickle file for later use. Finaly, I applied the distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

<img src=./output_images/test_undist.png  width="1200">

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for the rest of this writeup is all contained inside the following IPython notebook: *"P4-Advanced Lane Finding.ipynb"* and will be refered to as : "the notebook".

In the 2nd code cell of the notebook, I first load the "pickled" camera calibration. In the 3rd code cell, I load a set of images representing a straight road and finaly in the 4th cell I applied the distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
<img src=./output_images/undistorded_image_straight.png width="1200">

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in the 5th code cell of the IPython notebook. I chose the hardcode the following source and destination points:

| Source         | Destination | 
|:---------------:|:----------------:| 
| 560, 475      | 440, 0          | 
| 727, 475      | 840, 0          |
| 1115, 720    | 840, 720      |
| 216, 720      | 440, 720      |

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src=./output_images/perspective_xform_straight_images.png width="1200">





































































































#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The following figure describes the processing pipeline implemented to derive a thresholded binary image from the orginal RBG image. The whole pipeline is implemented in a single function called `bin_image_pipeline()` (see the 20th cell of the notebook).

<img src=./output_images/ThresholdingPipeline.png width="1200">






















































































The steps are divided as follows:
1. Undistord the image using the saved ("pickled") camera calibration: to get parallel lane lines.
2. Perspective shift: in order to give us a bird's eye view of the road.
3. Color thresholding: to identify white and yellow pixels. The thresholding are applied both in YUV and HLS color spaces. The HLS space was my first choice since it segments the colors values into a color base (Hue), a brightness value (Luminance) and an amount of color (Saturation). For example, it is easy to identify the Hue range of yellowish colors and then choose a threshold of saturation and brigthness for the yellow line color mask. Same goes for the white mask where the totality of the Hue range and saturation range is selected. My decision to go with a YUV conversion for color thresholding was made following a post seen on the Slack *#p-advanced-lane-lines* channel. The post suggested to substract the V channel from the U channel and then apply a threshold on the resulting channel to detect yellow lanes. Since I saw a slight performance improvement I decided to keep this yellow line detector in my pipeline alongside with my original solution.
4. Sobel filtering: to identify edges. I first apply Contrast Limited Adaptive Histogram Equalization on the luminance channel of the HLS in order to improve contrast. Since at this point the image is already perspective transformed, the lane lines are thus mostly vertical. Therefore I used only X direction sobel filters. To get as much information as possible, the filters were applied on L and S channel before being combined together. 
5. Region of interest (ROI) masking: to eliminate regions that can't contain lane lines. A ROI mask is finaly applied on the binary image composed by a bitwise AND of the color thresholded image and the sobel filtered image. an example of the final output is pictured below:

<img src=./output_images/binary_image.png width="400">

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After much experimentation with my own algorithms, I decided to stick with the algorithms suggested in the project lessons. On the first encountered frame, I run a "Slidding window search" and then the algorithm is switched to "Lane tracking" for the subsequent frames. If the sanity checks in the tracking algortihm rejects more than 50 frames in a row (1.66s of video), I fallback to the "slidding window search" algortihm and then start tracking again. 

##### Slidding window search

The code for these steps is located in the 22nd code cell of the notebook.

The first step is to sum the number of pixel along each column of the lower half of the binary image shown before. The result can be plot as an histogram, as shown in the next figure.

<img src=./output_images/binary_image_half_histo.png width="400">

As stated in the project lesson (concept #33):
>In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

The search algorithm can be explained a bit like this:
* The first windows centers are located at the histogram peaks. 
* If a sufficient number of "hot pixels" are detected, the following set of windows will be centered on the mean posiiton of these pixels, otherwise the same position is carried on for the following window. 
* When the windows reach the top of the image, I do a 2nd order polynomial fit on the pixel enclosed by the windows.

<img src=./output_images/slidingwindow.gif width="400">

##### Lane tracking

The code for these steps is located in the 27th code cell of the notebook.

Now that I have fitted a polynomial on the lane lines, I can use this prior information to do a more targeted search on the subsenquent frames.

The lane tracking algorithm can be explained a bit like this:
* First, a margin arround the previous fit is computed (see the green zone in the following figure).
* Then, pixels inside this zone are identified. If no pixels are found in either zones, the last fit is kept and I start incrementing a skiped frame counter.

<img src=./output_images/slid_window_tracking.png width="400">

* If line pixels were detected, a new 2nd order polynomial fit is computed on those pixels.
* The new lane width is then computed, if this width is above or below a threshold, the new fit values are droped and the old ones are carried on. The skip counter is also increased.
* If the new lane width is within limits, the difference between the new fit coefficients and the old ones are computed. If one of those differences is greater than its asscoiated threshold, the new fit values are droped and the old ones are carried on. The skip counter is also increased.
 * If everything is within the limits, I blend the curvature coefficents of both line lane based on the assumption that both lines should be parallel. When this step is reached, the skip counter is reseted.
 * If the skip counter reaches a value higher than 50, then the Windows Search Algorithm is called to output new fit values. 
 * After all these steps, I have either a new fit value computed by the lane tracking algorithm or a carried-on fit or a fit value from the slidding windows search. 
 * A low pass filter is then applied, which blends those fit values with the old ones. 


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for these steps is located in the 24th, 25th and 26th code cell of the notebook.

##### Real world units
Th first step is to convert my polynominal fit coeffcients from the pixel space to the meters space. This is computed by the `get_poly_m`function (see the 24th code cell). 
* To do so, I had to define conversion constants (`xm_per_pix` and `ym_per_pix`) based on the lane width (3.65m) and the dashed lane lines length (3m). 
* Then I used the simple method i.e.: plot both lines in pixel space, do the conversion on those pixels and then refit a polynomial on those scaled pixels.

##### Radius of curvature

The 2nd order polynomial equation we are fitting is described by the following:
<img src=./output_images/eq1.png height="35">

The radius of curvature at any point x of the function x=f(y) is given as follows:
<img src=./output_images/eq2.png height="80">

In our case the derivatives are:
<img src=./output_images/eq3.png height="100">

Hence the following curvature radius equation :
<img src=./output_images/eq4.png height="55">

The radius of curvature is computed by the `get_lane_curvature()` function (see the 25th code cell). The radius of curvature is computed nearest to the car which corresponds to a y coordinate of 719 (bottom of a 720x1280 image).

##### Vehicle position with respect to the center

The vehicle position with respect to the center and the lane width are computed in the `distance_from_center()` function (see the 26th code cell). 
* Using the real world unit polynomial, I compute the line positions at the bottom of the image (`y coordinate = 719 * ym_per_pix`).
* Then the right line position (`rightx`) is substracted from the left line position (`leftx`) to get the lane width. 
* The center of the image (width=1280pix) is converted in real world coordinate : `xcenter = 639*xm_per_pix`. 
* Then the distance from the center is computed by the following : `dist_off_center = xcenter - (leftx + rightx)/2`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the 29th code cell in the function named `draw_final_overlay()`.  Here is an example of my result on a test image:

<img src=./output_images/final_overlay_test3.png width="800">

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to [download](./project_video_output.mp4) my project video result.

Click on images below to see the full-length YouTube videos..

##### Project Video Output:
[![Project Video Output](./output_images/project_output_video.gif)](https://youtu.be/hjzrhPXpDg4)
##### Challenge Video Output:
[![Challenge Video Output](./output_images/challenge_output_video.gif)](https://youtu.be/1Wza0GCQSns)

---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Outlier rejection was a big issue. I found that the color mask were the most efficient way to discriminate the lines. The Sobel filters had a tendancy of picking up the wrong stuff. Especialy in the challenge video, in which the bottom intersection of the concrete barrier with the asphalt on the left of the lane was considered to be a strong line candidate by the Sobel filter. 

Nonetheless, the color masks weren't flawless either, I found that they were much more sensitive to brightness variation.  The segment, when the car passes under the overpass in the challenge video was especialy problematic for the color masks.

Maybe I should have applied my thresholding functions on the original images instead of on the perspective transformed images. I think the bilinear interpolation applied on the later had effect on the discrimitation performance on the farther away lane lines.

I think the pipeline could be a bit more robust with more sanity checks and if they were applied independantly on each lane line in order to preserve as much information as possible. In my current implementation, I drop both fited polynomial as soon as a sanity check isn't successful. However, I think the approach is limited and works only under certain assumptions. 

The following is a non exhaustive list of the conditions I think that won't be properly handled by my pipeline:
* A car overtaking too close to you
* Sudden, lightining condition change (tunnel)
* Steep climbs
* Sharp curves
* Old markings on the road
* Construction zones
* Dirt roads..
* Etc..




