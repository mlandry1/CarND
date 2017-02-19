## Finding Lane Lines on the Road


The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/0.whiteCarLaneSwitch.jpg "Original image"
[image2]: ./examples/1.processed-whiteCarLaneSwitch.png "Grayscale"
[image3]: ./examples/2.processed-whiteCarLaneSwitch.png "Gaussian Blur"
[image4]: ./examples/3.processed-whiteCarLaneSwitch.png "Canny Edge"
[image5]: ./examples/4.processed-whiteCarLaneSwitch.png "Region of interest"
[image6]: ./examples/5.processed-whiteCarLaneSwitch.png "Hough Transform"
[image7]: ./examples/6.processed-whiteCarLaneSwitch.png "Slope Filter"
[image8]: ./examples/7.processed-whiteCarLaneSwitch.png "Final result"
[image9]: ./examples/hilly_road.jpg "Hilly road"
---

## Reflection

####1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

#### Pipeline:

My pipeline consisted of 7 steps. 

- Step 0 : Original image
![alt text][image1]
- Step 1 : Convert the images to grayscale
![alt text][image2]
- Step 2 : Apply a Gaussian Blur
![alt text][image3]
- Step 3 : Run a Canny Edge Detection
![alt text][image4]
- Step 4 : Apply a mask to restrict the region of interest (ROI)
![alt text][image5]
- Step 5 : Detect lines throught a Hough Transform
![alt text][image6]
- Step 6 : Filter out the lines according to their slopes
![alt text][image7]
- Step 7 : Draw two single overlaying lines over the original image from the mean values of the remaining lines
![alt text][image8]

#### draw_lines() function:

In order to draw a single line on the left and right lanes, I modified the *draw_lines()* function.


    def draw_lines(img, lines, roi_top, roi_bottom, min_slope, max_slope, color=[255, 0, 0], thickness=2):

        #Initialize variables
        sum_fit_left = 0
        sum_fit_right = 0
        number_fit_left = 0
        number_fit_right = 0

        for line in lines:
            for x1,y1,x2,y2 in line:
                #find the slope and offset of each line found (y=mx+b)
                fit = np.polyfit((x1, x2), (y1, y2), 1)

                #limit the slope to plausible left lane values and compute the mean slope/offset
                if fit[0] >= min_slope and fit[0] <= max_slope:
                    sum_fit_left = fit + sum_fit_left
                    number_fit_left = number_fit_left + 1

                #limit the slope to plausible right lane values and compute the mean slope/offset
                if fit[0] >= -max_slope and fit[0] <= -min_slope:
                    sum_fit_right = fit + sum_fit_right
                    number_fit_right = number_fit_right + 1

        #avoid division by 0
        if number_fit_left > 0:
            #Compute the mean of all fitted lines
            mean_left_fit = sum_fit_left/number_fit_left
            #Given two y points (bottom of image and top of region of interest), compute the x coordinates
            x_top_left    = int((roi_top - mean_left_fit[1])/mean_left_fit[0])
            x_bottom_left = int((roi_bottom - mean_left_fit[1])/mean_left_fit[0])
            #Draw the line
            cv2.line(img, (x_bottom_left,roi_bottom), (x_top_left,roi_top), [255, 0, 0], 5)
        else:
            mean_left_fit = (0,0)

        if number_fit_right > 0:
            #Compute the mean of all fitted lines
            mean_right_fit = sum_fit_right/number_fit_right
            #Given two y points (bottom of image and top of region of interest), compute the x coordinates
            x_top_right    = int((roi_top - mean_right_fit[1])/mean_right_fit[0])
            x_bottom_right = int((roi_bottom - mean_right_fit[1])/mean_right_fit[0])
            #Draw the line
            cv2.line(img, (x_bottom_right,roi_bottom), (x_top_right,roi_top), [255, 0, 0], 5)
        else:
            fit_right_mean = (0,0)


-First it does a ***np.polyfit()*** on each line returned by the ***cv2.HoughLinesP()*** function.
-Then it filters out the lines with "not so plausible" slopes.
-It then computes the arrithmetic mean of the "m" and the "b" parameter (*y=mx+b*) for the left lane line and the right lane line.
-Since, the top and the botom of the region of interest is passed by the function parameters, it is then a matter of isolating x in *y=mx+b* to find the x coordinate of the lines extremities.
-It finaly calls the ***cv2.line()*** function to draw the lines the image passed in the function parameters.

###2. Identify potential shortcomings with your current pipeline

In my opinion, the major shortcomming of this pipeline would be that it assumes a praticaly straight road since it is designed to reject lines with too shallow slopes, it will therefore reject the lane lines when the road is too curvy. It will also be problematic on hilly roads where the algoritm can't "see" the rest of the road ahead because the camera is pointing towards the sky. This is without mentionning the case where a hilly road would result on multiple disconnected line detections (see the picture below). Debris on the road are also at risk of be accepted as line by the algorithm... Finaly, I believe that a bumpy road can shake the car enough to offset the lanes outside of the regions of interest mask temporarily.
![alt text][image9]

Another shortcoming could be that it seemed influenced by the lighting/color of the road, especially when I tried the *challenge.mp4* video. In that video, you can see the car going by a tree that projects a shadow on the road, then the road suface is  transitionning from asphalt to concrete and comes back to concrete exactly at the same time as the car passes by another tree. This causes the lane detection algorithm to go haywire, confusing all those contrasts to be lanes.

###3. Suggest possible improvements to your pipeline

A possible improvement would be to be able to adapt the region of interess dynamicaly, following inputs like the steering angle, pitch/yaw/roll rate, x,y,z accel...

Another potential improvement could be to be able to flater out the image as much as possible. The idea would be to feed a perfectly ligthed scene to our lane detection pipeline every time.