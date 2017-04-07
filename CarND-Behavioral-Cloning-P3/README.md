[//]: # (Image References)

[image1]: ./examples/center_lane_driving.png "Center lane driving - Track 1"
[image2]: ./examples/center_lane_driving2.png "Center lane driving - Track 2"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


[image8]: ./examples/generated_figures/figure1.png "Non augmented/steering angles values"
[image9]: ./examples/generated_figures/figure2.png "Non augmented/filtered speed values"
[image10]: ./examples/generated_figures/figure3.png "Histogram of original dataset steering angles values"
[image11]: ./examples/generated_figures/figure4.png "Histogram of original dataset speed values"
[image12]: ./examples/generated_figures/figure5.png "Filtered angles values"
[image13]: ./examples/generated_figures/figure6.png "Filtered speed values"
[image14]: ./examples/generated_figures/figure7.png "Histogram of filtered steering angles values"
[image15]: ./examples/generated_figures/figure8.png "Histogram of filtered speed values"
[image16]: ./examples/generated_figures/figure9.png "Preprocessed and resized, left, center and right camera view"
[image17]: ./examples/generated_figures/figure10.png "Augmented and flipped images generated from 1 time stamp"
[image18]: ./examples/generated_figures/figure11.png "Histogram of filtered/augmented/flipped steering angles"
[image19]: ./examples/generated_figures/model_loss_vs_epoch.png "Model mean squared error vs epoch"
[image20]: ./examples/ModelArchitecture.png "Model architecture"


##Use Deep Learning to Clone Driving Behavior
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I have implemented a driving behavior cloning network using convolutional neural networks. 

The Project
---
The goals / steps of this project are the following:
1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report

## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around both tracks at 20mph by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network (lines 477-502). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's [end to end learning article](https://arxiv.org/pdf/1604.07316.pdf). The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. (model.py lines 403-424). 

According to the [article](https://arxiv.org/pdf/1604.07316.pdf) (p.4): 
> The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations.

Further down they explain:
> [They] follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering, but [they] note that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.

I decided to include ELU layers to introduce nonlinearity (model.py lines 403-422). [This  article](http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/) points out the advantages of ELUs over RELUs. This activation function among other things can make learning faster.

The input data consists of the RGB channels of the image. NVIDIA's article uses YUV channels but RGB were more convenient since it prevented me from modifying drive.py, but I ended up modifying it anyway for other reasons stated below. The input image is then normalized and cropped in the model using a Keras lambda layer (model.py line 396) and a Keras Cropping2D layer (model.py line 399).

In order to reduce training time, I've also been able to resize the input image to 50% of its original dimensions without noticeable accuracy loss. To accommodate the new input size of the model, I had to modify drive.py (lines 58-65 and 80-81) to apply a cv2.resize on the images yeilded by the simulator.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 412-416-420). The dropout function in Keras takes in the drop probability as opposed to the one in tensorflow which take in the keep probability.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 380 and 475). The test portion was executed by running the model with the simulator.

The validation loss was monitored at then end of each epoch and the weights that yeilded the lowest validation loss were saved along the way. Those weights can then be restored and embeded into a model usable to drive the simulator. 

The fact that the vehicle can stay on both tracks is sort of an indicator of the model's capability to generalize.

#### 3. Model parameter tuning

The model used an Adam optimizer so I kept the base learning rate (1e-3) (model.py lines 20 and 486). A learning rate decay of 1e-7 was applied to get a better convergence.

https://arxiv.org/pdf/1412.6980.pdf

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Center lane driving was recorded on both track in the normal direction and in the reverse direction. Recovery manoeuvers were recorded along the totality of both tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look for papers that contained architecture examples.

My first step was to use a convolution neural network model similar to the in NVIDIA's [end to end learning paper](https://arxiv.org/pdf/1604.07316.pdf) and comma.ai's [Learning a Driving Simulator paper](https://arxiv.org/pdf/1608.01230.pdf). I thought this model might be appropriate because both companies are running real-life succesfull autonomous cars. Plus those papers are solving exactly the same problem as I am trying to.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers after each of the fully connected layers. Then I tuned the number of epochs and the dropout probability to get the lowest validation loss possible. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 
To improve the driving behavior in these cases, I did several things:
* I recorded more training examples and more recovery manoeuvers.
* I flipped the images and the steering angle to remove any bias in the steering angle data (model.py lines 88-89)
* I used all three cameras images by adding an offset (0.25 or 6.25&deg;) to the steering angle for the right and left cameras (model.py lines 143-145).
* I generated synthetic images by doing a brigthness randomisation and a vertical and horizontal perspective shift (model.py lines 81-85).
* I also filtered the data based on vehicle speed to remove steering angles suceptible to be too sharp.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 396-424) consisted of a convolution neural network with the following layers and layer sizes:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 80, 160, 3)        0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 43, 160, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 21, 79, 24)        672       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 39, 36)        7812      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 19, 48)         15600     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 18, 64)         12352     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 17, 64)         16448     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2176)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2176)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               217700    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 276,155.0
Trainable params: 276,155.0
Non-trainable params: 0.0
_________________________________________________________________
```
Here is a visualization of the architecture:

![alt text][image20]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first concentrated on center lane driving. I first recorded 1 lap on track one going in the normal direction then I recorded another lap going in the opposite direction. I did the same on track 2.  I used the mouse to control the steering angle and the keyboard for the throttle. I did not pay to much attention at keeping the speed constant throughout the laps.

Here's an two examples of center lane driving:

![alt text][image1]
![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center of the lane. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

During training, I noticed that I was suceptible to choose sharper recovery angles if I was driving at lower speed. I believe this is a natural driving reflex based on reaction and actuating time. Since I don't want my model to start oscillating from one edge of the road the other, I decided to remove the images and steering angles that have been recorded at speeds lower than 10mph (model.py line 342 ). 


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

![alt text][image10]
I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


# inclure un lien vers le simulateur


# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).



