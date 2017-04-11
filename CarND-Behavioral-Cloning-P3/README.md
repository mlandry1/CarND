## Use Deep Learning to Clone Driving Behavior
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
* *model.py* containing the script to create and train the model
* *drive.py* for driving the car in autonomous mode
* *model.h5* containing a trained convolution neural network 
* *README.md* summarizing the results
* *video.mp4* showing 1 lap around track 1 @30mph and 1 lap around track 2 @ 20mph.
* *video.py* python script used to create video.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my *drive.py* file, the car can be driven autonomously around both tracks at 20mph by executing 
```sh
python drive.py model.h5
```
It is also possible to drive arround track 1 @30mph, if you modify *drive.py* accordingly.

#### 3. Submission code is usable and readable

The *model.py* file contains the code for training and saving the convolution neural network (lines 477-502). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's [end to end learning article](https://arxiv.org/pdf/1604.07316.pdf). The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. (*model.py* lines 403-424). 

According to the [article](https://arxiv.org/pdf/1604.07316.pdf) (p.4): 
> The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations.

Further down they explain:
> [They] follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering, but [they] note that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.

I decided to include ELU layers to introduce nonlinearity (*model.py* lines 403-422). [This  article](http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/) points out the advantages of ELUs over RELUs. This activation function among other things can make learning faster.

The input data consists of the RGB channels of the image. NVIDIA's article uses YUV channels but RGB were more convenient since it prevented me from modifying *drive.py*, but I ended up modifying it anyway for other reasons stated below. The input image is then normalized and cropped in the model using a Keras lambda layer (*model.py* line 396) and a Keras Cropping2D layer (*model.py* line 399).

In order to reduce training time, I've also been able to resize the input image to 50% of its original dimensions without noticeable accuracy loss. To accommodate the new input size of the model, I had to modify *drive.py* (lines 56-63 and 79) to apply a *cv2.resize* on the images yeilded by the simulator.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (*model.py* lines 412-416-420). The dropout function in Keras takes in the drop probability as opposed to the one in tensorflow which take in the keep probability.

The model was trained and validated on different data sets to ensure that the model was not overfitting (*model.py* line 380 and 474-475). The test portion was executed by running the model with the simulator.

The validation loss was monitored at then end of each epoch and the weights that yeilded the lowest validation loss were saved along the way. Those weights can then be restored and embeded into a model usable to drive the simulator. 

The fact that the vehicle can stay on both tracks is sort of an indicator of the model's capability to generalize.

#### 3. Model parameter tuning

The model used an [Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf) so I kept the base learning rate (1e-3) (*model.py* lines 20 and 488). A learning rate decay of 1e-7 was applied to get a better convergence.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Center lane driving was recorded on both track in the normal direction and in the reverse direction. Recovery manoeuvers were recorded along the totality of both tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look for papers that contained architecture examples applied on a similar problem.

My first step was to use a convolution neural network model similar to the in NVIDIA's [end to end learning paper](https://arxiv.org/pdf/1604.07316.pdf) and comma.ai's [Learning a Driving Simulator paper](https://arxiv.org/pdf/1608.01230.pdf). I thought this model might be appropriate because both companies are running real-life succesfull autonomous cars. Plus those papers are solving exactly the same problem as I am trying to.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers after each of the fully connected layers. Then I tuned the number of epochs and the dropout probability to get the lowest validation loss possible. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 
To improve the driving behavior in these cases, I did several things:
* I recorded more training examples and more recovery manoeuvers.
* I flipped the images and the steering angle to remove any bias in the steering angle data (*model.py* lines 88-89)
* I used all three cameras images by adding an offset (0.25 or 6.25&deg;) to the steering angle for the right and left cameras (*model.py* lines 143-145).
* I generated synthetic images by doing a brigthness randomisation and a vertical and horizontal perspective shift (*model.py* lines 81-85).
* I also filtered the data based on vehicle speed to remove steering angles suceptible to be too sharp (*model.py* lines 119-130).

#### 2. Final Model Architecture

The final model architecture (*model.py* lines 396-424) consisted of a convolution neural network with the following layers and layer sizes:

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

<img src=./examples/ModelArchitecture.png width="1200">

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first concentrated on center lane driving. I first recorded 1 lap on track 1 going in the normal direction then I recorded another lap going in the opposite direction. I repeated the same process on track 2 in order to get more data points.  I used the mouse to control the steering angle and the keyboard for the throttle. I did not pay to much attention at keeping the speed constant throughout the laps.

Here's an two examples of center lane driving:

##### Recovery maneuver on track 1:
<img src=./examples/center_lane_driving.png  width="1200">

##### Recovery maneuver on track 2:
<img src=./examples/center_lane_driving2.png  width="1200">

I then recorded the vehicle recovering from the left and right sides of the road back to the center in order to teach the network to recover if ever the car was too far off on one side of the road. 

These images show what a recovery looks like :

##### Track 1:
|Step| Image|
|:----:|:---:|
| 1 |<img src=./examples/recoverytrack1step1.png  width="1200">|
| 2 |<img src=./examples/recoverytrack1step2.png  width="1200">|
| 3 |<img src=./examples/recoverytrack1step3.png  width="1200">|

##### Track 2:
|Step| Image|
|:----:|:---:|
| 1 |<img src=./examples/recoverytrack2step1.png  width="1200">|
| 2 |<img src=./examples/recoverytrack2step2.png  width="1200">|
| 3 |<img src=./examples/recoverytrack2step3.png  width="1200">|

During training, I noticed that I was suceptible to choose sharper recovery angles if I was driving at lower speed. I believe this is a natural driving reflex based on reaction and actuating time. Since I didn't want my model to start oscillating from one edge of the road to the other, I decided to remove the images and steering angles that have been recorded at speeds below 10mph (*model.py* line 119-130 and 342). 

To augment the data set, I also flipped images and angles thinking that this would cancel the steering angle occurence bias as we can see in the following images:
<img src=./examples/histo_before_flip.png  width="1000">
<img src=./examples/histo_after_flip.png  width="1000">

For example, here is an example of an image set that has been flipped:
<img src=./examples/flipped_example.png  width="1000">

To further augment the dataset and help the model generalize, I decided to implement a jittering function : *data_augmentation* (*model.py* lines 37-51). This function consists of a randomized brightness adjustment, a randomized horizon shift and a randomized lateral perspective shift. The horizon shift is meant to help the model to generalize in hilly conditions and thus perform better on track #2. The lateral shift, for its part, is meant to provide a greater variety of steering angle/image examples in order to help the network generalize. Points were chosen on the image (see figure below), and are then shifted to give a whole new perspective (*model.py* lines 74-83). A transfert function was then determined empiricaly to augment the steering angles as well (*model.py* line 104).

Here are the points chosen for the perspective shift:

<img src=./examples/points_perspective_shift.jpg  width="320">

Here are some example before and after augmentation:

<img src=./examples/augmented_example.png  width="1000">

These random transforms are applied on images from all three dashboard cameras to generate 3 new images. Then 3 other images are generated by flipping the former augmented images in order to keep the mean value of the generated to 0.
 
After the whole process, I had 65904 training sample and 28248 validation samples. 

I then preprocessed this data using these steps:
* convert the color map from BGR to RGB (*model.py* line 26)
* resize the image to the model's expected input image size (*model.py* line 29)
* apply a slight gaussian blur to smooth out artifacts (*model.py* line 32)

The color map conversion is required since the cv2.imread() function used to load the data for training outputs BGR images but the simulator for its part outputs RGB images. 

The whole process of augmenting, flipping and preprocessing both the training and the validation dataset is done using a Keras generator. This allows me to run this process on mini-batches with the CPU while the GPU is backpropagating the model without having to preload the entire dataset into the memory. 

The final step is to randomly shuffle the mini-batches data before yielding it to the *fit_generator* function (*model.py* line 466).

The validation set and training set seperation is done before the data enters the Keras generator. Indeed, I initially randomly shuffle the data set and put 30% of the data into a validation set (*model.py* line 380)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact that after the third epoch the validation loss started to go up again (see the graphic below). My final model had a validation loss of 0.0084 and a training loss of 0.0055.  I used an adam optimizer so that manually choising the learning rate wasn't necessary. I also used a learning rate decay of 1e-7 to try to get a better final solution (*model.py* lines 488 and 21).

<img src=./examples/model_loss_vs_epoch.png  width="1000">

To make sure I wasn't going to end up with an over or underfitted model , I used Keras's ModelCheckpoint callback (*model.py* line 492). This callback is executed after each epoch. It was configured to save the weights only when the current epoch validation loss is the lowest up to this point in the training process.

## Results

My final model had a validation loss of 0.0084 and a training loss of 0.0055. The car drives well on both tracks, a bit of oscillation can be seen when the car is crossing the bridge on track 1. I believe this related to a dataset quality issue. Maybe I should have generated more training data for this bridge or double check the existing data...

Click on images below to see the full-length YouTube videos..

##### Track 1:
[![30mph driving arround Track 1](./examples/giphy-track1.gif)](https://youtu.be/Ul6jH5K0HVA)
##### Track 2:
[![20mph driving arround Track 2](./examples/giphy-track2.gif)](https://youtu.be/pdipLkbhiq4)
