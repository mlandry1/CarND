[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/tr.jpeg "Cover image"
[image10]: ./examples/histogram_before.png "Histogram of the training dataset"
[image11]: ./examples/Training_dataset_sample_before.png "Sample of the training dataset"
[image12]: ./examples/histogram_after.png "Histogram of the training dataset after augmentation"
[image13]: ./examples/AugmentedDataset.png "Sample of the dataset after augmentation"
[image14]: ./examples/0_preprocess_original.png "Original sample"
[image15]: ./examples/1_preprocess_equalize_histogram.png "Sample after Y histogram equalization"
[image16]: ./examples/2_preprocess_grayscale.png "Sample after the grayscale conversion"
[image17]: ./examples/3_preprocess_grayscale.png "Image values after the grayscale conversion"
[image18]: ./examples/4_preprocess_normalize.png "Image values after the normalization"
[image19]: ./examples/ConvNet.png "Pierre Sermanet's and Yann LeCun's Conv Net Architecture"
[image20]: ./examples/architecture.png "My Conv Net Architecture"
[image21]: ./examples/mean_variance.png "Input data mean and variance"


## Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![alt text][image9]
















Overview
---
In this project, I have implemented a classifier using convolutional neural networks. 


The Project
---
The goals / steps of this project are the following:
1. Load the data set
2. Explore, summarize and visualize the data set
3. Augment and preprocess the dataset
4. Design, train and test a model architecture
5. Use the model to make predictions on new images
6. Analyze the softmax probabilities of the new images
7. Summarize the results with a written report

Test results
---
**Validation accuracy** : 97.9%
**Test accuracy** : 95.3%

## Rubric Points
See the [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)  for this project.

Here is a link to my [project code](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Summary of the data set

For this project, we were asked to use the [German Trafic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) (GTSRB) data. Download the formated dataset [here...](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
I used the numpy library to calculate summary statistics of the traffic signs data set. The code for this step is contained in the second code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb) :

* The size of training set is 34799 images.
* The size of validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

####2. Exploratory visualization of the dataset

The code for this step is contained in the third code cell of the  [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb) :


Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed accross the different classes.

![alt text][image10]

And here is a sample of the training dataset.
![alt text][image11]

###Design and Test a Model Architecture

#### Training data augmentation

The code for this step is contained in the 4th and 5th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb) :

Considering the great data inbalance seen in the above histogram, I've decided to "augment" my training set. Augmenting the training set helped me to improve the model performance. 

In order to do my data augmentation, I've decided to follow the recommendations found in this suggested [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) written by Pierre Sermanet and Yann LeCun. First, I needed to code a function that randomly transforms dataset images. Common data augmentation techniques include rotation, translation, zoom, flips, and/or color perturbation. These techniques can be used individually or combined. I've decided to go with position ([-2,2] pixels), zoom ([.9,1.1] ratio) and rotation ([-15,+15] degrees), following Lecun's example. Plus I decided to add a brightness variation. 
* I used *cv2.addWeighted()* for brightness variation, by adding (with random weight) a black or white image to the orignal image. I used this to overcome the overflow problem I was facing with other techniques.
* I used *cv2.resize* for scale variation. I also used *cv2.copyMakeBorder* for image padding (with zeros) to keep a 32x32x3 image.
* I used *cv2.getRotationMatrix2D" and *cv2.warpAffine* for my rotations.
* Finaly, I used *cv2.warpAffine* with a manualy entered transformation matrix for my translations.

Then I used this function in a loop to fill out the classes that needed more examples up to the number of examples of the most represented class.

Here is a bar chart showing how the training data is now distributed accross the different classes.

![alt text][image12]

And here is a sample of the augmented training dataset:

![alt text][image13]

#### Training data preprocessing
The code for this step is contained in the 6th, 7th and 8th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb)

In order to do my data preprocessing, I inspired myself from the work presented in the suggested [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here is the original image we will be studying as an example:
![alt text][image14]

As a first step, I would like to improve the contrast on the original image.
The first steps I'm going trought are:
* Convert the image to the YUV space using *cv2.cvtColor*
* Equalize the Y channel histogram using *cv2.equalizeHist*.  I leave the U and V channels intacts.
* Convert back the image to RGB using *cv2.cvtColor* once again.
The process yields the following contrast improved image:
![alt text][image15]
In the article it is suggested that converting the images to grayscale further improve the network classification accuracy, so I decided to go for it as well:
![alt text][image16]

As a last step, I normalized the image data in order to keep numerical stability. Indeed, we have to keep the values involved in the calculation of the loss function in the same range (never too big or too small). As a guiding principle, Vincent Vanhoucke says (In section 23 of lesson 6) that we should have input data with zero and equal variance whenever its possible. 
![alt text][image21]
On top of numerical stability, a well conditionned problem makes it a lot easier for the optimizer to do its job. In this lesson, Vincent also suggest that it is possible to normalize image data to get a zero mean. However in the TensorFlow introduction Lab at the end of lesson 6 we learned to do a min-max normalization on a grayscale image which yeilds a 0.5 mean value. I chose this last method since I had hands-on expericence with it and it preserves the ability to accurately show the image within a *matplotlib* figure.
Here are the values of the previous grayscale image before normalization:
![alt text][image17]
And here they are after normalization:
![alt text][image18]



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:


| Layer         |     Description	        					|  Input            |  Output                     |
|:-------------:|:---------------------------------------------:|:-----------------:|:---------------------------:| 
| Input         | 32x32x1 Grayscale image   					| Image             | Convolution 1               |
| Convolution 1 | 1x1 stride, valid padding, outputs 28x28x100 	| Input             | RELU                        |
| RELU 1		|												| Convolution 1     | Max Pooling 1               |
| Max pooling 1	| 2x2 stride,  outputs 14x14x100 				| RELU 1            | Convolution 2, Max Pooling 3|
| Convolution 2 | 1x1 stride, valid padding, outputs 10x10x200	| Max pooling 1     | RELU 2                      |
| RELU 2		|												| Convolution 2     | Max pooling 2               |
| Max pooling 2	| 2x2 stride,  outputs 5x5x200  				| RELU 2	        | Flatten 2                   |
| Max pooling 3	| 2x2 stride,  outputs 7x7x100   				| Max pooling 1     | Flatten 1                   |
| Flatten 1		| Input = 7x7x100, Output = 4900                | Max pooling 3     | Concatenate 1               |
| Flatten 2		| Input = 5x5x200, Output = 5000                | Max pooling 2     | Concatenate 1               |
| Concatenate 1 | Input1 = 4900, Input1 = 5000, Output = 9900   | Max pooling 2 and 3 |Fully connected            |
| Fully connected | Fully Connected. Input = 9900, Output = 100 | Concatenate 1     | Softmax                     |
| Softmax		| Fully Connected. Input = 100, Output = 43     | Fully connected   | Probabilities               |

![alt text][image19]
![alt text][image20]

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 