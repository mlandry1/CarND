[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/bumpyroad.jpg "Bumpy Road Traffic Sign"
[image5]: ./examples/Generalcaution.jpg "General Caution Traffic Sign"
[image6]: ./examples/limit30.jpg "30kph Limit Traffic Sign"
[image7]: ./examples/limit70.jpg "70kph Limit Traffic Sign"
[image8]: ./examples/nopassing.jpg "No Passing Traffic Sign"
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
[image22]: ./examples/web_images_classified.png "Web images classified"
[image23]: ./examples/softmax_prob1.png "70kph sign top 5 predictions"
[image24]: ./examples/softmax_prob2.png "30kph sign top 5 predictions"
[image25]: ./examples/softmax_prob3.png "General Caution sign top 5 predictions"
[image26]: ./examples/softmax_prob4.png "No Passing top 5 predictions"
[image27]: ./examples/softmax_prob5.png "Bumpy  top 5 predictions"


## Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![alt text][image9]
















Overview
---
In this project, I have implemented a traffic sign classifier using convolutional neural networks. 


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
**Training accuracy** : 99.6%
**Validation accuracy** : 98.8%
**Test accuracy** : 96.2%

## Rubric Points
See the [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)  for this project.

Here is a link to my [project code](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

For this project, we were asked to use the [German Trafic Sign Detection Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) (GTSRB) data. Download the formated dataset [here...](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
I used the numpy library to calculate summary statistics of the traffic signs data set. The code for this step is contained in the second code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb) :

* The size of training set is 34799 images.
* The size of validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the  [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb) :


Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed accross the different classes.

![alt text][image10]

And here is a sample of the training dataset.
![alt text][image11]

###Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
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
And here they are after  normalization:
![alt text][image18]




























#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

#### Data augmentation
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

#### Validation set
The supplied Pickle file already contained a Train, Testing and Validation set.

After data augmentation :
* My training set contained : 86430 examples
* The validation set : 4410 examples
* The testing set : 12630 examples


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb)

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

![alt text][image20]
























































I used the suggested as the starting point [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) to design this model architecture:

> Usual ConvNets are organized in strict feed-forward layered architectures in which the output of one layer is fed only to the layer above. Instead, the output of the first stage is branched out and fed to the classifier, in addition to the output of the second stage [...]. [W]e use the output of the first stage after pooling/subsampling rather than before. Additionally, applying a second subsampling stage on the branched output yielded higher accuracies than with just one. Therefore the branched 1st-stage outputs are more subsampled than in traditional ConvNets but overall undergoes the same amount of subsampling [...] than the 2nd-stage outputs. The motivation for combining representation from multiple stages in the classifier is to provide different scales of receptive fields to the classifier. In the case of 2 stages of features, the second stage extracts “global” and invariant shapes and structures, while the first stage extracts “local” motifs with more precise details.

I added a bit of dropout on the fully connected layer to regularize the training and prevent overfiting.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th, 11th, 12th, 13th and 14th code cells of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb). 

To train the model, I used a batch size of 256, a maximum epochs number of 100 and a fixed learning rate of 0.001. I also used a early stopping criteria to prevent overfitting. I used *adamoptimizer* with default settings for optimization. The optimization process itself took about 1 hour on a AWS small instance.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb). 

My final model results were:
* Training set accuracy : 99.6%
* Validation set accuracy : 98.8%
* Test set accuracy : 96.2%

I choose the model from the suggested article. I believed it was revelant since a great name in the domain is on the paper (Yann Lecun). Plus their model is applied on exactly the same dataset as us. This paper also sets a new "test" accuracy reccord for the dataset and finaly Udacity suggested the article to its students so it must be a bit relevant! The model looks very much alike the original LeNet model but it uses an inovative connection that skips the second convolution layer to get directly to the fully connected layer.

I think my model's performance is correct without being outstanding. From the discrepancy between the training accuracy and the test accuracy figures, we can see that the model has been a bit overfitting the training and validation set. Its ability to generalize seems a bit comprised. Maybe better/more agressive regularization techniques would have helped, along with a better stopping criteria. A decaying learning rate would also have been nice to play arround with..
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The code to load these images is located in the 16th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb). 

The first image (bumpy road) should be easy to classify since the sign is perfectly ligthed and level. The texture and the watermark on the white part may be tricky for the model, but should be ok.

The 2nd image (General Caution) should be a bit more difficult to classify since the sign is titled.

In the 3rd image (30km/h) we see a perspective effect that deforms the sign in a way that wasn't teached to the model. But since the defromation seems minor, I expect the model to be able to recognize the sign.

The 4th image (70km/h) should be the easiest to classify since it is the one with the crispiest image without any defromation/tilt

The 5th image (no passing) should be difficult to classify because of the watermark on top of it.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 17th cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb). 

Here are the results of the prediction:

| Image	                         |     Prediction                         | 
|:------------------------------:|:------------------------------------:| 
| Bumpy road      		| Bumpy road   		        | 
| General Caution		| General Caution	   		|
| 30 km/h			| 30 km/h				|
| 70 km/h	      		| 70 km/h		                |
| No passing    		| No passing                    	|

![alt text][image22]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set (96.2%)...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th code cell of the [IPython notebook](https://github.com/mlandry1/CarND/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb). 

For the first image, the model is relatively sure that this is a 70kph sign (probability of 0.61). The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:-----------------------:|:------------------------------------:| 
| .61         		| Speed limit (70km/h)   	| 
| .31    			| Speed limit (80km/h)	        |
| .28				| Speed limit (20km/h)           |
| .25	      			| Speed limit (30km/h)		|
| .21				| Speed limit (120km/h) 	|

![alt text][image23]

For the second image ... 

| Probability         	|     Prediction	        		| 
|:-----------------------:|:------------------------------------:| 
| .44         		| Speed limit (30km/h)   	| 
| .24     			| Speed limit (80km/h)	        |
| .17				| Speed limit (20km/h)           |
| .08	      			| Speed limit (70km/h)		|
| .05				| Speed limit (50km/h) 		|

![alt text][image24]

For the third image ... 

| Probability         	|     Prediction	        		| 
|:-----------------------:|:------------------------------------:| 
| .74         		| General caution		  	| 
| .21     			| Traffic signals		        |
| .07				| Pedestrians  		                |
| -.10      			| Bumpy Road				|
| -.12			| Road narrows on the right	|

![alt text][image25]

For the fourth image ... 

| Probability         	|     Prediction	        		| 
|:-----------------------:|:------------------------------------:| 
| .28         		| No passing  			  	| 
| .22     			| No passing for vehicles over 3.5 metric tons         |
| .13				| Vehicles over 3.5 metric tons prohibited	           |
| -.04      			| No vehicles				|
| -.07			| Stop					|

![alt text][image26]

For the fifth image ... 

| Probability         	|     Prediction	        		| 
|:-----------------------:|:------------------------------------:| 
| .91         		| Bumpy road 		         	| 
| .14     			| Slippery road 		        |
| .11				| Bicycles crossing		        |
| -.05			| Road work			        |
| -.12			| Turn left ahead      		|

![alt text][image27]