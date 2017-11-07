# Semantic Segmentation
### Introduction
In this project, the pixels of a road are labeled using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```
### Implementation details

* Adam optimiser
* Learning rate : 1e-4
* Dropout : 50%
* L2 Reg: 1e-3
* Epochs : 30
* Batch size : 16
* [FCN-8 network architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

### Results
Here are a couple of great results I was able to achieve:

<img src=./images/gif1.gif width="700">

<img src=./images/gif2.gif width="700">

<img src=./images/gif3.gif width="700">
