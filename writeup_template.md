# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[sample]: ./examples/sample_img.jpg "Sample Image"
[sample_flipped]: ./examples/sample_img_flipped.jpeg "Flipped Image"
[sample_reverse_driving]: ./examples/sample_img_reverse_driving.jpg "Sample Image Reverse Driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In addition to my neural network model.py also contains the data augmentation and reading images functions.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 filters (model.py lines 65-82) 

The model includes RELU layers to introduce nonlinearity (code lines 67-97), and the data is normalized in the model using a Keras lambda layer (code line 62). 

The model also includes BatchNormalization to normalize the weights foreach batch and Dropout to prevent overfitting. I chosed using BatchNormalization before activation functions (in forums there was a discussion on where to use it before or after activation fucntion, using before activation function worked for me) and used Dropout right after every fully connected layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88-96). 

In addition to dropout layer, model contains BatchNormalization layer which also helps to reduce overfitting (model.py lines 66-78).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. For recovering from the left and right sides of the road I used left and right side camera images in my training.

For details about how I created the training data, see the next section below. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because it has proved success in self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used keras itself for spliting 20% of data as validation set.

To combat the overfitting, I modified the model so that it has BatchNormalization before every activation function and Dropout right after the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were some spots where the vehicle fell off the track in sharp turns. to improve the driving behavior in these cases, I included left and right camera images and added some random data augmentation (brightness and flip). In addition to data augmentation I completed the track in reverse direction to collect additional data and used that data also for training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes can be seen below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 normalized and cropped image   		| 
| Convolution 5x5     	| 2x2 stride, 24 filters 	                    |
| Batch Normalization   |  	                                            |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, 36 filters 	                    |
| Batch Normalization   |  	                                            |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, 48 filters 	                    |
| Batch Normalization   |  	                                            |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, 64 filters 	                    |
| Batch Normalization   |  	                                            |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, 64 filters 	                    |
| RELU					|												|
| Flatten   	      	|      					                        |
| RELU					|												|
| Fully connected		| 512								            |
| DROPOUT				| 0.2 keep probability							|
| RELU					|												|
| Fully connected		| 100								            |
| DROPOUT				| 0.5 keep probability							|
| RELU					|												|
| Fully connected		| 10								            |
| DROPOUT				| 0.5 keep probability							|
| RELU					|												|
| Fully connected		| 1             								|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][sample]

I then recorded driving from first track in reverse so that I would have more variety of data to use. Image from the reverse driving can be seen below :

![alt text][sample_reverse_driving]

To augment the data set, I also flipped images and angles thinking that this would help my model to use more data for learning. In addition to that I also applied random brightness to images. For example, here is an image that has then been flipped:

![alt text][sample_flipped]

After the collection process, I had 45.153 number of data points. I then preprocessed this data and created additional by 31.450 new data points. In total I trained my model with 76.610 data points

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by after 10 epochs model was not improving anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.
