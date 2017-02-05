#**Udacity Self-driving Car Nano Degree: Project 3 Solution - Behavioral Cloning** 

## by Robert Ioffe

### February 5, 2017

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json, model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I use the model from the NVidia's ["End to End Learning for Self-Driving Cars" paper] (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
The model is modified in the following ways from the orinal described in the paper:
* The input images are 80 by 40 by 3 in HSV space instead of 200 by 66 by 3 in YUV space
* The normalization is applied to images outside the model (we experimented with the Lambda layer, 
   but found no perfomance difference)
* The last two convolutional blocks don't have max pooling layers in them accounting for the smaller input image size.
* Instead of a more traditional RELU activation, we use ELU activation, which we found to perform much better: 
   note, that both open-sourced Comma.AI steering model and some blogs use ELUs as well. 
* We apply dropout only to the last of convolutional layers and the first of the fully connected layers.

Here is a recap of the model (lines 167 throught 209):
My model consists of five convolutional blocks. The first three convolutional blocks consist of sets of 5 by 5 convolutions (24, 36,
and 48 deep respectively), followed by 2 by 2 max pooling followed by ELU activation. The fourth and fifth blocks are 3 by 3 convolutions followed by ELU activation (max pooling for these blocks is omitted). There are four fully connected layers of sizes 100, 50, 10 and 1, the first three of them followed by ELUs. Dropout of 0.5 is applied as described above.

I decided not to use a Keral lambda layer, since in my experiments I didn't find that it improves performance, so image normalization is typically done either right after reading and scaling the image.

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 197 and 203). Adding more dropout layers proved detrimental to model performance.

The model was trained and validated on different data sets to ensure that the model was not overfitting: the first 90% of rows of the Udacity data set was used for generating the training dataset, and the remaining 10% of rows was used for generating the validation dataset. The model was tested by running it through the simulator on both the first and the second track and ensuring that the vehicle could stay on the track (in the case of the second track, the car reaches the "Road Closed" signs without crashing).

####3. Model parameter tuning

The model used an adam optimizer, and actually found a learning rate of 0.0001 to perform best (model.py line 215).
Values of 0.1, 0.001, 0.00001 and 0.00003 were tried but found lacking.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
