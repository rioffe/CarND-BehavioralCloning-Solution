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
[flipped]: ./examples/center_image_flipped.png "Flipped Image"
[center]: ./examples/center_image_right_turn.png "Center Image of a Right Turn"
[dark_samples]: ./examples/dark_samples.png "Dark Samples"
[flips_shifts_and_dark]: ./examples/flips_shifts_and_dark.png "Flips, Shifts and Darkened Images from a Single Center Image"
[left_right_center]: ./examples/left_center_right_samples.png "Samples of Images from Left, Center and Right Cameras"
[random_shifts]: ./examples/random_shifts.png
[sorted_steering]: ./examples/sorted_steering_values.png "Sorted Steering Values"
[steering_values]: ./examples/steering_values.png "Steering Values over Time"


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

I decided not to use a Keras lambda layer, since in my experiments I didn't find that it improves performance, so image normalization is typically done either right after reading and scaling the image.

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 197 and 203). Adding more dropout layers proved detrimental to model performance.

The model was trained and validated on different data sets to ensure that the model was not overfitting: the first 90% of rows of the Udacity data set was used for generating the training dataset, and the remaining 10% of rows was used for generating the validation dataset. The model was tested by running it through the simulator on both the first and the second track and ensuring that the vehicle could stay on the track (in the case of the second track, the car reaches the "Road Closed" signs without crashing).

####3. Model parameter tuning

The model used an adam optimizer, and actually found a learning rate of 0.0001 to perform best (model.py line 215).
Values of 0.1, 0.001, 0.00001 and 0.00003 were tried but found lacking. I also found the batch size of 128 to produce better results than sizes of 64 and 256.

####4. Appropriate training data

We generate images from the first 90% of the rows in the Udacity provided data set.
We use the steering threshold to find images with steering angles larger than steering theshold, based on the Vivek Yadav's blog.
We perform the following pipeline to generate one image in a batch:
* Find a random row within the first 90% of the rows of the data set with a steering angle above threshold
* Randomly read image from left, center or right camera for that row
* Randomly darken the image
* Randomly flip the image (or not)
* Normalize the image
* Randomly shift the image left or right by a fairly large amount

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to adapt the architecture described in the NVidia paper to the problem at hand. 

My first step was to implement the model literally with the dropout layers added after every convolutional and fully connected block. I thought this model might be appropriate because NVidia already demostrated the success of the model on a real car. I gradually realized that I need to remove some of the dropout layers, since the model didn't show any improvements.

At one point I got the model fully operational on the first track, but then realized that I was severely overfitting, since the error on the training set dipped below the error of the validation set. I also realized the my training and validation sets were possibly overlapping, so I had to rewrite the training and validation generators to use non-overlapping parts of the data set.

My original approach to generating training data was ineffective as well, since I always included unmodified images from the center, left and right cameras in a batch. For large steering threshold, this meant always using the same images over and over again, reducing the new data the model was trained on in an epoch. I eventually realized that I have to switch to approach described in Vivek Yadav's blog, which guaranteed that there was very low probability of a model to see exactly the same image within an epoch, even if underlying data contained only 3 * 398 images (e.g. for the steering threshold of 0.3).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I did the following: 
* Flipped the image horizontally, which made the model handle right turns as confidently as left turns
* Used large image shifts (10 to 25 pixels in either left or right direction, simultaneously adjusting the steering angle)
* Used images from the left and the right cameras, simultaneously adding or subtracting to the steering value of the center image
* Switched from RGB to YUV and then finally to HSV color space, which helps the model clearly see the boundaries of the track lanes

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road. Note, that to enable successful runs around the second track, I had to add image darkening into my image generation pipeline, which is not necessary to reliably drive the car on the first track. Also note, that very large shifts are not strictly necessary for the good performance on the first track.

####2. Final Model Architecture

The final model architecture (model.py lines 167-209) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

My goal from the very beginning was to only use the Udacity dataset for training. I was able to produce a model that drove sufficiently well on the first track by using only center image, horizontal flip of the image and various symmetrical shifts of the center image. It didn't however generalize well to the second track. To account for large turns of the second track I was forced to focus on the very large shifts only, and then add left and right camera images and eventually added darkened images (though I overdid the darkening range and had to lower it to improve the model performance on the second track).

To capture good driving behavior, I started focusing my training on images generated with very large steering angles (e.g. starting with steering angles greater than 0.3). Here is the summary of the dataset with respect to steering angles:

* There is a total of 8036 image triples (left, center, right camera images) in the data set
* 4361 images have 0 steering
* The rest are split as follows: 1900 image triples with positive steering angles and 1775 images with negative steering angles
* There are 148 center images with the absolute value of steering greater than 0.4
* There are 250 center images with the absolute value of steering in the range (0.3, 0.4]
* There are 461 center images with the absolute value of steering in the range (0.2, 0.3]
* There are 1254 center images with the absolute value of steering in the range (0.1, 0.2]
* There are 1562 center images with the absolute value of steering in the range (0.0, 0.1]

Here is how the steering values look like over time:

![alt text][steering_values]

Here is how the steering values look when sorted: 

![alt text][sorted_steering]

We will first focus on training our network on images with absolute value of steering exceeding 0.3, and there are only 398 of those (well, actually 1196, if you count left and right camera images). So how do we generate 10240 images from those?


Here are some samples from the data set where the steering is either 0, negative or positve

![alt text][left_right_center]

To augment the data set, I randomly took images from left, center or right camera, flipped them, darkened them, and shifted them.

Here is an example image of center lane driving:

![alt text][center]

Here is a flipped image of the above: 

![alt text][flipped]

Now we take the first image and apply a number of random large left and right shifts. We focus on the large shifts to enable the model to handle large left and right turns:

![alt text][random_shifts]

We also create random dark images from the original by reducing the value of the V-channel of the image in HSV space:

![alt text][dark_samples]

We finally create a pipeline to generate flipped, shifted, and darkened images from the original, which could be left, center or right:

![alt text][flips_shifts_and_dark]

The training set generator creates images as described above from the first 90% of images in the Udacity dataset. After training for one epoch, the model weights are saved, the steering threshold is reduced by 1/10th of the value, and the training continues in such a way for 10 epochs. Training beyond 10 epochs proved unneccessary. In fact, frequently, the very best model is obtained on the first(!) epoch, since the model learns immediately how to handle large turns. The model is validated on the last 10% of the data set (only unmodified images from the center camera are used).

```
Epoch 1/1
10240/10240 [==============================] - 60s - loss: 0.1944 - val_loss: 0.0334
Epoch 1/1
10240/10240 [==============================] - 60s - loss: 0.1017 - val_loss: 0.0310
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0839 - val_loss: 0.0346
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0689 - val_loss: 0.0255
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0594 - val_loss: 0.0256
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0472 - val_loss: 0.0188
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0413 - val_loss: 0.0188
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0364 - val_loss: 0.0255
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0334 - val_loss: 0.0160
Epoch 1/1
10240/10240 [==============================] - 59s - loss: 0.0299 - val_loss: 0.0208
```

Here is a full [training log](./training_log.txt).

The model that I provide here was so successful (in fact it was generated right after the first epoch) that it was able to complete the second track in 2 minutes 31.77 seconds with a throttle value set at 0.45! Throttle values for typical models range from 0.27 to 0.31.
The remarkable fact is that only 1196 original images are used and only 398 original steering values to generate 10240 images for the first epoch, which proves enough to handle driving on the first and the second track with no issues at top speeds!
