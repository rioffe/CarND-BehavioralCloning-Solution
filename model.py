from pandas import read_csv as csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, ELU
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from scipy.ndimage.interpolation import shift
import numpy as np
import cv2
from platform import system
from random import randint

# The original images coming from the cameras are 320 by 160
# We found the for the problem at hand we can easily resize the image down to 80 by 40.
# Resizing significantly shortens the computation time
h = 40
w = 80

# We assume that driving_log.csv is located in the current directory
# and images from the cameras are located in IMG directory
# If the camera data and driving_log.csv are in the data directory, adjust accordingly
data = csv('driving_log.csv')
nrows = data.shape[0]
y_train = data['steering']

# This routine only used during generation of validation arrays
# Image is read, converted to HSV space, resized down to 80 by 40 and normalized.
def read_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = cv2.resize(im, (0,0), fx=.25, fy=.25)
    im = np.interp(im, [0, 255], [-0.5, 0.5])
    return im

# Read the image, convert to HSV space and resize down to 80 by 40 (to a quarter of a size in each direction)
def read_image_hsv(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = cv2.resize(im, (0,0), fx=.25, fy=.25)
    return im

# given the row number, randomly read the image from either left, center or right camera
# for the left and right cameras adjust the steering angle 
# return image and a corresponding steering angle
# Steering angle adjustment based on Vivek Yadav's blog
def random_read_image(k):
    center_path = data['center'][k]
    left_path   = data['left'][k][1:]
    right_path  = data['right'][k][1:]

    steering    = y_train[k] 
    im = read_image_hsv(center_path)
    i = randint(0, 2)
    if (i == 1):
        im = read_image_hsv(left_path)
        steering += .25
    if (i == 2):
        im = read_image_hsv(right_path)
        steering -= .25

    return im, steering

# from http://stackoverflow.com/questions/9170271/flipping-a-image-vertically-relation-ship-between-the-original-picture-and-the
# Horizontal image flip
def flip_image(im):
    flipped_im = np.ndarray((im.shape), dtype='uint8')
    flipped_im[:,:,0] = np.fliplr(im[:,:,0])
    flipped_im[:,:,1] = np.fliplr(im[:,:,1])
    flipped_im[:,:,2] = np.fliplr(im[:,:,2])
    return flipped_im

# Given the input image, steering pair, randomly flip the image and return the resulting image and negated steering
def random_flip_image(pair):
    im, steering = pair
    i = randint(0, 1)
    if (i == 1):
        im = flip_image(im)
        steering = -steering
    return im, steering

# Given the input image, steering pair, normalize the image and return normalized image and unchanged steering
def interpolate_image(pair):
    im, steering = pair
    im = np.interp(im, [0, 255], [-0.5, 0.5])
    return im, steering

# Given the input image, steering pair, randomly shift the image left or right to a random number of pixels
# We adjust the steering based on a number of pixels we shifted and the direction of the shift
# Return the adjusted image and a modified steering
# Note 1: shift distances are set to emphasize large shifts, which help the model to learn sharper turns
# Note 2: the number that we multiply shift distance by to obtain steering shift is based on Vivek Yadav's blog,
# adjusted for the fact that we deal with 80 by 40 images, instead of 320 by 160 images, so constant is 4X larger.
def random_shift_image(pair):
    im, steering = pair
    i = randint(0,1)
    if (i == 0):
        shift_distance = float(randint(-25, -10))
    else:
        shift_distance = float(randint(10, 25))
  
    shift_steering = 0.016*shift_distance
    steering += shift_steering

    im = shift(im, [0.0, shift_distance, 0.0], mode='nearest') 
    return im, steering

# Given the input image, steering pair, randomly darken the image. Return darkened image and unchanged steering.
# Image is darkened by reducing the V channel of an HSV image. Note that the range to darken was found empirically 
# for the model to perform best on both the first and the second track.
def random_dark_image(pair):
    im, steering = pair
    h_, s, v = cv2.split(im)

    darken = randint(0, 125)
    v = np.where(v > darken, v - darken, 0)

    im = cv2.merge((h_, s, v))
    return im, steering

# We generate images from the first 90% of the rows in the Udacity provided datacet.
# We use the steering threshold to find images with steering angles larger than steering theshold, based on the Vivek Yadav's blog.
# We perform the following pipeline to generate one image in a batch:
# 1. Find a random row within the first 90% of the rows of the data set with a steering angle above threshold
# 2. Randomly read image from left, center or right camera for that row
# 3. Randomly darken the image
# 4. Randomly flip the image (or not)
# 5. Normalize the image
# 6. Randomly shift the image left or right by a fairly large amount
def random_generator(batch_size = 128, steering_threshold = 0.1):
    i = 0
    while 1:
        X_batch = np.zeros((batch_size, h, w, 3))
        y_batch = np.zeros(batch_size)

        for i in range(batch_size):
            k = randint(0, int(0.9*nrows)-1)
            steering    = y_train[k]

            while (abs(steering) < steering_threshold):
                k = randint(0, int(0.9*nrows)-1)
                steering    = y_train[k]

            X_batch[i], y_batch[i] = random_shift_image(interpolate_image(random_flip_image(random_dark_image(random_read_image(k)))))
        
        yield (X_batch, y_batch)

# Generation of validation arrays is a much simpler affair
# We generate validation arrays from the last 10% rows of the data set
# We generate images only from the center camera, read the image, scale it to 80 by 40 pixels, convert it to HSV space 
# and normalize it. We yield a batch of normalized and scaled HSV images and the corresponding steering angles.
def generate_validation_arrays(batch_size = 128):
    i = 0
    while 1:
        X_batch = np.zeros((batch_size, h, w, 3))
        y_batch = np.zeros(batch_size)
        
        for i in range(batch_size):
            k = randint(int(0.9*nrows), nrows-1)
            steering    = y_train[k]
            center_path = data['center'][k]

            im = read_image(center_path)
            X_batch[i] = im
            y_batch[i] = steering
            
        yield (X_batch, y_batch)

# We use the model from the NVidia's "End to End Learning for Self-Driving Cars" paper
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# The model is modified in the following ways from the orinal described in the paper:
# 1. The input images are 80 by 40 by 3 in HSV space instead of 200 by 66 by 3 in YUV space
# 2. The normalization is applied to images outside the model (we experimented with the Lambda layer, 
#    but found no perfomance difference)
# 3. The last two convolutional blocks don't have max pooling layers in them accounting for the smaller input image size.
# 4. Instead of a more traditional RELU activation, we use ELU activation, which we found to perform much better: 
#    note, that both open-sourced Comma.AI steering model and some blogs use ELUs as well. 
# 5. We apply dropout only to the last of convolutional layers and the first of the fully connected layers.
model1 = Sequential()

model1.add(Convolution2D(24, 5, 5, border_mode='same', input_shape=(h, w, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Activation('elu'))

model1.add(Convolution2D(36, 5, 5, border_mode='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Activation('elu'))

model1.add(Convolution2D(48, 5, 5, border_mode='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Activation('elu'))

model1.add(Convolution2D(64, 3, 3, border_mode='valid'))
model1.add(Activation('elu'))

model1.add(Convolution2D(64, 3, 3, border_mode='valid'))
model1.add(Activation('elu'))

model1.add(Dropout(p=0.5))

model1.add(Flatten())
model1.add(Dense(output_dim=100))
model1.add(Activation("elu"))

model1.add(Dropout(p=0.5))

model1.add(Dense(output_dim=50))
model1.add(Activation("elu"))
model1.add(Dense(output_dim=10))
model1.add(Activation("elu"))
model1.add(Dense(output_dim=1))

print(model1.summary())

# We use Adam optimizer and actually found a learning rate of 0.0001 to perform best 
# (values of 0.1, 0.001, 0.00001 and 0.00003 were tried but found lacking)
adam = Adam(lr=0.0001)
model1.compile(loss='mean_squared_error', optimizer=adam)

# We train for 10 epochs, gradually reducing the steering threshold, similar to Vivek Yadav's blog.
# Couple of things to point out that are different from other solutions:
# 1. We start with a very large steering angle threshold of 0.3 - only 148 out of 8036 center camera images
#    meet that specification. This is consistent with the NVidia's paper suggestion to focus training on turns.
# 2. We found 10240 generated images per epoch to be sufficient to train the model, many times the model in the first epoch 
#    works flawlessly on both tracks; also using just 5120 images per epoch is sufficient for training the model to run on 
#    the first track only. 
# 3. We don't use the testing data set: instead we test each of the 10 produced models on two tracks and find the ones 
#    that perform best. Note: on some occasions, the best model is produced in the first epoch (only after 1 minute of training!).
# I select the best model out of 10 generated here
for i in range(10):
    model_name = 'model_' + str(i)
    
    history = model1.fit_generator(random_generator(steering_threshold=(0.30 - 0.03*i)), samples_per_epoch=10240, nb_epoch=1, 
                               validation_data=generate_validation_arrays(), nb_val_samples=1024)
    with open(model_name + '.json', mode='w', encoding='utf8') as f:
        f.write(model1.to_json())

    model1.save_weights(model_name + '.h5')
