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

h = 40
w = 80

data = csv('driving_log.csv')
nrows = data.shape[0]
y_train = data['steering']

def read_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = cv2.resize(im, (0,0), fx=.25, fy=.25)
    im = np.interp(im, [0, 255], [-0.5, 0.5])
    return im

def read_image_hsv(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = cv2.resize(im, (0,0), fx=.25, fy=.25)
    return im

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
def flip_image(im):
    flipped_im = np.ndarray((im.shape), dtype='uint8')
    flipped_im[:,:,0] = np.fliplr(im[:,:,0])
    flipped_im[:,:,1] = np.fliplr(im[:,:,1])
    flipped_im[:,:,2] = np.fliplr(im[:,:,2])
    return flipped_im

def random_flip_image(pair):
    im, steering = pair
    i = randint(0, 1)
    if (i == 1):
        im = flip_image(im)
        steering = -steering
    return im, steering

def interpolate_image(pair):
    im, steering = pair
    im = np.interp(im, [0, 255], [-0.5, 0.5])
    return im, steering

def random_shift_image(pair):
    im, steering = pair
    #shift_distance = float(randint(-25, 25))
    i = randint(0,1)
    if (i == 0):
        shift_distance = float(randint(-25, -10))
    else:
        shift_distance = float(randint(10, 25))
  
    shift_steering = 0.016*shift_distance
    steering += shift_steering

    im = shift(im, [0.0, shift_distance, 0.0], mode='nearest') 
    return im, steering

def random_dark_image(pair):
    im, steering = pair
    h_, s, v = cv2.split(im)

    darken = randint(0, 125)
    v = np.where(v > darken, v - darken, 0)

    im = cv2.merge((h_, s, v))
    return im, steering

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

adam = Adam(lr=0.0001)
model1.compile(loss='mean_squared_error', optimizer=adam)

for i in range(10):
    model_name = 'model_dark_10K_' + str(i)
    
    history = model1.fit_generator(random_generator(steering_threshold=(0.30 - 0.03*i)), samples_per_epoch=10240, nb_epoch=1, 
                               validation_data=generate_validation_arrays(), nb_val_samples=1024)
    with open(model_name + '.json', mode='w', encoding='utf8') as f:
        f.write(model1.to_json())

    model1.save_weights(model_name + '.h5')
