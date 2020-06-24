import os
import csv
from math import ceil

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Convolution2D, Dropout


samples = []
with open('./data/forward/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
with open('./data/backwards/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

sidecam_correction = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                center_image = cv2.imread(batch_sample[0])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                left_image = cv2.imread(batch_sample[1])
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = center_angle + sidecam_correction
                images.append(left_image)
                angles.append(left_angle)
                
                right_image = cv2.imread(batch_sample[2])
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = center_angle - sidecam_correction
                images.append(right_image)
                angles.append(right_angle)
                
                # flip horizontally
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)
                
                left_image_flipped = np.fliplr(left_image)
                left_angle_flipped = -left_angle
                images.append(left_image_flipped)
                angles.append(left_angle_flipped)
                
                right_image_flipped = np.fliplr(right_image)
                right_angle_flipped = -right_angle
                images.append(right_image_flipped)
                angles.append(right_angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

            
# Hyperparameters
batch_size=64
ch, row, col = 3, 160, 320  # Original image format
drop_rate = 0.3
initial_lr = 0.01
lr_decay=0.1
epochs = 15

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Build NN (inspired by http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)),
                     input_shape=(row, col, ch)))  # crop sky and trunk
model.add(Lambda(lambda x: x/255.0 - 0.5))  # Preprocess incoming data, centered around zero with small standard deviation 
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(rate=drop_rate))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(rate=drop_rate))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(rate=drop_rate))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(rate=drop_rate))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(rate=drop_rate))
model.add(Flatten())
model.add(Dropout(rate=drop_rate))
model.add(Dense(100))
model.add(Dropout(rate=drop_rate))
model.add(Dense(50))
model.add(Dropout(rate=drop_rate))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# Train NN
opt = keras.optimizers.Adam(lr=initial_lr, decay=lr_decay)
model.compile(loss='mse', optimizer=opt)
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=epochs, verbose=1)
model.save('model.h5')

