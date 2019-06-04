#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:51:30 2019

@author: Atharva
"""
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

trdata = 66024
vltdata = 12567
batch = 32

arr_result = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

training_data = 'nist_new/training'
validation_data = 'nist_new/validation'

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=26,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen=ImageDataGenerator(rescale = 1./255)

training_set=train_datagen.flow_from_directory(directory = training_data,
                                                 target_size = (64, 64),
                                                 batch_size = batch,
                                                 class_mode = 'sparse')

test_set=test_datagen.flow_from_directory(directory = validation_data,
                                            target_size = (64, 64),
                                            batch_size = batch,
                                            class_mode = 'sparse')

model.fit_generator(training_set,steps_per_epoch = 2000,         
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 400)                 #trdata//batch = 2000
                                                                #vldata//batch = 400

model.save('modelwts.h5')

from keras.preprocessing import image

model=load_model('modelwts.h5')

test_image=image.load_img('Q.png', target_size = (64, 64))
test_image=(image.img_to_array(test_image))/255
test_image=np.expand_dims(test_image, axis = 0)
result=model.predict(test_image)
np.reshape(result, 26)
training_set.class_indices
  
maxval = np.amax(result)
index = np.where(result == maxval)
print('\n','Predicted Character:',arr_result[index[1][0]])        
