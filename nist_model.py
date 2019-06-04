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

trdata = 66024            # My dataset had these many images. Verify the number of imgaes in your dataset by
vltdata = 12567           # executing lines 33-48 on your compiler
batch = 32

#The arr_result list may be modified according to the characters that are present in the dataset as independent categories
arr_result = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

training_data = 'nist_new/training'           #This was the location of the dataset on my disk. 
validation_data = 'nist_new/validation'       #The location of your dataset may vary.      

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))

model.add(Dense(units=26,activation='softmax'))  # The dimentionality of output space depends on the number of characters in question
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
                         validation_data = test_set,             #vldata//batch = 400
                         validation_steps = 400)                 #trdata//batch = 2000
                                                                 
model.save('modelwts.h5')
#Model Creation and Training code ends here


#This is the code for testing
from keras.preprocessing import image
model=load_model('modelwts.h5')

test_image=image.load_img('test_image.png', target_size = (64, 64))     #Name and directory location of the image being tested
test_image=(image.img_to_array(test_image))/255
test_image=np.expand_dims(test_image, axis = 0)
result=model.predict(test_image)
np.reshape(result, 26)
training_set.class_indices
  
maxval = np.amax(result)
index = np.where(result == maxval)

#Final Output of Prediction
print('\n','Predicted Character:',arr_result[index[1][0]])        
