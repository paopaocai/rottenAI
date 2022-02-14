# -*- coding: utf-8 -*-


# Data Import
import os
from PIL import Image
import numpy as np

# %% Data Import
currentWorkingDir = os.getcwd()

# Function to import images, resizing them to 256x256 and returning them as a numpy array
def dataImporterAndResizer(pathForImages,imageHeight=64,imageWidth=64): 
    os.chdir(pathForImages) #Changing working directory to images path for the upcoming 'for' loop
    images = np.zeros([len(os.listdir()),imageHeight,imageWidth,3],dtype=np.int32) #Initializing a zero image array with shape [totalImages, height, width, Colors] Colors = 1 for greyscale and 3 for rgb
    fileNum=0 #initializing file number to 0
    for file in os.listdir(): #Iterating through every file in the directory. 'file' gives the file name
        # print('Importing {}'.format(file))
        importedImage = Image.open(file) #Opening the image
        importedImage = importedImage.resize([imageHeight,imageWidth]) #resizing the image to imageHeightximageWidth
        images[fileNum,:,:,:] = np.array(importedImage) #Converting the image to a numpy array and storing it in the pre-initialized rottenImages array
        # print('Successfully imported {}'.format(file))
        fileNum += 1 #Incrementing file number
    return images

#Image Size Determination
imageHeight = 256
imageWidth = imageHeight
# Importing Rotten Images and resizing them to imageHeightximageWidth
pathForRottenImages = 'F:\Project AI Ext\RottenAI\Data\Rotten' #Path to rotten images
rottenImages = dataImporterAndResizer(pathForRottenImages, imageHeight, imageWidth)
    
# Importing Fresh Images - Repeating the same steps above
pathForFreshImages = 'F:\Project AI Ext\RottenAI\Data\Fresh'
freshImages = dataImporterAndResizer(pathForFreshImages, imageHeight, imageWidth)


os.chdir(currentWorkingDir)

del currentWorkingDir, pathForRottenImages, pathForFreshImages

# %% Data Normalization

imageMin = 0.0
imageMax = 255.0

#Normalizing Rotten and Fresh images to -1,1 and changing dtype to float32
normRottenImages = -1 + (2*(rottenImages-imageMin)/(imageMax-imageMin)) 
normRottenImages = normRottenImages.astype(np.float32)
normFreshImages = -1 + (2*(freshImages-imageMin)/(imageMax-imageMin))
normFreshImages = normFreshImages.astype(np.float32)

# %% Neural Net

# Creating Training Data

np.random.shuffle(normRottenImages) #Shuffling data
np.random.shuffle(normFreshImages)  #Shuffling data

trainDataInputs = np.concatenate((normRottenImages[:-20,...],normFreshImages[:-20,...]),axis=0)
trainDataOutputs = np.concatenate((np.ones([len(normRottenImages)-20,1],dtype=np.float32),np.zeros([len(normFreshImages)-20,1],dtype=np.float32)),axis=0)

testDataInputs = np.concatenate((normRottenImages[-20:,...],normFreshImages[-20:,...]),axis=0)
testDataOutputs = np.concatenate((np.ones([20,1],dtype=np.float32),np.zeros([20,1],dtype=np.float32)),axis=0)


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='tanh', input_shape=(imageHeight, imageWidth, 3), kernel_regularizer='l2'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='tanh', kernel_regularizer='l2'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), padding='same', activation='tanh', kernel_regularizer='l2'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2, (3, 3), padding='same', activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2, (3, 3), padding='same', activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='tanh', kernel_regularizer='l2'))
model.add(layers.Dense(1,activation='sigmoid', kernel_regularizer='l2'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

history = model.fit(trainDataInputs, trainDataOutputs, epochs=20, validation_data=(testDataInputs,  testDataOutputs))

testOutput = model.predict(testDataInputs)

f,ax = plt.subplots(4,10)
imageNum = 0
for i in range(4):
    for j in range(10):
        ax[i,j].imshow((testDataInputs[imageNum,...]+1)/2.0)
        ax[i,j].title.set_text(np.round(testOutput[imageNum],2))
        imageNum += 1
