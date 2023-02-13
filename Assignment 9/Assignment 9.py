# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:48:10 2020

@author: Khamar Uz Zama
"""
# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from tensorflow.keras import layers,models

from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,BatchNormalization,Activation,Add,Flatten,Dense,Input,ZeroPadding2D

from tensorflow.keras.models import Model, Sequential
from keras.optimizers import SGD
from tensorflow.keras import activations

import keras.backend as K
from matplotlib import pyplot

import os
from tensorflow.keras.initializers import glorot_uniform

folder = "cifar_attempts"
file1 = "data1.npz"
file2 = "data2.npz"
file3 = "data3.npz"
file4 = "data4.npz"

folderPath = os.path.join(os.getcwd(), folder)


def splitNPZ(b):

    train_imgs = b['train_imgs']
    train_lbls = b['train_lbls']
    
    val_imgs = b['val_imgs']
    val_lbls = b['val_lbls']
    
    test_imgs = b['test_imgs']
    test_lbls = b['test_lbls']

    return train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls


def lbls_countPlot(train_lbls, val_lbls, test_lbls):
    sns.countplot(train_lbls)
    sns.countplot(val_lbls)
    sns.countplot(test_lbls)

def plotImages(imgs, lbls):
    plt.figure(figsize=(10,10))
    
    for i in range(25):
      plt.subplot(5,5,i+1)
      
      idx = np.random.randint(1,100)
      
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(imgs[idx], cmap=plt.cm.binary)
      plt.xlabel(lbls[idx])
      
    plt.show()
    
def preProcess(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls):

    train_norm = train_imgs.astype('float32')
    val_norm = val_imgs.astype('float32')
    test_norm = test_imgs.astype('float32')
    
    # normalize
    train_norm = train_norm / 255.0
    val_norm = val_norm / 255.0
    test_norm = test_norm / 255.0
    
    trainY = to_categorical(train_lbls)
    valY = to_categorical(val_lbls)
    testY = to_categorical(test_lbls)

    return train_norm, trainY, val_norm, valY, test_norm, testY




def define_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.add(layers.Activation(activations.relu))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file


def plotHistory(history):
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

model = define_model()

# data1

file1Path = os.path.join(folderPath, file1)

data1 = np.load(file1Path)


train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = splitNPZ(data1)

# Free memory
del data1


plotImages(train_imgs, train_lbls)
lbls_countPlot(train_lbls, val_lbls, test_lbls)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = preProcess(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls)


d1history = model.fit(train_imgs, train_lbls, epochs=2, batch_size=64, verbose=1, 
          validation_data=(val_imgs, val_lbls))

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_imgs, test_lbls, batch_size=64)
print("test loss, test acc:", results)

#
## Generate predictions (probabilities -- the output of the last layer)
## on new data using `predict`
#print("Generate predictions for 3 samples")
#predictions = model.predict(x_test[:3])
#print("predictions shape:", predictions.shape)

plotHistory(d1history)


# data2

file1Path = os.path.join(folderPath, file2)

data2 = np.load(file1Path)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = splitNPZ(data2)

# Free memory
del data2


plotImages(train_imgs, train_lbls)
lbls_countPlot(train_lbls, val_lbls, test_lbls)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = preProcess(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls)


d2history = model.fit(train_imgs, train_lbls, epochs=2, batch_size=64, verbose=1, 
          validation_data=(val_imgs, val_lbls))

results = model.evaluate(test_imgs, test_lbls, batch_size=64)
plotHistory(d2history)

# data3

file1Path = os.path.join(folderPath, file3)

data3 = np.load(file1Path)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = splitNPZ(data3)

# Free memory
del data3


plotImages(train_imgs, train_lbls)
lbls_countPlot(train_lbls, val_lbls, test_lbls)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = preProcess(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls)


d3history = model.fit(train_imgs, train_lbls, epochs=3, batch_size=64, verbose=1, 
          validation_data=(val_imgs, val_lbls))

results = model.evaluate(test_imgs, test_lbls, batch_size=64)
plotHistory(d3history)

# data4

file1Path = os.path.join(folderPath, file3)

data4 = np.load(file1Path)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = splitNPZ(data4)

# Free memory
del data4


plotImages(train_imgs, train_lbls)
lbls_countPlot(train_lbls, val_lbls, test_lbls)

train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = preProcess(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls)


d4history = model.fit(train_imgs, train_lbls, epochs=3, batch_size=64, verbose=1, 
          validation_data=(val_imgs, val_lbls))

results = model.evaluate(test_imgs, test_lbls, batch_size=64)
plotHistory(d4history)


