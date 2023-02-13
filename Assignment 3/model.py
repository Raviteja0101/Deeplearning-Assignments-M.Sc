# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:47:32 2020

@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_labels.shape)

# Whats the meaning of reshape?
train_labels = train_labels.reshape((-1,))
test_labels = test_labels.reshape((-1,))

plt.imshow(train_images[0], cmap="Greys_r")

# first difference: data is not reshaped to 784 anymore, but 28x28x1
# note the 1 color channel!! this is important
# Whats the meaning of reshape?
train_data = tf.data.Dataset.from_tensor_slices(
    (train_images.reshape([-1, 32, 32, 3]).astype(np.float32) / 255, train_labels.astype(np.int32)))
train_data = train_data.shuffle(buffer_size=60000).batch(128).repeat()

test_data = tf.data.Dataset.from_tensor_slices(
    (test_images.reshape([-1, 32, 32, 3]).astype(np.float32) / 255, test_labels.astype(np.int32))).batch(10000)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_steps = 2000

initializer = tf.keras.initializers.GlorotUniform()
def build_model(width, height, depth, classes):
	# initialize the input shape and channels dimension to be
	# "channels last" ordering
	inputShape = (height, width, depth)
	# build the model using Keras' Sequential API
	model = tf.keras.Sequential([
		# CONV => RELU => BN => POOL layer set
        # First Layer
		Conv2D(16, (3, 3), padding="same", input_shape=inputShape, kernel_initializer=initializer),
		Activation("relu"),
		MaxPooling2D(pool_size=(2, 2)),
        
		Conv2D(32, (3, 3), padding="same",  kernel_initializer=initializer),
		Activation("relu"),
		MaxPooling2D(pool_size=(2, 2)),
        
 		Conv2D(64, (3, 3), padding="same",  kernel_initializer=initializer),
		Activation("relu"),
		MaxPooling2D(pool_size=(2, 2)),       
        # FCC
		Flatten(),
		Dense(128),
		Activation("relu"),
		
        Dense(10),
        Activation("softmax")
	])
	# return the built model to the calling function
	return model

opt = tf.optimizers.Adam()
# from_logits = True!! #neverforget
loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


# this basically hasn't changed
model = build_model(32, 32, 3, 10)

for step, (img_batch, lbl_batch) in enumerate(train_data):

    if step > train_steps:
        break

    with tf.GradientTape() as tape:
        logits = model(img_batch)
        xent = loss_fn(lbl_batch, logits)

    grads = tape.gradient(xent, model.trainable_variables)
      
    opt.apply_gradients(zip(grads, model.trainable_variables))

    
    if not step % 100:
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                             tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))


# here's some evaluation magic ;) bonus: figure out how this works...
big_test_batch = next(iter(test_data))
test_preds = tf.argmax(model(big_test_batch[0]), axis=1,
                       output_type=tf.int32)
test_labels = tf.reshape(big_test_batch[1], [-1])
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, test_labels),
                             tf.float32))
print(acc)