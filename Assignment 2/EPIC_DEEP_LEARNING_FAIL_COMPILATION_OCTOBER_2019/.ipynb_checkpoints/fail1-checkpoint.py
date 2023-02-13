import tensorflow as tf
import os
import math
from datasets import MNISTDataset
import time
# first change: set up log dir and file writer(s)
os.getcwd()
logdir = os.path.join("logs", "linear" + str(time.time()))
train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))

# get the data
(train_imgs, train_lbls), (test_imgs, test_lbls) = tf.keras.datasets.mnist.load_data()
mnist = MNISTDataset(train_imgs.reshape((-1, 784)), train_lbls,
                     test_imgs.reshape((-1, 784)), test_lbls,
                     batch_size=128, seed=int(time.time()))


# define the model first, from input to output

# this is a super deep model, cool!
n_units = 100
n_layers = 8
w_range = 0.4

# just set up a "chain" of hidden layers
layers = []
for layer in range(n_layers):
    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu,
        kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                         maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001),
        activity_regularizer=tf.keras.regularizers.l2(0.1)))

# finally add the output layer
layers.append(tf.keras.layers.Dense(
    10, kernel_initializer=tf.initializers.RandomUniform(minval=-w_range,
                                                         maxval=w_range),
        bias_initializer=tf.initializers.constant(0.001),
        activity_regularizer=tf.keras.regularizers.l2(0.1)))

weightHistory = []
lr=0.0025
for step in range(300):
    img_batch, lbl_batch = mnist.next_batch()

    with tf.GradientTape() as tape:
        # here we just run all the layers in sequence via a for-loop
        out = img_batch
        for layer in layers:
            out = layer(out)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=out, labels=lbl_batch))
    weights = [var for l in layers for var in l.trainable_variables]
    grads = tape.gradient(xent, weights)
    for grad, var in zip(grads, weights):
        var.assign_sub(lr*grad)
    i=0        
#    with train_writer.as_default():
#        tf.summary.scalar("loss", xent, step=step)
#        tf.summary.histogram("logits", out, step=step)
#        for W in weights:
#            tf.summary.histogram("weights"+str(i), W, step=step)
#            i+=1
    if not step % 100:
        preds = tf.argmax(out, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))
#        with train_writer.as_default():
#            tf.summary.scalar("accuracy", acc, step=step)
#            tf.summary.image("input", tf.reshape(img_batch, [-1, 28, 28, 1]), step=step)

    History = [layers[0].trainable_variables[0]]
    weightHistory.append(History)
out = mnist.test_data
for layer in layers:
    out = layer(out)
test_preds = tf.argmax(out, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, mnist.test_labels), tf.float32))
#with test_writer.as_default():
#    tf.summary.scalar("accuracy", acc, step=step)
print("Final test accuracy: {}".format(acc))


import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
len(weightHistory)
weightHistory[45][0]
yy = weightHistory
x = 784        # board will be X by X where X = boardsize
y = 100
pad = 0               # padded border, do not change this!
initial_cells = 1500  # this number of initial cells will be placed in randomly generated positions
my_board = np.zeros((x+pad,y+pad))
my_board = weightHistory[25][0].numpy()
fig = plt.gcf()
im = plt.imshow(my_board)
plt.show()

i = 0
def animate(frame):
    print("xxxxxxx",frame)
    my_board = np.array(weightHistory[frame][0].numpy())
    im.set_data(my_board)
    return im

anim = animation.FuncAnimation(fig, animate, frames=10,interval=10)

anim.save('animation.gif')





























