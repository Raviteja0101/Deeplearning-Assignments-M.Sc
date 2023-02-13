# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:29:14 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:11:35 2020
https://blog.aloni.org/posts/backprop-with-tensorflow/
@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from datasets import MNISTDataset


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[0], cmap="Greys_r")
pause(1)

data = MNISTDataset(train_images.reshape([-1, 784]), train_labels, 
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size=128)

#suraj - increased train_steps to 5000
train_steps = 5000
learning_rate = 0.1

#suraj - remove this comments later, just adding for bug tracking
# weights initialized to zeros
# bias matrix should be two dim even though it has only one row as having one dim bias may cause unkown bugs later
# added the dtype as float32 else it would be considered as double type and causes error during multiplication later
W1 = tf.Variable(np.random.rand(784, 100), dtype= tf.float32)
b1 = tf.Variable(np.random.rand(1, 100), dtype= tf.float32)
W2 = tf.Variable(np.random.rand(100, 10) , dtype= tf.float32)
b2 = tf.Variable(np.random.rand(1, 10), dtype= tf.float32)


for step in range(train_steps):
    img_batch, lbl_batch = data.next_batch()
    
    with tf.GradientTape() as tape:
        
        #suraj -  corrected forward propagation function
        #suraj - renaming op1 to "z"  and activations to "a" according to neural net standards
        z1 = tf.matmul(img_batch, W1) + b1
        #suraj - bugs in forming neural network
        """
        neural net structure with one hidden layer
        img_batch  ==> (W1, b1) ==> z1 ==> (activation) ==> a1 ==> (W2, b2) ==> z2 also called logits ===> softmax(if required)
        """
        #suraj - output z1 is passed trough activation such as sigmoid, tanh and it is not logits. logits is the unnormalized output of the final output layer
        
        a1 = tf.nn.tanh(z1);
        #suraj - bug logits should be calculated from a1 and not from img_batch
        logits = tf.matmul(a1, W2) + b2

        # why was this required? zlogits = logits.numpy()
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=lbl_batch))
        zxent = xent.numpy()

    # suraj - tape was used twice when it can be use only one
    #suraj - get all parameters from the same tape and not just w2 and b2       
        grads = tape.gradient(xent, [W2, b2, W1, b1])

    #suraj - grads list will contain partial derivatives representing errors in order of which it is presented to tape
    # [dW2, db2, dW1, db1]

    # suraj - changing the order of grads list assignment
    W2.assign_sub(learning_rate * grads[0])
    b2.assign_sub(learning_rate * grads[1])
    W1.assign_sub(learning_rate * grads[2])
    b1.assign_sub(learning_rate * grads[3])
        
    if not step % 100:
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                             tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))

#suraj - added the test forward prop for hidden layer
z1_test = tf.matmul(data.test_data, W1) + b1;
a1_test = tf.nn.tanh(z1_test);
logits_test = tf.matmul(a1_test, W2) + b2;
test_preds = tf.argmax(logits_test, axis=1,output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
print("Test accuracy: " + str(acc))