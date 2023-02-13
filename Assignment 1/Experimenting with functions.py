
"""
Created on Fri Apr 24 17:11:35 2020
https://blog.aloni.org/posts/backprop-with-tensorflow/
@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

data = MNISTDataset(train_images.reshape([-1, 784]), train_labels, 
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size=128)

def experimentOne():
    training_steps = 1000
    lr = 0.05
    W1 = tf.Variable(np.random.rand(784, 10), dtype= tf.float32)
    b1 = tf.Variable(np.random.rand(1, 10), dtype= tf.float32)
    train_stats = {}
    train_stats["acc"] = []
    train_stats["steps"] = []
    train_stats["loss"] = []
    for step in range(training_steps):
        img_batch, lbl_batch = data.next_batch()
        with tf.GradientTape() as tape:
            logits = tf.matmul(img_batch, W1) + b1
            error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = lbl_batch, logits = logits))
  
        grads1 = tape.gradient(error, [W1, b1])
        W1.assign_sub(lr * grads1[0])
        b1.assign_sub(lr * grads1[1])

        if(step % 100 == 0):
            preds = tf.argmax(logits, axis = 1, output_type = tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch), tf.float32))
            print("Loss: {} and Accuracy: {}".format(error, accuracy))
            train_stats["acc"].append(accuracy.numpy())
            train_stats["steps"].append(step)  
            train_stats["loss"].append(error.numpy())
    test_preds = tf.argmax(tf.matmul(data.test_data, W1) + b1, axis=1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
    print(acc)
    return train_stats

def experimentTwo():
#  Added another layer with Relu activation
    train_steps = 1000
    learning_rate = 0.05
    train_stats = {}
    train_stats["acc"] = []
    train_stats["steps"] = []
    train_stats["loss"] = []
    W1 = tf.Variable(np.random.rand(784, 100), dtype= tf.float32)
    b1 = tf.Variable(np.random.rand(1, 100), dtype= tf.float32)
    W2 = tf.Variable(np.random.rand(100, 10) , dtype= tf.float32)
    b2 = tf.Variable(np.random.rand(1, 10), dtype= tf.float32)
    
    
    for step in range(train_steps):
        img_batch, lbl_batch = data.next_batch()
        
        with tf.GradientTape() as tape:

            z1 = tf.add(tf.matmul(img_batch, W1), b1)
            a1 = tf.nn.relu(z1)
            
            # Do not change the activation for the last layer
            logits = tf.add(tf.matmul(a1, W2), b2)
            a2 = tf.nn.softmax(logits)
            error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl_batch))
    
        grads = tape.gradient(error, [W1, b1, W2, b2])
        W1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        W2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
            
        if not step % 100:
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                                 tf.float32))
            print("Loss: {} Accuracy: {}".format(error, accuracy))
            train_stats["acc"].append(accuracy.numpy())
            train_stats["steps"].append(step)  
            train_stats["loss"].append(error.numpy())
    
    z1_test = tf.matmul(data.test_data, W1) + b1
    a1_test = tf.nn.relu(z1_test)

    logits_test = tf.matmul(a1_test, W2) + b2
    test_preds = tf.argmax(logits_test, axis=1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
    print("Test accuracy: " + str(acc))
    
    predBools = tf.equal(test_preds, data.test_labels).numpy()
    incorrectImages = np.array(data.test_data[np.argwhere(predBools == False)])
    incorrectLabels = np.array(test_preds.numpy()[np.argwhere(predBools == False)])

    
    return train_stats, incorrectImages, incorrectLabels

def experimentFour():

    train_steps = 1000
    learning_rate = 0.05
    train_stats = {}
    train_stats["acc"] = []
    train_stats["steps"] = []
    train_stats["loss"] = []
    W1 = tf.Variable(np.random.rand(784, 100)*0, dtype= tf.float32)
    b1 = tf.Variable(np.random.rand(1, 100)*0, dtype= tf.float32)
    W2 = tf.Variable(np.random.rand(100, 10)*0, dtype= tf.float32)
    b2 = tf.Variable(np.random.rand(1, 10)*0, dtype= tf.float32)

#    W1 = tf.constant(value=0, shape=(784,100), dtype= tf.float32)
#    W1 = tf.convert_to_tensor(np.zeros([784,100]))
    for step in range(train_steps):
        img_batch, lbl_batch = data.next_batch()
        
        with tf.GradientTape() as tape:

            z1 = tf.add(tf.matmul(img_batch, W1), b1)
            a1 = tf.nn.relu(z1)
            
            # Do not change the activation for the last layer
            logits = tf.add(tf.matmul(a1, W2), b2)
            a2 = tf.nn.softmax(logits)
            error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl_batch))
    
        grads = tape.gradient(error, [W1, b1, W2, b2])
        W1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        W2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
            
        if not step % 100:
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                                 tf.float32))
            print("Loss: {} Accuracy: {}".format(error, accuracy))
            train_stats["acc"].append(accuracy.numpy())
            train_stats["steps"].append(step)  
            train_stats["loss"].append(error.numpy())
    
    z1_test = tf.matmul(data.test_data, W1) + b1
    a1_test = tf.nn.relu(z1_test)

    logits_test = tf.matmul(a1_test, W2) + b2
    test_preds = tf.argmax(logits_test, axis=1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
    print("Test accuracy: " + str(acc))
    
    
    return train_stats

def experimentThree():

    train_steps = 1000
    learning_rate = 0.01
    train_stats = {}
    train_stats["acc"] = []
    train_stats["steps"] = []
    train_stats["loss"] = []
    
    W1 = tf.Variable(np.random.rand(784, 200), dtype= tf.float32)
    b1 = tf.Variable(np.random.rand(1, 200), dtype= tf.float32)
    W2 = tf.Variable(np.random.rand(200, 50) , dtype= tf.float32)
    b2 = tf.Variable(np.random.rand(1, 50), dtype= tf.float32)
    W3 = tf.Variable(np.random.rand(50, 10) , dtype= tf.float32)
    b3 = tf.Variable(np.random.rand(1, 10), dtype= tf.float32)    
    
    for step in range(train_steps):
        img_batch, lbl_batch = data.next_batch()
        
        with tf.GradientTape() as tape:

            z1 = tf.add(tf.matmul(img_batch, W1), b1)
            a1 = tf.nn.relu(z1)

            z2 = tf.add(tf.matmul(a1, W2), b2)
            a2 = tf.nn.relu(z2)

            # Do not change the activation for the last layer
            logits = tf.add(tf.matmul(a2, W3), b3)
            a3 = tf.nn.softmax(logits)
            error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl_batch))
    
        grads = tape.gradient(error, [W1, b1, W2, b2, W3, b3])
        W1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        W2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        W3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])        
            
        if not step % 100:
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                                 tf.float32))
            print("Loss: {} Accuracy: {}".format(error, accuracy))
            train_stats["acc"].append(accuracy.numpy())
            train_stats["steps"].append(step)  
            train_stats["loss"].append(error.numpy())            
    
    z1_test = tf.matmul(data.test_data, W1) + b1
    a1_test = tf.nn.relu(z1_test)

    z2_test = tf.matmul(a1_test, W2) + b2
    a2_test = tf.nn.relu(z2_test)
    
    logits_test = tf.matmul(a2_test, W3) + b3
    test_preds = tf.argmax(logits_test, axis=1,output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
    print("Test accuracy: " + str(acc))
    
    
    return train_stats


def main():
    train_stats1 = experimentOne()
#    %matplotlib inline in notebook
    plt.plot(train_stats1["steps"], train_stats1["acc"], label='A1')
    plt.plot(train_stats1["steps"], train_stats1["loss"], label='L1')
    plt.title("Experiment One Stats")
#    plt.show()
    
    train_stats2 = experimentTwo()
    plt.plot(train_stats2["steps"], train_stats2["acc"], label='A2')
    plt.plot(train_stats2["steps"], train_stats2["loss"], label='L2')
    plt.title("Experiment One Stats")
#    plt.show()
    train_stats3 = experimentThree()
    plt.plot(train_stats3["steps"], train_stats3["acc"], label='A3')
    plt.plot(train_stats3["steps"], train_stats3["loss"], label='L3')
    plt.title("Experiment One Stats")
#    plt.show()    
    train_stats4 = experimentFour()
    plt.plot(train_stats4["steps"], train_stats4["acc"], label='A4')
    plt.plot(train_stats4["steps"], train_stats4["loss"], label='L4')
    plt.title("Experiment One Stats")
    plt.show()
#main()



train_stats2,incorrectImages,incorrectLabels  = experimentTwo()
plt.plot(train_stats2["steps"], train_stats2["acc"], label='A2')
plt.plot(train_stats2["steps"], train_stats2["loss"], label='L2')
plt.title("Experiment Two Stats")
plt.legend()
plt.show()
plt.clf()



columns = 3
rows = 1
fig=plt.figure(figsize=(8, 8))

for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    rand = np.random.randint(0,incorrectImages.shape[0])
    img = incorrectImages[rand]
    label = incorrectLabels[rand]
    img=img.reshape(28,28)    
    plt.imshow(img)
    title = "Predicted: " + str(label)
    plt.title(title)

plt.show()