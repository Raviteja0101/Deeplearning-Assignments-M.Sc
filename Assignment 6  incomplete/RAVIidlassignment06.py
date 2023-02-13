#!/usr/bin/env python
# coding: utf-8

# In[13]:


##Loading preprocessed data skp.tfrecords & skp_vocab
from prepare_data import parse_seq
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras import datasets, layers, models

bs = 256
seq_len = 200
# this is just a datasets of "bytes" (not understandable)
data = tf.data.TFRecordDataset("skp.tfrecords")

# this maps a parser function that properly interprets the bytes over the dataset
# (with fixed sequence length 200)
# if you change the sequence length in preprocessing you also need to change it here
data = data.map(lambda x: parse_seq(x, 200))

# a map from characters to indices
vocab = pickle.load(open("skp_vocab", mode="rb"))
vocab_size = len(vocab)
# inverse mapping: indices to characters
ind_to_ch = {ind: ch for (ch, ind) in vocab.items()}

print(vocab)
print(vocab_size)


n_h = 512
w_xh = tf.Variable(tf.initializers.glorot_uniform()([vocab_size, n_h]))
w_hh = tf.Variable(tf.initializers.glorot_uniform()([n_h, n_h]))
b_h = tf.Variable(tf.zeros([n_h]))

w_ho = tf.Variable(tf.initializers.glorot_uniform()([n_h, vocab_size]))
b_o = tf.Variable(tf.zeros([vocab_size]))

all_vars = [w_xh, w_hh, b_h, w_ho, b_o]


steps = 20*35000 // bs
opt = tf.optimizers.Adam()
loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


#@tf.function
def run_rnn_on_seq(seq_batch):
    with tf.GradientTape() as tape:
        state = tf.zeros([tf.shape(seq_batch)[0], n_h])
        total_loss = tf.constant(0.)

        for time_step in tf.range(tf.shape(seq_batch)[1] - 1):
            inp_here = tf.one_hot(seq_batch[:, time_step], vocab_size)
            state = tf.nn.tanh(tf.matmul(inp_here, w_xh) + tf.matmul(state, w_hh) + b_h)
            logits = tf.matmul(state, w_ho) + b_o

            loss_here = loss_fn(seq_batch[:, time_step+1], logits)
            total_loss += loss_here
            
        total_loss /= tf.cast(tf.shape(seq_batch)[1] - 1, tf.float32)
    grads = tape.gradient(total_loss, all_vars)
    
    # this is gradient clipping
    glob_norm = tf.linalg.global_norm(grads)
    grads = [g/glob_norm for g in grads]
    
    opt.apply_gradients(zip(grads, all_vars))

    return total_loss



for step, seqs in enumerate(data):
    seqs.shape
    tf.shape(seqs)
    seqs.numpy
    
    data.shape
    tf.shape(data)
    data[0].numpy
    
    xent_avg = run_rnn_on_seq(seqs)

    if not step % 200:
        print("Step: {} Loss: {}".format(step, xent_avg))
        print()
        

    if step > steps:
        break








































import numpy as np

def sample(n_steps):
    state = tf.zeros([1, n_h])
    gen = [0]

    for step in range(n_steps):
        state = tf.nn.tanh(tf.matmul(tf.one_hot(gen[-1:], depth=vocab_size), w_xh) + tf.matmul(state, w_hh) + b_h)
        probs = tf.nn.softmax(tf.matmul(state, w_ho) + b_o).numpy()[0]
        #gen.append(np.argmax(probs))  # use argmax instead of choice if you want
        gen.append(np.random.choice(vocab_size, p=probs))
    return "".join([ind_to_ch[ind] for ind in gen])
        
agg = sample(2000)


# In[12]:


print(agg)

