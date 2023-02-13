# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:36:39 2020

@author: Khamar Uz Zama
"""

from prepare_data import parse_seq
import pickle
import tensorflow as tf
import numpy as np



# this is just a datasets of "bytes" (not understandable)
data = tf.data.TFRecordDataset("skp.tfrecords")

# this maps a parser function that properly interprets the bytes over the dataset
# (with fixed sequence length 200)
# if you change the sequence length in preprocessing you also need to change it here
sequenceLength = 200
data = data.map(lambda x: parse_seq(x, sequenceLength))

# a map from characters to indices
vocab = pickle.load(open("skp_vocab", mode="rb"))
vocab_size = len(vocab)
# inverse mapping: indices to characters
ind_to_ch = {ind: ch for (ch, ind) in vocab.items()}

print(vocab)
print(vocab_size)

batch_size = 64
batch_OneHot_data = data.shuffle(buffer_size = 10 * batch_size).batch(batch_size=batch_size, drop_remainder=True).repeat()
batch_size = tf.convert_to_tensor(batch_size, dtype=None, dtype_hint=None, name=None)
print(batch_size)

def initializer(hunits, vocab_size, sequenceLength):
    
    
    # Input to hidden layer
    W_InpToHidden = tf.Variable(tf.initializers.glorot_uniform()(shape=[vocab_size, hunits]))
    b_InpToHidden = tf.Variable(tf.initializers.glorot_uniform()(shape=[hunits]))
    
    # hidden to hidden layer
    W_HiddenToHidden = tf.Variable(tf.initializers.glorot_uniform()(shape=[hunits,hunits]))

    
    # hidden to output layer
    W_HiddenToOutput = tf.Variable(tf.initializers.glorot_uniform()(shape=[hunits,vocab_size]))
    b_HiddenToOutput = tf.Variable(tf.initializers.glorot_uniform()(shape=[vocab_size]))
    
    trainVariables = [W_InpToHidden, b_InpToHidden, W_HiddenToHidden, W_HiddenToOutput, b_HiddenToOutput]
    return trainVariables

units = 256


opt = tf.optimizers.Adam()
loss_func= tf.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

n_time_steps = 200

#All operations were executed using tensorflow as python operations do not work under tensorflow

# @tf.function
def RNNCell(seq):
  with tf.GradientTape() as tape:
    Hiddenstate = tf.zeros([batch_size, units])
    


    for i in tf.range(n_time_steps-1):
        inp = tf.one_hot(seq[:, i], vocab_size)
        # # print("inp:  ", inp.shape)
        op_InpToHidden = tf.matmul(inp, trainVariables[0])
        # # print("op_InpToHidden :  ", op_InpToHidden.shape) 
        op_HiddenToHidden = tf.matmul(Hiddenstate, trainVariables[2]) + trainVariables[1] 
        # # print("op_HiddenToHidden :  ", op_HiddenToHidden.shape)
        op_FromHidden = op_InpToHidden + op_HiddenToHidden
        # # print("op_FromHidden :  ", op_FromHidden.shape)
        Hiddenstate = tf.nn.tanh(op_FromHidden)
        # print("op_FromHidden :  ", op_FromHidden.shape)

        # Second node
        y = tf.matmul(Hiddenstate, trainVariables[3]) + trainVariables[4]
        
        # print(y)

        labels = seq[:, i+1]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, y)
        loss = tf.reduce_mean(loss)
        acc = accuracy(labels, y)




    grads = tape.gradient(loss, trainVariables)    
    opt.apply_gradients(zip(grads, trainVariables))

    return loss, acc

def trainModel():
    steps = 10000
    
    for step, seqs in enumerate(batch_OneHot_data):
    
        loss, acc= RNNCell(seqs)
    
        if not step % 100:
            print("Step: {} Loss: {} Acc: {}\n".format(step, loss, acc))
    
        if step > steps:
            break
try:
    with open('WandB.pickle', 'rb') as f:
        trainVariables = pickle.load(f)
except:        
    trainVariables = initializer(units, vocab_size, sequenceLength)
    trainModel()
    
    
for value in trainVariables:
    print(value.shape)

def generate(steps):
    state = tf.zeros([1, units])
    seq = [0]
    temp = []
    for step in range(steps):

        inp = tf.one_hot(seq[-1: ], vocab_size)
        # print("inp:  ", inp.shape)
        op_InpToHidden = tf.matmul(inp, trainVariables[0])
        # print("op_InpToHidden :  ", op_InpToHidden.shape) 
        op_HiddenToHidden = tf.matmul(state, trainVariables[2]) + trainVariables[1] 
        # print("op_HiddenToHidden :  ", op_HiddenToHidden.shape)
        op_FromHidden = op_InpToHidden + op_HiddenToHidden
        # print("op_FromHidden :  ", op_FromHidden.shape)
        state = tf.nn.tanh(op_FromHidden)
        # print("op_FromHidden :  ", op_FromHidden.shape)

        # Second node
        op_FromOPLayer = tf.matmul(state, trainVariables[3]) + trainVariables[4]
        y = tf.nn.softmax(op_FromOPLayer)
        
        y = y.numpy()[0]

        # seq.append(y)
        seq.append(np.random.choice(vocab_size, p=y))
        temp.append(np.argmax(y))

    char = [ind_to_ch[ind] for ind in seq]
    argMaxSentence = [ind_to_ch[ind] for ind in temp]
    return "".join(char), argMaxSentence
        
sentence, argMaxSentence = generate(2000)

print(sentence)
print('xxxxx')
print("".join(argMaxSentence))


import pickle

filename = "WandB.pickle"
with open(filename, 'wb') as f:
   pickle.dump(trainVariables,f)

    
    
    

"""#Reference
1. https://github.com/vzhou842/rnn-from-scratch
2. http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
3. https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/
"""