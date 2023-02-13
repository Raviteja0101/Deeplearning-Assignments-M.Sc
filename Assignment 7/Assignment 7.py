# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:02:32 2020

@author: Khamar Uz Zama
"""

import tensorflow as tf			


"""
completed
Task 1
Given a 2D tensor of shape (?, n), extract the k (k <= n) highest values for each row
into a tensor of shape (?, k). 
Hint: There might be a function to get the “top k” values of a tensor.
"""


twoDTensor = tf.constant([ [1, 2, 3],
                          [5, 4, 9] ],tf.int16)
maxTensor = tf.math.top_k(twoDTensor, k=1, sorted=True)

print("Top K values")
maxTensor.values

print("Indices of top K values")
maxTensor.indices


"""
completed
Task 2
Given a tensor of shape (?, n), find the argmax in each row and return a new tensor 
that contains a 1 in each of the argmax’ positions, and 0s everywhere else.
"""

#twoDTensor = tf.constant([ [1, 2, 3, 23],
#                          [5, 40, 9, 1] ],tf.int16)
#
#maxTensor = tf.math.top_k(twoDTensor, k=1, sorted=True)
#
#oneHotArgMax = tf.one_hot(maxTensor.indices, depth = twoDTensor.shape[1], dtype=tf.int32)
#
#tf.print(oneHotArgMax)

"""
Task 3
As in 1., but instead of “extracting” the top k values, create a new tensor with shape (?, n) 
where all but the top k values for each row are zero. 
Try doing this with a 1D tensor of shape (n,) (i.e. one row) first. 
Getting it right for a 2D tensor is more tricky; consider this a bonus.
Hint: You should look for a way to “scatter” a tensor of values into a different tensor. 
For two or more dimensions, you need to think carefully about the indices.
"""

twoDTensor = tf.constant([ [1, 2, 3],
                          [5, 40, 9] ],tf.int16)
maxTensor = tf.math.top_k(twoDTensor, k=1, sorted=True)

print("Top K values")
maxTensor.values

print("Indices of top K values")
indices = maxTensor.indices.numpy()
indices
indices[1][0]

zzz = tf.split(
    twoDTensor, 2, axis=0, num=None, name='split'
)

update = tf.zeros(shape=[1,twoDTensor.shape[1]])

for index, temp in enumerate(zzz):
    tf.tensor_scatter_nd_update(temp, indices[index], update)

"""
completed
Task 4
Implement an exponential moving average. That is, given a decay rate a 
and an input tensor of length T,
create a new length T tensor where 
new[0] = input[0] and new[t] = a * new[t-1] + (1-a) * input[t] otherwise. 
Do not use tf.train.ExponentialMovingAverage.
"""

oneDTensor = tf.constant([1, 2, 3, 23], dtype = tf.float32)
decay = tf.constant(0.1)
ta = tf.TensorArray(tf.dtypes.float32, size = 0, dynamic_size=True, clear_after_read=False)

for index,x in enumerate(oneDTensor):
    if (index == 0):
        ta = ta.write(index, x)
    else :
        s = decay * ta.read(index-1)
        a = (1-decay) * x
        print(s+a)
        ta = ta.write(index, s+a)
    print(ta.stack())

ta.stack()
    

#tf.map_fn(lambda index, value : value[0] if (index == 0) else decay * value[index-1] + (1-decay) * input[index],
#          oneDTensor)

"""
incomplete
Task 5
Find a way to return the last element in 4. without using loops. 
That is, return new[T] only – 
you don’t need to compute the other time steps (if you can avoid it).
"""

"""
Task 6
Given three integer tensors x, y, z all of the same (arbitrary) shape, 
create a new tensor that takes values from y where x is even and from z where x is odd.
"""


X = tf.constant([1, 2, 3, 4], dtype = tf.float32)
Y = tf.constant([11, 12, 13, 14])
Z = tf.constant([21, 22, 23, 24])

#asd = tf.map_fn(lambda y if(x%2 == 0) else z, for x,y,z in zip(X,Y,Z))

zz = [lambda x,y,z: y if(x%2 == 0) else z for x,y,z in zip(X,Y,Z)]


"""
completed
Task 7: Given a tensor of arbitrary and unknown shape 
(but at least one dimension), return 100 if the last dimension has size > 100,
 12 if the last dimension has size <= 100 and > 44, and return 0 otherwise.
"""


def task7(inp):
  lastDimension = inp.shape[-1]
  print(lastDimension)
  if lastDimension > 100:
    return 100
  elif (lastDimension in range(44,101)):
    return 12
  else:
    return 0

tensor = tf.zeros([13,220,344])

task7(tensor)






"""
completed
Task 8: As 7., but also create three global counts (integers),
where count i should grow by 1 if condition i happened. 
Run the function from 7. multiple times to test whether your counting works. 
Now, add a @tf.function decorator to the function from 7.
Does your counter still work? If not, why not? Can you change it so it does work?
"""

counter1 = tf.Variable(0)
counter2 = tf.Variable(0)
counter3 = tf.Variable(0)
tensor = tf.zeros([13,220,45])

@tf.function
def task8(inp):
  last_dim = inp.shape[-1]
  if last_dim > 100:
    counter1.assign_add(1)
    return 100
  elif (last_dim in range(44,101)):
    counter2.assign_add(1)
    return 12
  else:
    counter3.assign_add(1)
    return 0

tensor = tf.zeros([13,220,45])
task8(tensor)

tf.print(counter1)
tf.print(counter2)
tf.print(counter3)

tensor = tf.zeros([13,220,450])
task8(tensor)

tf.print(counter1)
tf.print(counter2)
tf.print(counter3)

# Didnt encounter any error
# Reference:https://www.tensorflow.org/api_docs/python/tf/Variable
# https://www.tensorflow.org/api_docs/python/tf/print


"""
Task 9: Given two 1D tensors of equal length n, create a tensor of shape (n, n) where element i,j 
is the ith element of the first tensor minus the jth element of the second tensor.
No loops! Hint: Tensorflow supports broadcasting much like numpy.
"""


@tf.function
def task9(x, y):
  s = x.shape[0]
  temp = tf.repeat(x, repeats=s, axis=0)

  temp = tf.reshape(temp, [s,s])

  xMinusy = tf.subtract(temp,y)

  return xMinusy


tensor1 = tf.random.uniform([3],minval=1, maxval=5, dtype=tf.dtypes.int32)
tensor2 = tf.random.uniform([3],minval=1, maxval=5, dtype=tf.dtypes.int32)

tf.print(task9(tensor1,tensor2))



