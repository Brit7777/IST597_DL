"""
author:-aam35
"""
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tfe.enable_eager_execution()

# random seed to get the consistent result
tf.random.set_random_seed(6)

# Create data
NUM_EXAMPLES = 500

#define inputs and outputs with some noise 
X = tf.random_normal([NUM_EXAMPLES])  #inputs 
noise = tf.random_normal([NUM_EXAMPLES]) #noise 
y = X * 3 + 2 + noise  #true output

# Create variables.
W = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float32))
b = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float32))


train_steps = 1000
learning_rate = 0.001

# Define the linear predictor.
def prediction(x):
    y_predicted = tf.add(tf.multiply(X,W),b)
    return y_predicted

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    loss = tf.reduce_mean(tf.square(y-y_predicted))
    return loss

def huber_loss(y, y_predicted, m=1.0):
    error = y-y_predicted
    loss = tf.Variable(0.0)
    for i in error:
        if tf.abs(i)<=m:
            loss = tf.add(loss,0.5*tf.square(i))
        else:
            loss = tf.add(loss,0.5*tf.square(i)+m*(tf.abs(i)-m))
    return loss//NUM_EXAMPLES


for i in range(train_steps+1):
  #watch the gradient flow 
    with tf.GradientTape() as tape:
    
        #get prediction
        y_predicted = prediction(X)
    
        #calcuate the loss (difference squared error)
        #loss = squared_loss(y,y_predicted)
        # caculate the loss (huber loss)
        loss = huber_loss(y,y_predicted)
        # calculate hybrid loss
        loss = squared_loss(y,y_predicted) + huber_loss(y,y_predicted)
  
    #evaluate the gradient with the respect to the paramters
    dW, db = tape.gradient(loss, [W, b])

    #update the paramters using Gradient Descent  
    W.assign_sub(dW * learning_rate)
    b.assign_sub(db* learning_rate)

    #print the loss every 20 iterations 
    if i % 100 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss))
      
    
# print the result
print(f'W : {W.numpy()} , b  = {b.numpy()} ')
plt.plot(X, y, 'bo',label='org')
plt.plot(X, X * W.numpy() + b.numpy(), 'r',
         label="huber regression")
plt.legend()
plt.show