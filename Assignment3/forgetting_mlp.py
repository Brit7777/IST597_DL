# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time
tf.enable_eager_execution()
tf.executing_eagerly()

# random seed to get the consistent result
tf.random.set_random_seed(42)

## Permuted MNIST
## a training set of 55,000 examples, validation is 5000, and a test set of 10,000 examples
data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)


## parameters
num_tasks_to_run = 10
num_epochs_per_task = 20
minibatch_size = 64
learning_rate = 0.0001
num_train = len(data.train.labels)
num_test = len(data.test.labels)

# Generate the tasks specifications as a list of random permutations of the input pixels.
#  permuting the pixels in all images with the same permutation
# for training
train_permutation = []
# for validation
validation_permutation = []
# for test
test_permutation = []
# permutation
permutation = []
for task in range(num_tasks_to_run):
    # 28*28 pixels
    permutation.append(np.random.RandomState(seed=task*(42)).permutation(784))
    # permute dataset
    train_permutation.append(data.train.images[:,permutation[task]])
    validation_permutation.append(data.validation.images[:,permutation[task]])
    test_permutation.append(data.test.images[:,permutation[task]])


#Based on tutorial provided create your MLP model for above problem
#For TF2.0 users Keras can be used for loading trainable variables and dataset.

## ####################model 1 with 2 hidden layers##############################################
size_input = 784 # MNIST data input (img shape: 28*28)
size_hidden_1 = 256
size_hidden_2 = 256
size_output = 10 # MNIST total classes (0-9 digits)


# Define class to build mlp model
class MLP1(object):
    def __init__(self, size_input, size_hidden_1, size_hidden_2, size_output, device=None):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.size_input, self.size_hidden_1, self.size_hidden_2, self.size_output, self.device =\
        size_input, size_hidden_1, size_hidden_2, size_output, device
    
        # Initialize weights between input layer and hidden layer1
        self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
        # Initialize biases for hidden layer
        self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
        # Initialize weights between input layer and hidden layer
        self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
        # Initialize biases for hidden layer
        self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
        # Initialize weights between hidden layer and output layer
        self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_output]))
        # Initialize biases for output layer
        self.b3 = tf.Variable(tf.random_normal([1, self.size_output]))
        

    
        # Define variables to be updated during backpropagation
        self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        # Test with SGD,Adam, RMSProp
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        
    
    # prediction
    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
      
        return self.y
    
    
    ## loss function
    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        # for L1
        #loss = y_true_tf - y_pred_tf
        #L1 = tf.reduce_mean(tf.abs(loss))
        # for L2
        #L2 = tf.losses.mean_squared_error(y_true_tf, y_pred_tf)
        # for L1 + L2
        #return L1 + L2
        # for NLL
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_true_tf))


        
  
    def backward(self, X_train, y_train):
        """
        backward pass
        """  
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # Remember to normalize your dataset before moving forward
        # Compute values in hidden layer
        what1 = tf.matmul(X_tf, self.W1) + self.b1
        hhat1 = tf.nn.relu(what1)
        # dropout
        d_hhat = tf.nn.dropout(hhat1, 0.5)
        # Compute output
        what2 = tf.matmul(hhat1, self.W2) + self.b2
        hhat2 = tf.nn.relu(what2)
        #d_hhat = tf.nn.dropout(hhat1, 0.3)
        output = tf.matmul(hhat2, self.W3) + self.b3
        #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
        #Second add tf.Softmax(output) and then return this variable
        return output      


################################## model 2 with 3 hidden layers#########################################
#uncomment this part when you want to use this model
# size_input = 784 # MNIST data input (img shape: 28*28)
#size_hidden_1 = 256
#size_hidden_2 = 256
#size_hidden_3 = 256
#size_output = 10 # MNIST total classes (0-9 digits)


# Define class to build mlp model
class MLP2(object):
    def __init__(self, size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_output, device=None):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.size_input, self.size_hidden_1, self.size_hidden_2, self.size_hidden_3, self.size_output, self.device =\
        size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_output, device
    
        # Initialize weights between input layer and hidden layer1
        self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
        # Initialize biases for hidden layer
        self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
        # Initialize weights between input layer and hidden layer
        self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
        # Initialize biases for hidden layer
        self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
        # Initialize weights between hidden layer and output layer
        self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_hidden_3]))
        # Initialize biases for hidden layer
        self.b3 = tf.Variable(tf.random_normal([1, self.size_hidden_3]))
        # Initialize weights between hidden layer and output layer
        self.W4 = tf.Variable(tf.random_normal([self.size_hidden_3, self.size_output]))
        # Initialize biases for output layer
        self.b4 = tf.Variable(tf.random_normal([1, self.size_output]))
        

    
        # Define variables to be updated during backpropagation
        self.variables = [self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4]
        # Test with SGD,Adam, RMSProp
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        
    
    # prediction
    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
      
        return self.y
    
    
    ## loss function
    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        # for L1
        #loss = y_true_tf - y_pred_tf
        #L1 = tf.reduce_mean(tf.abs(loss))
        # for L2
        #L2 = tf.losses.mean_squared_error(y_true_tf, y_pred_tf)
        # for L1 + L2
        #return L1 + L2
        # for NLL
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_true_tf))

        
  
    def backward(self, X_train, y_train):
        """
        backward pass
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # Remember to normalize your dataset before moving forward
        # Compute values in hidden layer
        what1 = tf.matmul(X_tf, self.W1) + self.b1
        hhat1 = tf.nn.relu(what1)
        # dropout
        d_hhat = tf.nn.dropout(hhat1, 0.5)
        # Compute values in hidden layer
        what2 = tf.matmul(hhat1, self.W2) + self.b2
        hhat2 = tf.nn.relu(what2)
        
        #Compute values in hidden layer
        what3 = tf.matmul(hhat2, self.W3) + self.b3
        hhat3 = tf.nn.relu(what3)
        # Compute output
        output = tf.matmul(hhat3, self.W4) + self.b4
        #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
        #Second add tf.Softmax(output) and then return this variable
        return output       

########################## model 3 with 4 hidden layers##################################################
#uncomment this part if you want to use this model
# size_input = 784 # MNIST data input (img shape: 28*28)
#size_hidden_1 = 256
#size_hidden_2 = 256
#size_hidden_3 = 256
#size_hidden_4 = 256
#size_output = 10 # MNIST total classes (0-9 digits)


# Define class to build mlp model
class MLP3(object):
    def __init__(self, size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_hidden_4, size_output, device=None):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.size_input, self.size_hidden_1, self.size_hidden_2, self.size_hidden_3, self.size_hidden_4, self.size_output, self.device =\
        size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_hidden_4, size_output, device
    
        # Initialize weights between input layer and hidden layer1
        self.W1 = tf.Variable(tf.random_normal([self.size_input, self.size_hidden_1]))
        # Initialize biases for hidden layer
        self.b1 = tf.Variable(tf.random_normal([1, self.size_hidden_1]))
        # Initialize weights between input layer and hidden layer
        self.W2 = tf.Variable(tf.random_normal([self.size_hidden_1, self.size_hidden_2]))
        # Initialize biases for hidden layer
        self.b2 = tf.Variable(tf.random_normal([1, self.size_hidden_2]))
        # Initialize weights between hidden layer and output layer
        self.W3 = tf.Variable(tf.random_normal([self.size_hidden_2, self.size_hidden_3]))
        # Initialize biases for hidden layer
        self.b3 = tf.Variable(tf.random_normal([1, self.size_hidden_3]))
        # Initialize weights between hidden layer and output layer
        self.W4 = tf.Variable(tf.random_normal([self.size_hidden_3, self.size_hidden_4]))
        # Initialize biases for hidden layer
        self.b4 = tf.Variable(tf.random_normal([1, self.size_hidden_4]))
        # Initialize weights between hidden layer and output layer
        self.W5 = tf.Variable(tf.random_normal([self.size_hidden_4, self.size_output]))
        # Initialize biases for output layer
        self.b5 = tf.Variable(tf.random_normal([1, self.size_output]))
        

    
        # Define variables to be updated during backpropagation
        self.variables = [self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2, self.b3, self.b4, self.b5]
        # Test with SGD,Adam, RMSProp
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    
    # prediction
    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
      
        return self.y
    
    
    ## loss function
    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        # for L1
        #loss = y_true_tf - y_pred_tf
        #L1 = tf.reduce_mean(tf.abs(loss))
        # for L2
        #L2 = tf.losses.mean_squared_error(y_true_tf, y_pred_tf)
        # for L1 + L2
        #return L1 + L2
        # for NLL
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_true_tf))
        
  
    def backward(self, X_train, y_train):
        """
        backward pass
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables),
                              global_step=tf.train.get_or_create_global_step())
        
        
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # Remember to normalize your dataset before moving forward
        # Compute values in hidden layer
        what1 = tf.matmul(X_tf, self.W1) + self.b1
        hhat1 = tf.nn.relu(what1)
        # dropout
        d_hhat = tf.nn.dropout(hhat1, 0.5)
        # Compute values in hidden layer
        what2 = tf.matmul(hhat1, self.W2) + self.b2
        hhat2 = tf.nn.relu(what2)
        
        #Compute values in hidden layer
        what3 = tf.matmul(hhat2, self.W3) + self.b3
        hhat3 = tf.nn.relu(what3)
              
        #Compute values in hidden layer
        what4 = tf.matmul(hhat3, self.W4) + self.b4
        hhat4 = tf.nn.relu(what4)
        # Compute output
        output = tf.matmul(hhat4, self.W5) + self.b5
        #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
        #Second add tf.Softmax(output) and then return this variable
        return output       


 # Initialize model using CPU
mlp_on_cpu = MLP1(size_input, size_hidden_1, size_hidden_2, size_output, device='cpu')
#mlp_on_cpu = MLP2(size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_output, device='cpu')
#mlp_on_cpu = MLP3(size_input, size_hidden_1, size_hidden_2, size_hidden_3, size_hidden_4, size_output, device='cpu')

time_start = time.time()
Ptest_dataset_images = []
Ptest_dataset_labels = []
validation_accuracy = []
test_accuracy = []

# bwt function
def get_bwt(accuracy, task):
  bwt = 0.0
  if task == 0 :
    return tf.Variable(0.0)
  # for more than 2 tasks
  for i in range(task):
    bwt += accuracy[task] - accuracy[i]
    
  result = tf.reduce_mean(tf.cast(bwt, tf.float32))
  return result
  
# plot validation results
def plot_val(accuracy):
  ## Plotting chart of training and testing accuracy as a function of iterations
  task = list(range(1,num_tasks_to_run+1))
  plt.xlabel('task')
  plt.ylabel('Validation Accuracy')
  plt.xticks(task)
  plt.plot(task,validation_accuracy)
  plt.show()



# training process
for run in range(num_tasks_to_run):
    # train for 50 epochs for task1
    if run == 0:
        num_epochs_per_task = 50
    else :
        num_epochs_per_task = 20
    for epoch in range(num_epochs_per_task):
        train_ds = tf.data.Dataset.from_tensor_slices((train_permutation[run], data.train.labels)).map(lambda x, y: (x, tf.cast(y, tf.float32)))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=minibatch_size)
        loss_total = tf.Variable(0, dtype=tf.float32)
        for inputs, outputs in train_ds:
            preds = mlp_on_cpu.forward(inputs)
            loss_total = loss_total + mlp_on_cpu.loss(preds, outputs)
            mlp_on_cpu.backward(inputs, outputs)
        # print loss
        print('Number of Epoch = {} - loss:= {:.4f}'.format(epoch + 1, loss_total.numpy() / num_train))
        # train accuracy
        preds = mlp_on_cpu.compute_output(train_permutation[run])
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(data.train.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print ("Training Accuracy = {}".format(accuracy.numpy()))
        
    # validation accuracy
    preds = mlp_on_cpu.compute_output(validation_permutation[run])
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(data.validation.labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    validation_accuracy.append(accuracy.numpy())
 
    # accumulate test dataset
    Ptest_dataset_images.extend(test_permutation[run])
    Ptest_dataset_labels.extend(data.test.labels)
    
    # test accuracy
    preds = mlp_on_cpu.compute_output(Ptest_dataset_images)
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(Ptest_dataset_labels, 1))
    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # To save test accuracy 
    test_accuracy.append(accuracy.numpy())
    print ("Test Accuracy = {}".format(accuracy.numpy()))
    
    # calculate btw
    bwt = get_bwt(test_accuracy,run)
    print ("BWT = {}".format(bwt.numpy()))


# plot validation results
plot_val(validation_accuracy)
time_taken = time.time() - time_start
print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
#For per epoch_time = Total_Time / Number_of_epochs 