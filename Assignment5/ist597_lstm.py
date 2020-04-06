# -*- coding: utf-8 -*-
"""IST597_LSTM.ipynb

# IST597
Routine to create RNN Cells in Tensorflow 2.0 using eager execution.


# Code adapted from Google AI Language Team
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

import matplotlib.pyplot as plt

import os
# openCV-python package
import cv2
# scikit-image package
import skimage
import numpy as np
import random

from sklearn.model_selection import train_test_split

tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

# unzip tar file
import tarfile
tf = tarfile.open("drive/My Drive/Colab Notebooks/notMNIST_large.tar")
tf.extractall()
tf.close()

train_dir = "notMNIST_large/"


imgs = []
labels = []

imageSize = 28


# load data and labels
def get_data(folder):
    imgs = []
    labels = []  
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder+folderName+'/'+image_filename)
                if img_file is not None:
                    #print(img_file)
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize,1))
                    #print(img_file)
                    #print(label)
                    img_arr = np.asarray(img_file)
                    #print(img_arr.shape)
                    imgs.append(img_arr)
                    labels.append(label)
    X = np.asarray(imgs)
    Y = np.asarray(labels)
    return X,Y

X_train, y_train = get_data(train_dir)

from keras.utils import to_categorical

nb_classes = 10

X_train = X_train.astype('float32') / 255.
X_train = X_train.reshape((-1, 28, 28))  # 28 timesteps, 28 inputs / timestep
# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
#y_test_ohe = tf.one_hot(y_train, depth=n_classes).numpy()
y_test_ohe = to_categorical(y_train, num_classes=nb_classes)

print('X_train:', X_train.shape) #X_train: (529114, 28, 28)
print('y_train:', y_test_ohe.shape) #y_train: (529114, 10)

X_train,X_test, Y_train, Y_test = train_test_split(X_train, y_test_ohe, test_size=0.25)

print('X_train:', X_train.shape) #X_train: (370379, 28, 28)
print('X_test:', X_test.shape) #X_test: (158735, 28, 28)

# LSTM modelt
class BasicLSTM(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(BasicLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states

        def bias_initializer(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # input gate
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),  # forget gate
                tf.keras.initializers.Zeros()((self.units * 2,), *args, **kwargs),  # context and output gates
            ])

        self.kernel = tf.keras.layers.Dense(4 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(4 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer)

    def call(self, inputs, training=None, mask=None, initial_states=None):
        # LSTM Cell in pure TF Eager code
        # reset the states initially if not provided, else use those
        if initial_states is None:
            #h_state = tf.zeros((inputs.shape[0], self.units))
            #c_state = tf.zeros((inputs.shape[0], self.units))
            h_state = tf.zeros((1, self.units))
            c_state = tf.zeros((1, self.units))

        else:
            assert len(initial_states) == 2, "Must pass a list of 2 states when passing 'initial_states'"
            h_state, c_state = initial_states

        h_list = []
        c_list = []

        for t in range(inputs.shape[1]):
            # LSTM gate steps
            ip = inputs[:, t, :]
            z = self.kernel(ip)
            z += self.recurrent_kernel(h_state)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            # gate updates
            i = tf.keras.activations.sigmoid(z0)
            f = tf.keras.activations.sigmoid(z1)
            c = f * c_state + i * tf.nn.tanh(z2)

            # state updates
            o = tf.keras.activations.sigmoid(z3)
            h = o * tf.nn.tanh(c)

            h_state = h
            c_state = c

            h_list.append(h_state)
            c_list.append(c_state)

        hidden_outputs = tf.stack(h_list, axis=1)
        hidden_states = tf.stack(c_list, axis=1)

        if self.return_states and self.return_sequence:
            return hidden_outputs, [hidden_outputs, hidden_states]
        elif self.return_states and not self.return_sequence:
            return hidden_outputs[:, -1, :], [h_state, c_state]
        elif self.return_sequence and not self.return_states:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]

class BasicLSTMModel(tf.keras.Model):
    def __init__(self, units, num_classes):
        super(BasicLSTMModel, self).__init__()
        self.units = units
        self.lstm = BasicLSTM(units)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        h = self.lstm(inputs)
        output = self.classifier(h)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output
    
    
# GRU model
class GRU(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(GRU, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states
 
        def bias_initializer(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # update gate
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),  # reset gate
            ])
        def bias_initializer1(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # update gate
    
            ])
        self.kernel = tf.keras.layers.Dense(2 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(2 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer)

        self.kernel1 = tf.keras.layers.Dense(1 * units, use_bias=False)
        self.recurrent_kernel1 = tf.keras.layers.Dense(1 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer1)
  
    def call(self, inputs, training=None, mask=None, initial_states=None):
        # LSTM Cell in pure TF Eager code
        # reset the states initially if not provided, else use those
        if initial_states is None:
            h_state = tf.zeros((1, self.units))


        else:
            assert len(initial_states) == 1, "Must pass a list of 2 states when passing 'initial_states'"
            h_state = initial_states

        h_list = []

        for t in range(inputs.shape[1]):
            # LSTM gate steps
            ip = inputs[:, t, :]
            
            z = self.kernel(ip)
            z += self.recurrent_kernel(h_state)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]

            # gate updates
            # update gate
            Z = tf.keras.activations.sigmoid(z0)
            # reset gate
            R = tf.keras.activations.sigmoid(z1)
            
            z2 = self.kernel1(ip)
            z2 += self.recurrent_kernel1(R*h_state)
            S_ = tf.nn.tanh(z2)
       
            # state updates
            # output gate
            # Remember that the hidden state contains information on previous inputs
            S = (1-Z)*h_state + Z*S_

            h_state = S
      
            h_list.append(h_state)

        hidden_outputs = tf.stack(h_list, axis=1)

        if self.return_states and self.return_sequence:
            return hidden_outputs, [hidden_outputs, hidden_states]
        elif self.return_states and not self.return_sequence:
            return hidden_outputs[:, -1, :], [h_state]
        elif self.return_sequence and not self.return_states:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]
        

class GRUModel(tf.keras.Model):
    def __init__(self, units, num_classes):
        super(GRUModel, self).__init__()
        self.units = units
        self.gru = GRU(units)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        h = self.gru(inputs)
        output = self.classifier(h)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output
    
# MGU model
class MGU(tf.keras.Model):
    def __init__(self, units, return_sequence=False, return_states=False, **kwargs):
        super(MGU, self).__init__(**kwargs)
        self.units = units
        self.return_sequence = return_sequence
        self.return_states = return_states
 
        def bias_initializer(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # update gate
            ])
        def bias_initializer1(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # update gate
            ])
        self.kernel = tf.keras.layers.Dense(1 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(1 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer)

        self.kernel1 = tf.keras.layers.Dense(1 * units, use_bias=False)
        self.recurrent_kernel1 = tf.keras.layers.Dense(1 * units, kernel_initializer='glorot_uniform', bias_initializer=bias_initializer1)
  
    def call(self, inputs, training=None, mask=None, initial_states=None):
        # LSTM Cell in pure TF Eager code
        # reset the states initially if not provided, else use those
        if initial_states is None:
            h_state = tf.zeros((1, self.units))


        else:
            assert len(initial_states) == 1, "Must pass a list of 2 states when passing 'initial_states'"
            h_state = initial_states

        h_list = []

        for t in range(inputs.shape[1]):
            # LSTM gate steps
            ip = inputs[:, t, :]
            
            z = self.kernel(ip)
            z += self.recurrent_kernel(h_state)

            z0 = z[:, :self.units]
            #z1 = z[:, self.units: 2 * self.units]

            # gate updates
            # update gate
            f = tf.keras.activations.sigmoid(z0)
            
            z1 = self.kernel1(ip)
            z1 += self.recurrent_kernel1(f*h_state)
            #S_ = tf.nn.tanh(tf.matmul(ip, self.Woutg) +
            #            tf.matmul(R * h_state, self.Uoutg) + self.boutg)
            S_ = tf.nn.tanh(z1)
       
            # state updates
            # output gate
            # Remember that the hidden state contains information on previous inputs
            S = (1-f)*h_state + f*S_

            h_state = S
      
            h_list.append(h_state)

        hidden_outputs = tf.stack(h_list, axis=1)

        if self.return_states and self.return_sequence:
            return hidden_outputs, [hidden_outputs, hidden_states]
        elif self.return_states and not self.return_sequence:
            return hidden_outputs[:, -1, :], [h_state]
        elif self.return_sequence and not self.return_states:
            return hidden_outputs
        else:
            return hidden_outputs[:, -1, :]
        
class MGUModel(tf.keras.Model):
    def __init__(self, units, num_classes):
        super(MGUModel, self).__init__()
        self.units = units
        self.mgu = MGU(units)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        h = self.mgu(inputs)
        output = self.classifier(h)

        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(output)

        return output
    

# train and evaluate
device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device):
    # build model and optimizer
    model = MGUModel(units, num_classes)
    #model = GRUModel(units, num_classes)
    #model = BasicLSTMModel(units, num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # TF Keras tries to use entire dataset to determine shape without this step when using .fit()
    # Fix = Use exactly one sample from the provided input dataset to determine input/output shape/s for the model
    dummy_x = tf.zeros((1, 28, 28))
    model._set_inputs(dummy_x)

    # train
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, Y_test), verbose=1)

    # evaluate on test set
    scores = model.evaluate(X_test, Y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

# plot learning curve
#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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