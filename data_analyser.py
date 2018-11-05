import requests
import json

#parameters = {'begin': 4, 'end': 10}
#response = requests.get("http://128.0.134.242/data", params=parameters)

#print (response.json())

from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import matplotlib.pyplot as plt
import scipy
import numpy as np


batch_size = 1000 # in each iteration, we consider 128 training examples at once
num_epochs = 20 # we iterate twenty times over the entire training set
#hidden_size = 15 # there will be 512 neurons in both hidden layers

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)


#resize_Pic(X);

def get_data_portion(begin, length):
    #return X_train[begin:begin+length], Y_train[begin:begin+length]
    #(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data
    parameters = {'begin': begin, 'end': begin+length}
    response = requests.get("http://128.0.134.242/data", params=parameters)
    parsed_json = json.loads(response.json())
    
    length = parsed_json["length"]
    X_train = parsed_json["data"][0]
    Y_train = parsed_json["data"][1]
    
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32')
    
    
    X_train /= 255 # Normalise data to [0, 1] range
    X_test /= 255 # Normalise data to [0, 1] range
    
    X_train = X_train.reshape(length, height * width) # Flatten data to 1D
    X_test = X_test.reshape(length, height * width) # Flatten data to 1D
    
    X_train = X_train.reshape(length, height * width) # Flatten data to 1D
    X_test = X_test.reshape(length, height * width) # Flatten data to 1D
    
        
    Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

def evaluate_data (X_train, X_test, hidden_size):      
    inp = Input(shape=(height * width,)) # Our input is a 1D vector of size 784
    hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
    hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
    out = Dense(num_classes, activation='softmax')(hidden_2) # Output softmax layer
    
    model = Model(input=inp, output=out) # To define a model, just specify its input and output layers
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy
    
    for i in range(0, num_train, batch_size):    
        print (i)
        X_train_portion, Y_train_portion = get_data_portion (i, batch_size)
    
        model.fit(X_train_portion, Y_train_portion, # Train the model using the training set...
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=0, validation_split=0.1) # ...holding out 10% of the data for validation
    return model.evaluate(X_test, Y_test, verbose=1)[1] # Evaluate the trained model on the test set!

results = []
results_n = []
horizontal = []

for neurons in range(15, 350, 50):
    horizontal.append(neurons)
    results.append(evaluate_data(X_train, X_test, neurons))
    
fig, ax = plt.subplots()
ax.plot(horizontal, results)

ax.set(xlabel='accurasy', ylabel='hidden layes size',
       title='orange - normilized')
ax.grid()

plt.show()