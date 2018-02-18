from pylab import *
import numpy as np
from numpy import random as rd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import matplotlib.cbook as cbook

from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
from scipy.io import loadmat

import os
import time
import urllib as url

try:
   import cPickle as pickle
except:
   import pickle



os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')



def plot_samples(M, filename='resources/part1.jpg'): #Part 1
    '''Given a dictionary M formatted as ___
    save an image to filename with 10 random images for each digit '''
    
    rd.seed(0)
    fig, ax = plt.subplots(10,10)

    for i in range(10):
        key = 'train' + str(i)
        length = len( M[key] )
        imgs = [ rd.randint(0, length) for i in range(10) ]
        for k in range( len(imgs) ):
            img = reshape( M[key][ imgs[k] ], (28,28) )
            plt.sca(ax[i, k]) #set current axes
            plt.imshow(img, cmap = cm.gray)
            plt.axis('off')
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)    
    
    #plt.show()
    plt.savefig(filename)
    plt.close()
    return


def softmax(y): #Provided by Profs
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def no_hidden_layers(x, W, b): #Part 2
    '''Compute the network'''
    #the first column of W contains the weights for output 1
    L1 = np.dot(W.T, x) + b
    return softmax(L1)


def grad(y_, y, x): #Part 3b
    '''Compute the gradient wrt weights and biases'''
    dC_do = sum(y, 0) - sum(y_, 0) #y is output of softmax
    
    grad_W = np.sum(dC_do * x, 1) #could use np.mean() for smaller values
    grad_b = dC_do
    return grad_W, grad_b

def NLL(y, y_): #Cost func provided by Profs
    #y is output of network, y_ is correct results
    return -sum(y_*log(y))

def NN(x, y_, W, b, rate, max_iter):
    
    iter = 0
    while iter < max_iter:
        y = no_hidden_layers(x, W, b)
        
        grad_W, grad_b = grad(y_, y, x)
        W -= rate*grad_W
        b -= rate*grad_b
        
        iter += 1
    
    return W, b



def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )




if __name__ == "__main__":

    M = loadmat("mnist_all.mat") #Load the MNIST digit data
    
    #Display the 150-th "5" digit from the training set
    if False:
        plt.imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
        plt.show()
        plt.close()
        
    #Part 1
    plot_samples(M, 'resources/part1.jpg')
    
    #Parts 2 and 3 are implemented as functions above

    
    X = { key:M[key]/255.0 for key in M.keys() if key[0] == 't' } #remove extra keys, and normalize
    a = [np.array( X[key] ) for key in X.keys() if key[0:2] == 'tr']
    x = np.concatenate( ( a ), axis=0 ).T
    
    y0 = np.array([[1,0,0,0,0,0,0,0,0,0]]*len(M['train0']) )
    y1 = np.array([[0,1,0,0,0,0,0,0,0,0]]*len(M['train1']) )
    y2 = np.array([[0,0,1,0,0,0,0,0,0,0]]*len(M['train2']) ) 
    y3 = np.array([[0,0,0,1,0,0,0,0,0,0]]*len(M['train3']) )
    y4 = np.array([[0,0,0,0,1,0,0,0,0,0]]*len(M['train4']) )
    y5 = np.array([[0,0,0,0,0,1,0,0,0,0]]*len(M['train5']) )
    y6 = np.array([[0,0,0,0,0,0,1,0,0,0]]*len(M['train6']) )
    y7 = np.array([[0,0,0,0,0,0,0,1,0,0]]*len(M['train7']) )
    y8 = np.array([[0,0,0,0,0,0,0,0,1,0]]*len(M['train8']) )
    y9 = np.array([[0,0,0,0,0,0,0,0,0,1]]*len(M['train9']) )
    y_ = np.concatenate( (y0,y1,y2,y3,y4,y5,y6,y7,y8,y9), axis=0 ).T

    W = rd.rand(784, 10)
    b = rd.rand(10, 1)
    rate = 1e-5
    max_iter = 1000
    W, b = NN(x, y_, W, b, rate, max_iter)









    #Load sample weights for the multilayer neural network
    snapshot = pickle.load(open("snapshot50.pkl", 'rb'), encoding='latin1')
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))
    
    #Load one example from the training set, and run it through the
    #neural network
    x = M["train5"][148:149].T
    L0, L1, output = forward(x, W0, b0, W1, b1)
    #get the index at which the output is the largest
    y = argmax(output)
    
    ################################################################################
    #Code for displaying a feature from the weight matrix mW
    #fig = figure(1)
    #ax = fig.gca()
    #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
    #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    #show()
    ################################################################################
