from pylab import *
import numpy as np
from numpy import random as rd
import numpy.linalg as la

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



#os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')



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
    # For part 2, y has the dimensions (10x60000)
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    #return dimension is (10x60000)


def no_hidden_layers(x, W, b): #Part 2
    '''Compute the network'''
    #the first column of W contains the weights for output 1
    # W is (784x10), x is (784x60 000) and b is (10x60 000)
    # So L1 is (10x60000)
    L1 = np.dot(W.T, x) + tile(b, (1, np.shape(x)[1] ) )
    return softmax(L1)


def grad(y_, y, x): #Part 3b
    '''Compute the gradient wrt weights and biases'''
    #y and y_ have dimension (10x60000)
    #x has dimension (784x60 000)
    
    diff = (y - y_) #y is output of softmax
        # diff is p_j - y_j
    
    grad_W = np.dot( x, diff.T )
    grad_b = np.sum( diff, 1)  #could try with np.mean()
    grad_b = np.reshape(10,1)
    
    return  grad_W, grad_b

def NLL(y_, y): #Cost func provided by Profs, Part 3
    #y is output of network, y_ is correct results
    return -np.sum(y_*log(y))

def finite_diff(y_, x, W, h, b):  #Part 3
    '''note that for simplicity of the code, h is a matrix. the placement of its nonzero element determines the 
    direction along which the finite difference is calculated'''
    y2 = no_hidden_layers(x, W+h, b)
    y1 = no_hidden_layers(x, W, b)
    fd = NLL(y_, y2) - NLL(y_, y1) 
    return fd/np.max(h)

def format_y(M, set):           #Part 4
    '''format y_ for the training and test set'''
    y0 = np.array([[1,0,0,0,0,0,0,0,0,0]]*len(M[set+'0']) )
    y1 = np.array([[0,1,0,0,0,0,0,0,0,0]]*len(M[set+'1']) )
    y2 = np.array([[0,0,1,0,0,0,0,0,0,0]]*len(M[set+'2']) ) 
    y3 = np.array([[0,0,0,1,0,0,0,0,0,0]]*len(M[set+'3']) )
    y4 = np.array([[0,0,0,0,1,0,0,0,0,0]]*len(M[set+'4']) )
    y5 = np.array([[0,0,0,0,0,1,0,0,0,0]]*len(M[set+'5']) )
    y6 = np.array([[0,0,0,0,0,0,1,0,0,0]]*len(M[set+'6']) )
    y7 = np.array([[0,0,0,0,0,0,0,1,0,0]]*len(M[set+'7']) )
    y8 = np.array([[0,0,0,0,0,0,0,0,1,0]]*len(M[set+'8']) )
    y9 = np.array([[0,0,0,0,0,0,0,0,0,1]]*len(M[set+'9']) )
    y_ = np.concatenate( (y0,y1,y2,y3,y4,y5,y6,y7,y8,y9), axis=0 ).T 
    return y_

def backprop(x_train, y_train, x_val, y_val, W, b, rate, max_iter, mom=0, filename=''):
    # Part 4
    iter_acc = []
    train_acc = []
    test_acc = []
    
    nu_W = np.zeros( np.shape(W) )
    nu_b = np.zeros( np.shape(b) )
    
    iter = 0
    while iter <= max_iter:
        y = no_hidden_layers(x_train, W, b)
        #prevW = W.copy()        #don't need
        
        grad_W, grad_b = grad(y_train, y, x_train)
        nu_W = mom*nu_W + rate*grad_W
        nu_b = mom*nu_b + rate*grad_b
        W -= nu_W
        b -= nu_b
        
        if iter%50 == 0:
            iter_acc += [iter]
            
            y = no_hidden_layers(x_train, W, b)
            res = check_results(y_train, y)
            #print( 'Train Results: ' + str(res.count(1)) + '/' + str(len(res)) )
            train_acc += [ res.count(1)/len(res) ]
            
            y = no_hidden_layers(x_val, W, b)
            res = check_results(y_val, y)
            #print( 'Test Results: ' + str(res.count(1)) + '/' + str(len(res)) ) 
            test_acc += [ res.count(1)/len(res) ]
        iter += 1
    
    #Part 4.1.1: Plot the learning curves. 
    if filename:
        plt.scatter(iter_acc, train_acc, label='Training Data')
        plt.scatter(iter_acc, test_acc, label='Test Data')
        plt.title('Learing Curve')
        plt.xlabel('number of iterations')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('resources/' + filename)
        #plt.show()
        plt.close()
    
    return W, b

def optimize_params(rates, x_train, y_train, x_val, y_val, W, b, max_iter, mom=0):
    # Part 4
    val_acc = []
    k = 0
    for rate in rates:
        k += 1
        rd.seed(0)  
        W = rd.rand(784, 10)
        b = rd.rand(10, 1)
        W, b = backprop(x_train, y_train, x_val, y_val, W, b, rate, max_iter, mom=0, filename='part4_optimize'+str(k)+'.jpg' )
        y = no_hidden_layers(x_val, W, b) #the network guesses for the validation set
        res = check_results(y_val, y)
        val_acc += [ res.count(1)/len(res) ]
    return val_acc

def check_results(y_, y):            #Part 4
    '''return an array of 0/1's indicating the correctness of NN's output'''
    results = []
    for k in range( len(y[1,:]) ):
        if np.argmax(y_[:,k]) == np.argmax(y[:,k]): 
            results += [1]
        else:
            results += [0]
    return results

def image_W(W, filename):
    # 
    fig, ax = plt.subplots(1,10)
    
    for k in range( np.shape(W)[1] ):
        img = np.reshape(W[:,k], (28,28) )
        plt.sca(ax[k]) #set current axes
        plt.imshow(img, cmap = cm.gray)
        plt.axis('off')
        
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)    
    
    #plt.show()
    plt.savefig('resources/' + filename)
    plt.close()
    return




def tanh_layer(y, W, b):   #Provided by profs
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1): #Provided by profs
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):  #provided by profs
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )




if __name__ == "__main__":  


    M = loadmat("mnist_all.mat") #Load the MNIST digit data

    if False:     #Display the 150-th "5" digit from the training set
        plt.imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
        plt.show()
        plt.close()
        
    '''plot_samples(M, 'resources/part1.jpg')  #Part 1'''
    
    #Parts 2 and 3 are implemented as functions above
    
    


    #Part 4
    
    #Part4.1: Train the neural network you constructed using gradient descent 
    # (without momentum). 
    
    #Setup data needed for training & testing 
    x_train = np.concatenate( ( [np.array( M[key]/255.0 ) for key in M.keys() if key[0:2] == 'tr'] ), axis=0 ).T
    x_testall = np.concatenate( ( [np.array( M[key]/255.0 ) for key in M.keys() if key[0:2] == 'te'] ), axis=0 ).T
    x_test = x_testall[:, 0::2]
    x_val = x_testall[:, 1::2]
    
    y_train = format_y(M, 'train')
    y_testall = format_y(M, 'test')
    y_test = y_testall[:, 0::2]
    y_val = y_testall[:, 1::2]


    #find the best learning rate
    #rates = [1e-3, 1e-4, 5e-5, 1e-5, 1e-6]
    rates = [1e-4, 1e-5]
    max_iter = 1000
    rd.seed(0)  
    W = rd.rand(784, 10)
    b = rd.rand(10, 1)
    #val_acc = optimize_params(rates, x_train, y_train, x_val, y_val, W, b, max_iter)

    rate = 1e-4 #parameters for gradient descent
    max_iter = 1000
    rd.seed(0)  
    W = rd.rand(784, 10)
    b = rd.rand(10, 1)
    # Part 4.1.1: Plot the learning curves. 
    W, b = backprop(x_train, y_train, x_val, y_val, W, b, rate, max_iter, mom=0, filename='part4.jpg')
    y = no_hidden_layers(x_train, W, b)
    res = check_results(y_train, y)
    print( 'Train Results: ' + str(res.count(1)) + '/' + str(len(res)) )
    y = no_hidden_layers(x_val, W, b)
    res = check_results(y_val, y)
    print( 'Val Results: ' + str(res.count(1)) + '/' + str(len(res)) )
    y = no_hidden_layers(x_test, W, b)
    res = check_results(y_test, y)
    print( 'Test Results: ' + str(res.count(1)) + '/' + str(len(res)) )

    #Part 4.2: Display the weights going into each of the output units.
    image_W(W, filename='weights.jpg')
    
    
    
    
    



    if False:
        #Load sample weights for the multilayer neural network (given)
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
