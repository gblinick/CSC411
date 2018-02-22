import os
from torch.autograd import Variable
import torch

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt

from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
from scipy.io import loadmat
  

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')


def format_data(acts, trsizes, folders, rd_seed=0):
    '''format data as dictionary for use as with MNIST data'''
    
    data = {}
    
    for act in acts:
        k = acts.index(act)
        folder = folders[k]
        total = [folder+'/'+filename for filename in os.listdir(folder) if filename.startswith(act)] #list of all filenames for given act 
        rd.seed(rd_seed)
        rd.shuffle(total) #randomize order of files
        
        x = np.array([ imread(total[i]).flatten() for i in range( len(total) ) ])/255.0
        
        trsize = trsizes[act]
        trdata = x[:trsize, :] #training set for act
        tedata = x[trsize:trsize+10, :] #testing set for act
        vadata = x[trsize+10:trsize+30, :] #validation set for act
        
        data['train_'+act] = trdata
        data['val_'+act] = vadata
        data['test_'+act] = tedata
        
    return data


def get_set_data(M, acts, set):
    
    batch_xs = np.zeros((0, 32*32))
    batch_y_s = np.zeros( (0, len(acts) ))
    set_k = [set+'_'+act for act in acts]
    
    for k in range( len(acts) ):
        batch_xs = np.vstack((batch_xs, ((np.array(M[set_k[k]])[:])/255.)  ))
        one_hot = np.zeros( len(acts) )
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[set_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def train(train_x, train_y, val_x, val_y, test_x, test_y, params):
    dim_h, rate, no_epochs, iter = params
    dim_x = 32*32
    dim_out = 12
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    
    train_acc = [] #will hold data for learning curve
    val_acc = []
    
    torch.manual_seed(0)
    #torch.manual_seed_all(0)
    
    #set up PyTorch model
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),        )
    
    model[0].weight = torch.nn.Parameter( torch.randn( model[0].weight.size() ) )
    model[0].bias = torch.nn.Parameter( torch.randn( model[0].bias.size() ) )
    
    loss_fn = torch.nn.CrossEntropyLoss()     #set loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=rate) #set learning rate

    for k in range(no_epochs):
        rd.seed(k)
        batches = np.array_split( np.random.permutation(range(train_x.shape[0]))[:] , 6)
        
        for mini_batch in batches:
            x = Variable(torch.from_numpy(train_x[mini_batch]), requires_grad=False).type(dtype_float)
            y_classes = Variable(torch.from_numpy(np.argmax(train_y[mini_batch], 1)), requires_grad=False).type(dtype_long)
            
            for t in range(iter):
                y_pred = model(x)
                loss = loss_fn(y_pred, y_classes)
                
                model.zero_grad()  # Zero out the previous gradient computation
                loss.backward()    # Compute the gradient
                optimizer.step()   # Use the gradient information to make a step
            
            #Get results on train set
            x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
            y_pred = model(x).data.numpy()
            train_res = np.mean( np.argmax(y_pred, 1) == np.argmax(train_y, 1) )
            
            #Get results on val set
            x = Variable(torch.from_numpy(val_x), requires_grad=False).type(dtype_float)
            y_pred = model(x).data.numpy()
            val_res = np.mean( np.argmax(y_pred, 1) == np.argmax(val_y, 1) )
            
            train_acc += [train_res]
            val_acc += [val_res]

    #Get results on test set
    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    test_res = np.mean( np.argmax(y_pred, 1) == np.argmax(test_y, 1) )
    
    return train_acc, val_acc, test_res

def optimize_params(train_x, train_y, val_x, val_y, test_x, test_y):
    
    print('Trial 1')
    p1 = (25, 1e-2, 5, 1000)
    train_res, val_res, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, p1)
    print('Train Acc: ' + str(train_res) )
    print('Val Acc: ' + str(val_res) )
    print('Test Res: ' + str(test_res) + '\n')
    
    print('Trial 2')
    p2 = (25, 1e-3, 5, 1000)
    train_res, val_res, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, p2)
    print('Train Acc: ' + str(train_res) )
    print('Val Acc: ' + str(val_res) )
    print('Test Res: ' + str(test_res) + '\n')
        
    print('Trial 3')
    p3 = (25, 1e-3, 10, 800)
    train_res, val_res, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, p3)
    print('Train Acc: ' + str(train_res) )
    print('Val Acc: ' + str(val_res) )
    print('Test Res: ' + str(test_res) + '\n')
        
    print('Trial 4')
    p4 = (20, 1e-2, 5, 800)
    train_res, val_res, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, p4)
    print('Train Acc: ' + str(train_res) )
    print('Val Acc: ' + str(val_res) )
    print('Test Res: ' + str(test_res) + '\n')
        
    print('Trial 5')
    p5 = (20, 1e-3, 5, 1000)
    train_res, val_res, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, p5)
    print('Train Acc: ' + str(train_res) )
    print('Val Acc: ' + str(val_res) )
    print('Test Res: ' + str(test_res) + '\n')
        
    return 


if __name__ == "__main__": 

    
    #Set up data
    acts_m = ['hader', 'carell', 'baldwin']
    acts_f = ['harmon', 'bracco', 'gilpin']
    acts = acts_m + acts_f
    folders = ['resources/croppedMale']*3 + ['resources/croppedFemale']*3
    trsizes = {'hader':70, 'carell':70, 'baldwin':70, 'harmon':70, 'bracco':70, 'gilpin':37}
    
    data = format_data(acts, trsizes, folders, 2) #changing seed gives different results
    
    #get & format training, validation, test sets
    train_x, train_y = get_set_data(data, acts, 'train')
    val_x, val_y = get_set_data(data, acts, 'val')
    test_x, test_y = get_set_data(data, acts, 'test')
    
    #use validation set to optimize hyper-parameters
    optimize_params(train_x, train_y, val_x, val_y, test_x, test_y)

    #set chosen hyper-parameters    
    dim_h = 25      #started at 20
    rate = 1e-3
    no_epochs = 5
    iter = 1000      #iterations per mini_batch
    params = (dim_h, rate, no_epochs, iter)
    train_acc, val_acc, test_res = train(train_x, train_y, val_x, val_y, test_x, test_y, params)
    print('Final Train Acc: ' + str(train_acc[len(train_acc) -1]) )
    print('Final Val Acc: ' + str(val_acc[len(val_acc)-1]) )
    print('Test Res: ' + str(test_res) )
    
    if True:
        filename = 'part8.jpg'
        epochs = np.linspace(1, no_epochs*6, no_epochs*6)
        plt.scatter(epochs, train_acc, label='Training Data')
        plt.scatter(epochs, val_acc, label='Validation Data')
        plt.title('Learning Curve')
        plt.xlabel('number of mini-batches')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('resources/' + filename)
        #plt.show()
        plt.close()
