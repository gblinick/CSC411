import os
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from numpy import float32

import torch.nn as nn

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')

# a list of class names
from caffe_classes import class_names

# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
        
    def conv_output(self, x):
        ''' return the activations from the conv layers as a numpy array'''
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = x.data.numpy()
        return x

def get_activations(imgs):
    activations = []
    model = MyAlexNet()
    model.eval()
    for im in imgs:
        # Read an image
        im = imread(im)[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        
        # turn the image into a numpy variable
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False) 
        
        # get conv outputs
        conv_act = model.conv_output(im_v)
        activations += [ conv_act ]
    
    return activations


def format_data(acts, trsizes, folders, rd_seed=0):
    '''format data as dictionary'''
    
    data = {}
    
    for act in acts:
        k = acts.index(act)
        folder = folders[k]
        total = [folder+'/'+filename for filename in os.listdir(folder) if filename.startswith(act)] #list of all filenames for given act 
        rd.seed(rd_seed)
        rd.shuffle(total) #randomize order of files
                
        trsize = trsizes[act]
        trdata = total[:trsize] #training set for act
        tedata = total[trsize:trsize+10] #testing set for act
        vadata = total[trsize+10:trsize+20] #validation set for act
        
        x = 'resources/croppedMale_p10/baldwin11.jpg'
        y = 'resources/croppedFemale_p10/gilpin36.jpg'
        if x in trdata: 
            trdata.remove(x) #these are b/w images
        if y in trdata: 
            trdata.remove(y)
        
        data['train_'+act] = trdata
        data['val_'+act] = vadata
        data['test_'+act] = tedata
    
    return data


def format_y(M, set):   
    '''format y_ for the training, validation, and test set'''
    y0 = np.array([[1,0,0,0,0,0,0,0,0,0]]*len(M[set+'hader']) )
    y1 = np.array([[0,1,0,0,0,0,0,0,0,0]]*len(M[set+'carell']) )
    y2 = np.array([[0,0,1,0,0,0,0,0,0,0]]*len(M[set+'baldwin']) ) 
    y3 = np.array([[0,0,0,1,0,0,0,0,0,0]]*len(M[set+'harmon']) )
    y4 = np.array([[0,0,0,0,1,0,0,0,0,0]]*len(M[set+'bracco']) )
    y5 = np.array([[0,0,0,0,0,1,0,0,0,0]]*len(M[set+'gilpin']) )
    y_ = np.concatenate( (y0,y1,y2,y3,y4,y5), axis=0 ).T 
    return y_


def train(train_x, train_y, val_x, val_y, test_x, test_y, dim_x, params):
    ''' train the neural network on the output of conv layers'''
    dim_h, rate, no_epochs, iter = params
    dim_out = 6
    
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
        print('Starting Epoch: ' + str(k) )
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
    
    return train_acc, val_acc, test_res, model



if __name__ == "__main__": 

    dim_x = 9216

    #Set up data
    acts_m = ['hader', 'carell', 'baldwin']
    acts_f = ['harmon', 'bracco', 'gilpin']
    acts = acts_m + acts_f
    folders = ['resources/croppedMale_p10']*3 + ['resources/croppedFemale_p10']*3
    trsizes = {'hader':60, 'carell':60, 'baldwin':60, 'harmon':60, 'bracco':60, 'gilpin':32}
    
    data = format_data(acts, trsizes, folders, 2) #changing seed gives different results
    
    #get & format training, validation, test sets
    x_train_files = [ data[k] for k in data.keys() if k[0:2] == 'tr' ]
    x_train_files = [item for sublist in x_train_files for item in sublist]
    x_train = get_activations(x_train_files)
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, ( np.shape(x_train)[0] ,dim_x) )
    y_train = format_y(data, 'train_').T
    
    x_val_files = [ data[k] for k in data.keys() if k[0:2] == 'va' ]
    x_val_files = [item for sublist in x_val_files for item in sublist]
    x_val = get_activations(x_val_files)
    x_val = np.array(x_val)
    x_val = np.reshape(x_val, ( np.shape(x_val)[0] ,dim_x) )
    y_val = format_y(data, 'val_').T
    
    x_test_files = [ data[k] for k in data.keys() if k[0:2] == 'te' ]
    x_test_files = [item for sublist in x_test_files for item in sublist]
    x_test = get_activations(x_test_files)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, ( np.shape(x_test)[0] ,dim_x) )
    y_test = format_y(data, 'test_').T
    
    
    #Creat Torch NN
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    
    dim_h = 150      #started at 20
    rate = 1e-3
    no_epochs = 5
    iter = 100      #iterations per mini_batch
    params = (dim_h, rate, no_epochs, iter)
    train_acc, val_acc, test_res, nn = train(x_train, y_train, x_val, y_val, x_test, y_test, dim_x, params)
    print('Final Train Acc: ' + str(train_acc[len(train_acc) -1]) )
    print('Final Val Acc: ' + str(val_acc[len(val_acc)-1]) )
    print('Test Res: ' + str(test_res) )
    