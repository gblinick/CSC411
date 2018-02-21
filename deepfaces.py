from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
  

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project2/CSC411/')


def format_data():
    '''format data as dictionary for use as with MNIST data'''
    
    folders = ['resources/croppedMale']*6 + ['resources/croppedFemale']*6
    acts_m = ['hader', 'butler', 'carell', 'radcliffe', 'baldwin', 'vartan']
    acts_f = ['harmon', 'bracco', 'gilpin', 'drescher', 'chenoweth', 'ferrera']
    acts = acts_m + acts_f
    trsizes = {'hader':79, 'butler':79, 'carell':79, 'radcliffe':79, 'baldwin':79, 'vartan':79}
    trsizes.update( {'harmon':79, 'bracco':79, 'gilpin':39, 'drescher':79, 'chenoweth':79, 'ferrera':79} )
    
    data = {}
    
    for act in acts:
        k = acts.index(act)
        folder = folders[k]
        total = [folder+'/'+filename for filename in os.listdir(folder) if filename.startswith(act)] #list of all filenames for given act 
        rd.seed(0)
        rd.shuffle(total) #randomize order of files
        
        x = np.array([ imread(total[i]).flatten() for i in range( len(total) ) ])/255.0
        
        trsize = trsizes[act]
        trdata = x[:trsize, :] #training set for act
        vadata = x[trsize+10:trsize+10, :] #validation set for act
        tedata = x[trsize:trsize+10, :] #testing set for act
        
        data['train_'+act] = trdata
        data['val_'+act] = vadata
        data['test_'+act] = tedata
        
    return data
    


def get_test(M, acts):
    '''
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    test_k =  ["test"+str(i) for i in range(10)]
    '''
    
    batch_xs = np.zeros((0, 32*32))
    batch_y_s = np.zeros( (0, 12))
    test_k = ['test_'+act for act in acts]
    
    for k in range(12):
        batch_xs = np.vstack((batch_xs, ((np.array(M[test_k[k]])[:])/255.)  ))
        one_hot = np.zeros(12)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_train(M,acts):
    '''
    batch_xs = np.zeros((0, 28*28))
    batch_y_s = np.zeros( (0, 10))
    train_k =  ["train"+str(i) for i in range(10)]
    '''
    
    batch_xs = np.zeros((0, 32*32))
    batch_y_s = np.zeros( (0, 12))
    train_k = ['train_'+act for act in acts]
    
    for k in range(12):
        batch_xs = np.vstack((batch_xs, ((np.array(M[train_k[k]])[:])/255.)  ))
        one_hot = np.zeros(12)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s,   np.tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s



'''
M = loadmat("mnist_all.mat")

train_x, train_y = get_train(M)
test_x, test_y = get_test(M)

dim_x = 28*28
dim_h = 20
dim_out = 10
'''

data = format_data()
acts_m = ['hader', 'butler', 'carell', 'radcliffe', 'baldwin', 'vartan']
acts_f = ['harmon', 'bracco', 'gilpin', 'drescher', 'chenoweth', 'ferrera']
acts = acts_m + acts_f

train_x, train_y = get_train(data, acts)
test_x, test_y = get_test(data, acts)

dim_x = 32*32
dim_h = 20
dim_out = 12

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor



################################################################################
#Subsample the training set for faster training

train_idx = np.random.permutation(range(train_x.shape[0]))[:1000]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(dtype_long)
#################################################################################


model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step

x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()

res = np.mean( np.argmax(y_pred, 1) == np.argmax(test_y, 1) )
