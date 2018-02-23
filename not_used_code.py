#Cut out code



##DIGITS.py

# Code not used from Profs
    if False:
        
        
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


    #Part 6e
    '''
    W = W_part5.copy()
    w1_pos = (1,1)
    w2_pos = (1,2)
    #w1 = W[w1_pos]
    #w2 = W[w2_pos]
    W[w1_pos] += 7
    W[w2_pos] += 7
    w1_prog, w2_prog = backprop_monitor_W(x_train, y_train, w1_pos, w2_pos, W, b, 1e-2, 20, mom=0)
    W = W_part5.copy()
    W[w1_pos] += 7
    W[w2_pos] += 7
    w1_prog_mom, w2_prog_mom = backprop_monitor_W(x_train, y_train, w1_pos, w2_pos, W, b, 1e-4, 20, mom=0.8)
    CS = plt.contour(hor, ver, vals)
    plt.plot(w1_prog, w2_prog, label='No Momentum')
    plt.plot(w1_prog_mom, w2_prog_mom, label='Momentum')
    plt.title('Contour Plot with Weight Trajectories')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.legend()
    plt.savefig('resources/part6e.jpg')
    plt.show()
    plt.close()
    '''
    
    
##DEEPFACES.PY

    if False:
        # model_orig = torchvision.models.alexnet(pretrained=True)
        model = MyAlexNet()
        model.eval()
        
        # Read an image
        im = imread('resources/kiwi227.png')[:,:,:3]
        im = im - np.mean(im.flatten())
        im = im/np.max(np.abs(im.flatten()))
        im = np.rollaxis(im, -1).astype(float32)
        
        # turn the image into a numpy variable
        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)    
        
        # run the forward pass AlexNet prediction
        softmax = torch.nn.Softmax()
        all_probs = softmax(model.forward(im_v)).data.numpy()[0]
        sorted_ans = np.argsort(all_probs)
        
        for i in range(-1, -6, -1):
            print("Answer:", class_names[sorted_ans[i]], ", Prob:", all_probs[sorted_ans[i]])
        
        ans = np.argmax(model.forward(im_v).data.numpy())
        prob_ans = softmax(model.forward(im_v)).data.numpy()[0][ans]
        print("Top Answer:", class_names[ans], "P(ans) = ", prob_ans)