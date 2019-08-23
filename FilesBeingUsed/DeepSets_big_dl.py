#THIS NETWORK TRIES TO PREDICT THE DECAY LENGTH (DL)

'''
Changes to make:
 - try L1loss maybe
 - add in a histogram for making the pT cut post running the nn
 - do the diffs, dists calculations or whatever on the np arrays, not lists, and reshape the np arrays
'''

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import statistics as stat


matplotlib.use('Agg')

#DETAILS OF THE ARCHITECTURE - TO MODIFY BETWEEN FILES
Arch = ",Hidden_layers_in_phi:100,50,50,30,F:50,100,50,30,POCAS,MULTIlr__DECAYDIST__"
nod = 100

err_opts = ['hasErr', 'noErr']
mepz_opts = ['hasTHETA_CHI2', 'hasTHETA_CHI2_dlCut', 'noMEPZ', 'hasEPZ', 'hasMEPZ']
poca = "pocas"
scal = "scal"
pTcut = 1.0
ptc = "pTcut:" + str(pTcut) + "GeV"
#could be for loops
mepz = 'hasTHETA_CHI2'
iserr = 'noErr'

n_epochs = 150
batch_size = 100
learning_rate = 0.001
lr_gamma = 0.5
latent_space_dim = 30
for  mepz in mepz_opts[:2]:
    if (mepz == mepz_opts[0]): dlcut = "trained on all decay lengths"
    else: dlcut = "trained only on decay length > 5mm"
    model_deets = "Latent_dim:%s"%latent_space_dim + Arch + ",eps:" + str(n_epochs) + "batch_size:" + str(batch_size) + "lr:" + str(learning_rate) + "lr_gamma:" + str(lr_gamma) + ",mass,energy,pz:" + mepz + ",err:" + iserr + "scal_inps"
    print(model_deets)
    X_train = torch.load('X_train_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    y_train = torch.load('ydl_train_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    X_test = torch.load('X_test_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    y_test = torch.load('ydl_test_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
        
    

    '''
    print(X_train.size())
    print("some pv indexes: ", X_train[:3,:,1])
    print("some jet indexes: ", X_train[:3,:,7])
    print("some pts: ", X_train[:3,:,0])
    '''

    """
    A few questions:
    - is the input gonna have a whole event, and we're gonna have to try and find both the LLP and anti-LLP vertex?
    - if so, how to train gradient descent if we're not sure what the target is?
    - can I just put all-zero particles for the events where there is less particles?
        - if not, would have to restructure this code
    - also, will there be any difficulty in getting the data due to overflow of previous event into non-existent particles?
    - loss function: x-y dist, or 3d, and sqd dist or dist?

    - pre-processing - do I need to make them all between 0 and 1 or something?
    """

    nvar = X_train.size(-1)


    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            
            #The phi part
            #Each particle in the jet is one row in the input matrix for each event
            self.fc1 = nn.Linear(nvar, nod)
            self.fc2 = nn.Linear(nod, 50)
            self.fc3 = nn.Linear(50, 50)
            self.fc4 = nn.Linear(50, 30)
            self.fc5 = nn.Linear(30, latent_space_dim)


            """
            The Summation
            We need to sum each column in the output matrix - how to perform on numpy array?
            Question: is x a tensor or a numpy array or both?
            this is just going to be an operation on a pytorch tensor, so better perform it in the forward bit        
            It will be performed by using torch.sum on the dimension in which we have particles (-2)
            Use negative indices as this will then be invariant whether or not we choose to use batching
            """

            #The F (regression to 3-D point) part
            self.fcA = nn.Linear(latent_space_dim, 50)
            self.fcB = nn.Linear(50, nod)
            self.fcC = nn.Linear(nod, 50)
            self.fcD = nn.Linear(50, 30)
            self.fcZ = nn.Linear(30 ,1)

        def forward(self, x):
            #phi layer
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.fc5(x)
            #Summation
            x = torch.sum(x, -2)
            #F part
            x = self.fcA(x)
            x = F.relu(x)
            x = self.fcB(x)
            x = F.relu(x)
            x = self.fcC(x)
            x = F.relu(x)
            x = self.fcD(x)
            x = F.relu(x)
            x = self.fcZ(x)
            return x
        
    # Which device to use for NN calculations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # Create network object
    model = NeuralNet().to(device)

    # Loss function
    criterion = nn.MSELoss()
    # Adam optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 40, 65], gamma=lr_gamma)

    # Keep track of the accuracies (note these are python lists)
    train_losses = []
    test_losses = []

    train_examples = X_train.size(0) #this is the number of elements along 0th dimension
    n_batches = int(train_examples/batch_size)
    n_test_batches = int(X_test.size(0) / batch_size)
    # Loop over the epochs
    for ep in range(n_epochs):
        # reorder the training events for each epoch
        idx = np.arange(X_train.size(0))
        np.random.shuffle(idx)
        #if (ep == 0): print(X_train)
        X_train = X_train[idx]
        y_train = y_train[idx]
        #if (ep == 0): print(X_train)
        
        # Each epoch is a complete loop over the training data
        for i in range(n_batches):
            # Reset gradient
            optimizer.zero_grad()
            
            i_start = i*batch_size
            i_stop  = (i+1)*batch_size
            
            # Convert x and y to proper objects for PyTorch
            #WHAT IS REL. OF DATATYPE HERE??
            #if (ep == 0): print("input", i, X_train[i_start:i_stop])
            #X_train[a:b] does take the ath to bth element on 0th dimension of the tensor
            x = X_train[i_start:i_stop].clone().detach().cuda()
            y = y_train[i_start:i_stop].clone().detach().cuda()

            # Apply the network 
            net_out = model(x)
            #print(net_out)
            # Calculate the loss function
            loss = criterion(net_out,y)
                    
            # Calculate the gradients
            loss.backward()
            
            # Update the weights
            optimizer.step()
            scheduler.step()
        # end of loop over batches
        
        #Calculate Ave Loss for the whole epoch, add to train_losses, test_losses
        #now we've finished training (for this epoch), we put the whole training set through, and the whole test set through
        #but this is just to calculate MSE and plot it, no training is done

        #torch.no_grad() is to disable gradient finding, because this will store the whole tensor in the gpu i think
        #which is why we were crashing going over memory before probably
        with torch.no_grad():
            y_pred_train = model(X_train[:batch_size].cuda())
            y_pred_test = model(X_test[:batch_size].cuda())


            for ss in range(1, n_batches + 1):
                tra1 = ss * batch_size
                tra2 = (ss + 1) * batch_size
                y_pred_train = torch.cat((y_pred_train, model(X_train[tra1:tra2].cuda())), 0)
            for gg in range(1, n_test_batches + 1): 
                tes1 = gg * batch_size
                tes2 = (gg + 1) * batch_size
                y_pred_test = torch.cat((y_pred_test, model(X_test[tes1:tes2].cuda())), 0)

        #using the squared 3d distance as our loss measure
        #to do 2d distance, im not quite sure how to do it - just remove z component from data I guess
        #pj = per jet
        #.item() converts a one-value tensor to a number
        train_loss_pj = (y_pred_train.cpu()-y_train)**2
        train_loss = torch.mean(train_loss_pj).item()
        test_loss_pj = (y_pred_test.cpu()-y_test)**2
        test_loss = torch.mean(test_loss_pj).item()
        print("epoch: ", ep, "train loss: ", train_loss, "test loss: ", test_loss)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        

    # Loss function over each epoch
    plt.plot(train_losses,label="train")
    plt.plot(test_losses,label="test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig("Loss_over_epochs_draft_%s.png"%model_deets)
    plt.clf()

    print("Best loss:", min(test_losses), "Final loss:", test_losses[-1])

    #How good was the model in the end?
    
    with torch.no_grad():    
        y_pred_test_final = model(X_test[:batch_size].cuda())
        for tt in range(1, n_test_batches + 1):
            ru1 = tt * batch_size
            ru2 = (tt + 1) * batch_size
            y_pred_test_final = torch.cat((y_pred_test_final, model(X_test[ru1:ru2].cuda())), 0)
    



    #Making same plots as last one, but using diffs (between real and pred decay length) rather than dists

    diffs = torch.sqrt((y_pred_test_final.cpu() - y_test)**2).tolist()
    diffs_np = np.array(diffs)
    diffs_np = diffs_np.reshape(diffs_np.shape[0])
    np.save("./Diff_Pred_True_Decay_Length%s.npy"%model_deets, diffs_np)
    print("Mean Diff:", np.mean(diffs_np))
    print("Diff Std. Dev.: ", np.std(diffs_np))

    #Histogram of Difference
    plt.hist(diffs_np, bins=200, range=(0,20))
    plt.title("Diff. Between Real Decay Length and Deep-Sets Predicted, \n %s"%dlcut)
    plt.xlabel("Diff (mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("Diff_Pred_true_hist_%s.png"%model_deets)
    plt.clf()

    #Scatterplot
    plt.scatter(y_test, diffs_np, s=0.5)
    plt.xlim((0,20))
    plt.ylim((0,20))
    plt.title("Diff Between true and pred. Decay Length \n vs. True Decay length, \n %s"%dlcut)
    plt.xlabel("Decay Length (mm)")
    plt.ylabel("Prediction error (mm)")
    plt.savefig("Diff_vs_True_decay_length_%s.png"%model_deets)
    plt.clf()

    #Now also create some plots after applying a cut on the lifetime
    dlsnp = np.array(y_test)
    print(dlsnp)
    ev_ord = np.flip(dlsnp.argsort(0), 0)
    ev_ord = ev_ord.reshape(ev_ord.shape[0])
    print(ev_ord)

    dlsord = dlsnp[ev_ord]
    ff = 0
    k = 0
    while dlsord[k] > 5:
        ff = k + 1
        k += 1
    print(ff)

    af_ord = ev_ord[:ff] #af = above five

    diffs_above5 = diffs_np[af_ord]
    print("diffs_above5", diffs_above5)
    print("Mean Diffs above 5:", np.mean(diffs_above5))
    print("Diffs above 5 Std. Dev.: ", np.std(diffs_above5))

    plt.hist(diffs_above5, bins=200, range=(0,20))
    plt.title("Diff. Between Real Decay Length and Deep-Sets Predicted, \n for Decay length > 5mm, \n %s"%dlcut)
    plt.xlabel("Diff (mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("diff_hist_decay_length_5_cut_%s.png"%model_deets)
    plt.clf()
    

