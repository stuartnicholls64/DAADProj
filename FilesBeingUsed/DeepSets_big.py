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
Arch = ",Hidden_layers_in_phi:100,50,50,30,F:50,100,50,30,POCAS,MULTIlr"
nod = 100

mepz_opts = ['hasTHETA_CHI2', 'hasTHETA_CHI2_dlCut', 'noMEPZ', 'hasEPZ', 'hasMEPZ']
poca = "pocas"
scal = "scal"
pTcut = 1.0
ptc = "pTcut:" + str(pTcut) + "GeV"
#could be for loops
iserr = 'noErr'

n_epochs = 150
batch_size = 100
learning_rate = 0.001
lr_gamma = 0.5

latent_space_dim = 30
for mepz in mepz_opts[:2]:
    if (mepz == mepz_opts[0]): dlcut = "trained on all decay lengths"
    else: dlcut = "trained only on decay length > 5mm"
    model_deets = "Latent_dim:%s"%latent_space_dim + Arch + ",eps:" + str(n_epochs) + "batch_size:" + str(batch_size) + "lr:" + str(learning_rate) + "lr_gamma:" + str(lr_gamma) + ",mass,energy,pz:" + mepz + ",err:" + iserr + "scal_inps"
    print(model_deets)
    X_train = torch.load('X_train_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    y_train = torch.load('y_train_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    X_test = torch.load('X_test_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    y_test = torch.load('y_test_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    dec_len_test = torch.load('ydl_test_%s_%s_%s_%s_%s.pt'%(iserr, mepz, poca, scal, ptc))
    

    nvar = X_train.size(-1)


    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            
            #The phi part
            #Each particle in the jet is one row in the input matrix for each event
            elf.fc1 = nn.Linear(nvar, nod)
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
            self.fcZ = nn.Linear(30 ,3)

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
        train_loss_pj = torch.sum((y_pred_train.cpu()-y_train[:])**2, -1)
        train_loss = torch.mean(train_loss_pj).item()
        test_loss_pj = torch.sum((y_pred_test.cpu()-y_test)**2, -1)
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
    dists = torch.sqrt(torch.sum((y_pred_test_final.cpu() - y_test)**2, -1)).tolist()
    dists_np = np.array(dists)
    np.save("./Distances_to_SV_%s.npy"%model_deets, dists_np)
    print("Mean Distance:", stat.mean(dists))
    print("Dist Std. Dev.: ", stat.stdev(dists))

    #Histogram of the distances to the vertex
    plt.hist(dists, bins=200, range=(0,20))
    plt.title("Distance from Gen. Vertex to Deep-Sets Predicted, \n %s"%dlcut)
    plt.xlabel("Dist (mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("Distance_to_true_draft_%s.png"%model_deets)
    plt.clf()

    #Finding the radial Dists as well
    print("y_test", y_test)
    print("y_test x,y: ", y_test[:,:2])
    print("y_pred_test: ", y_pred_test_final.cpu())
    dists2D = torch.sqrt(torch.sum((y_pred_test_final[:,:2].cpu() - y_test[:,:2])**2, -1)).tolist()
    dists2D_np = np.array(dists2D)
    print("2D dists: ", dists2D_np)
    np.save("./2D_Dists_to_SV_%s"%model_deets, dists2D_np)
    print("Mean 2D Distance:", stat.mean(dists2D))
    print("2D-Dist Std. Dev.: ", stat.stdev(dists2D))

 
    plt.hist(dists2D, bins=200, range=(0,20))
    plt.title("2D (radial) distance from Gen. Vertex to Deep-Sets Predicted\n %s"%dlcut)
    plt.xlabel("2D Dist(mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("2D_Distance_to_true_draft_%s.png"%model_deets)
    plt.clf()

    #Finding the z Dists as well
    print("y_test", y_test)
    print("y_test z: ", y_test[:,2])
    print("y_pred_test: ", y_pred_test_final.cpu())
    distsZ_abs = torch.sqrt((y_pred_test_final[:,2].cpu() - y_test[:,2])**2).tolist()
    distsZ = (y_pred_test_final[:,2].cpu() - y_test[:,2]).tolist()
    distsZ_np = np.array(distsZ)
    print("Z dists: ", distsZ_np)
    np.save("./Z_Dists_to_SV_%s"%model_deets, distsZ_np)
    print("Mean Abs Z Distance:", stat.mean(distsZ_abs))
    print("Abs. Z-Dist Std. Dev.: ", stat.stdev(distsZ_abs))
    print("Mean Z Dist: ", stat.mean(distsZ))
    print("Z-Dist Std. Dev.: ", stat.stdev(distsZ))

    plt.hist(distsZ, bins=400, range=(-20,20))
    plt.title("Z distance from Gen. Vertex to Deep-Sets Predicted\n %s"%dlcut)
    plt.xlabel("Z Dist(mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("Z_Distance_to_true_draft_%s.png"%model_deets)
    plt.clf()

    #A scatterplot showing the distance as a function of the decay distance
    decay_dists = torch.sqrt(torch.sum(y_test**2, -1)).tolist()
    decay_dists_np = np.array(decay_dists)
    np.save('./decay_dists_(test).npy%s'%model_deets, decay_dists)
    plt.scatter(decay_dists, dists, s=0.5)
    plt.xlim((0,20))
    plt.ylim((0,20))
    plt.title("Distance from Pred. vertex to Gen Vertex \n vs. Decay Distance (3D)\n %s"%dlcut)
    plt.xlabel("Decay Distance (dist. from origin to LLP decay) (mm)")
    plt.ylabel("Distance from Pred. to true Vertex (mm)")
    plt.savefig("Dist_to_true_vs_Decay_dist_%s.png"%model_deets)
    plt.clf()

    #A scatterplot showing the distance as a function of the decay distance
    decay_dists2D = torch.sqrt(torch.sum(y_test[:,:2]**2, -1)).tolist()
    decay_dists2D_np = np.array(decay_dists2D)
    np.save('./decay_dists2D_(test).npy%s'%model_deets, decay_dists)
    plt.scatter(decay_dists2D, dists, s=0.5)
    plt.xlim((0,20))
    plt.ylim((0,20))
    plt.title("Distance from Pred. vertex to Gen Vertex \n vs. Decay Distance (2D)\n %s"%dlcut)
    plt.xlabel("Decay Distance (Radial dist. from origin to LLP decay) (mm)")
    plt.ylabel("Distance from Pred. to true Vertex (mm)")
    plt.savefig("Dist_to_true_vs_Decay_dist2D_%s.png"%model_deets)
    plt.clf()

    #Histogram of loss distance after a >5mm cut (should be the same as the histo already made
    #if we trained on >5mm)
    dlsnp = np.array(dec_len_test)
    ev_ord = np.flip(dlsnp.argsort(0), 0)
    ev_ord = ev_ord.reshape(ev_ord.shape[0])
    dlsord = dlsnp[ev_ord]

    ff = 0
    k = 0
    while dlsord[k] > 5:
        ff = k + 1
        k += 1

    af_ord = ev_ord[:ff] #af = above five
    print("af_ord: ", af_ord.shape, af_ord)
    dists_above5 = dists_np[af_ord]
    print("dists_above5:", dists_above5)
    print("mean dists (above 5):", np.mean(dists_above5))
    print("std dev dists (above 5):", np.std(dists_above5))

    plt.hist(dists_above5, bins=200, range=(0,20))
    plt.title("Distance from true to Deep Sets Pred. Vertex \n for decay length > 5mm \n %s"%dlcut) 
    plt.xlabel("Dist (mm)")
    plt.ylabel("Frequency (absolute)")
    plt.savefig("Dist_hist_decay_length_5_cut_%s.png"%model_deets)
    plt.clf()
    #copy and edit the rest over from DeepSets_big_dl once that is finished and seems to work
