matplotlib.use('Agg')
import torch
from torchvision import transforms, utils
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import copy
from datetime import datetime



"""
***Classes definition and path to data***
"""
classes = ['Calcite','Gibbsite','Dolomite','Hematite']
path = 'Databases/Mix_norm/'

#GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
*** Class Dataset definition ***
"""
class Diffractogramm(Dataset):
    def __init__(self, train = 'train'):
        """
            train (str) : 'test','train', 'val'
        """
        self.train = train
        if ( self.train == 'train'):
            self.datalist = pd.read_csv(path+'label_train.csv',
                sep=',',header=None,names=['file','Calcite','Gibbsite','Dolomite','Hematite','empty'])
        elif (self.train == 'test'):
            self.datalist = pd.read_csv(path+'label_test.csv',
                sep=',',header=None,names=['file','Calcite','Gibbsite','Dolomite','Hematite','empty'])
        else :
            self.datalist = pd.read_csv(path+'label_val.csv',
            sep=',',header=None,names=['file','Calcite','Gibbsite','Dolomite','Hematite','empty'])
        
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file_name = self.datalist['file'][idx] #Get the .txt file with a data
        fichier = pd.read_csv(path+file_name,header=None,skiprows=4,names=['Intensity'])
        #Open this file with pandas
        diffracto = fichier['Intensity'] #Get the intensity
        #Get the proportions
        proportion_Calcite = self.datalist['Calcite'][idx]
        proportion_Gibbsite = self.datalist['Gibbsite'][idx]
        proportion_Dolomite = self.datalist['Dolomite'][idx]
        proportion_Hematite = self.datalist['Hematite'][idx]
        proportion =[proportion_Calcite,proportion_Gibbsite,proportion_Dolomite,proportion_Hematite]
        proportion = np.asarray(proportion)
        diffracto = torch.tensor(diffracto).float()
        diffracto = torch.unsqueeze(diffracto,0)
        sample = [diffracto,proportion]
        return sample
        
#Create the datasets/dataloader (train & validation)
batch_size_train = 10
batch_size_val = 10
trainset = Diffractogramm(train = 'train')
valset = Diffractogramm(train = 'val')
dataloader = DataLoader(trainset, batch_size = batch_size_train, shuffle = True, num_workers=0)
validloader = DataLoader(valset, batch_size = batch_size_val , shuffle = True, num_workers=0)


"""
*** Neural Network with PyTorch***
"""
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 8, stride = 8,padding=100)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 5)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 3)
        self.fc1 = nn.Linear(32*25, 1024)
        self.fc2 = nn.Linear(1024, 4)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
Definition of a class (nn.Module) for the 5 loss function :
    - forward method is used during the training (loss function)
    - prediction method is used for validation and test set like a predictor

*** PARAMETERS *** (for each function)
    - a is the output of the neural network
    - p is the proportion vector from the ground truth
"""

class LossDirichlet_MSE(nn.Module):
    def forward(self, a , p):
        b,K = a.shape # batch size and number of classes
        alpha = F.relu(a) + 1.0 #phi function of the paper
        S = torch.sum(alpha, dim = 1, keepdim = True) # sum of alpha
        moment_1 = alpha / S #moment of order 1 for Dirichlet Distribution
        moment_2 = alpha*(1+alpha) / (S*(1+S)) #moment of order 2 for Dirichlet Distribution
        L = torch.sum((p*p - 2*p*moment_1 + moment_2), dim = 1)
        L = torch.mean(L)
        return(L)
    
    def prediction(self,a):
        output = F.relu(a) + 1.0
        S = torch.sum(output,dim = 1, keepdim = True)
        output = output/S
        return(output)




"""
**** TRAIN FUNCTION ****
Goal : return the model with the best infinite norm on the validation set among all the epochs
******** PARAMETERS **********
- model : Neural Network to train
- nb_epoch : Number of epochs for the training
- criterion : Loss function we use to train
- eps : Value for the percent of good support on validation set
- N_train = number of the training (useful for Tensorboard)
"""

def train(model, nb_epoch, criterion, eps, N_train):
    norm_val = np.inf
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    n_train = len(trainset)
    iter_info = 100
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(nb_epoch):
        since_epoch = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        model.train()
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images = data[0].to(device)
            proportion = data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs.float(),proportion.float()) #need .float()?
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item() * batch_size_train
            if i % iter_info == (iter_info-1):    # print every iter_info mini-batches
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / iter_info))
                running_loss = 0.0
        time_elapsed_epoch = time.time() - since_epoch
        print('Epoch completed in {:.0f}m {:.0f}s'.format(
            time_elapsed_epoch // 60, time_elapsed_epoch % 60))
        epoch_loss = epoch_loss / n_train
        print('Epoch loss: %.3f' % (epoch_loss))

        #Validation set
        norme_inf = 0
        norme_2 = 0
        good_support = 0
        with torch.no_grad():
            model.eval()
            for data in validloader:
                images = data[0].to(device)
                p = data[1]
                output = model(images)
                output = criterion.prediction(output)
                res = output.cpu()
                
                predictive_support = res > eps
                true_support = (p > 0)
                for m in range(predictive_support.shape[0]):
                    if np.array_equal(true_support[m],predictive_support[m]):
                        good_support = good_support + 1
                
                Diff_abs = torch.abs(p-res)
                Diff_abs = np.array(Diff_abs)
                norme_inf = norme_inf + np.sum(np.amax(Diff_abs,axis = 1))
                norm = np.array( (p-res)**2 )
                norme_2 = norme_2 + np.sum(np.mean(norm,axis = 1))

            norme_inf = norme_inf / len(valset)
            norme_2 = norme_2 / len(valset)
            RMSE = np.sqrt(norme_2)
            supp = good_support/len(valset)
            
            
            print(str(criterion))
            print('Epoch: ',epoch+1,', MMAE : ', norme_inf)
            print('RMSE : ',RMSE)
            print('RRS : ',supp)
            
            if (norme_inf < norm_val):
                norm_val = norme_inf
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best epoch')
            
        
    print('Finished Training')
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
    model.load_state_dict(best_model_wts)
    return(model)

nb_epoch = 100
eps = 1e-2 # threshold for RRS
nb_tr = 1 # training number

for w in range(nb_tr):
    model = Net()
    criterion = LossDirichlet_MSE()
    print('Train '+ str(w))
    model_train = train(model = model, nb_epoch = nb_epoch, criterion = criterion, eps = eps, N_train = w)
    torch.save(model_train.state_dict(),f = 'trained_nn'+str(w))
