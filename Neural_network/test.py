"""
Modules
"""
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


"""
***Classes definition and path to data***
"""
classes = ['Calcite','Gibbsite','Dolomite','Hematite']
path = '~/Databases/Mix_norm/'

# Using a GPU

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
        
#Testloader
batch_size_test = 10
testset = Diffractogramm(train='test')
testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=0)

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

#Loss 1 : DIR & MSE
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
Function to test the trained neural network on the pixels of the testset
*** PARAMETERS ***
- model = trained neural network
- output_file = file for write the output (RMSE, Infinite norm, %Support)
- criterion : criterion use for trained NN (use "out" method)
"""

def test (model, output_file, criterion):
    model_ft = model
    norme_inf = 0
    norme_2 = 0
    norme_2_classes = np.zeros(len(classes))
    good_support = 0
    eps = 1e-2
    with torch.no_grad():
        model_ft.eval()
        for data in testloader:
            images = data[0]
            p = data[1]
            y = model_ft(images)
            y = criterion.prediction(y)
            res = y.cpu()
            
            predictive_support = res > eps
            true_support = (p > 0)
            for m in range(predictive_support.shape[0]):
                if np.array_equal(true_support[m],predictive_support[m]):
                    good_support = good_support + 1
            
            Diff_abs = torch.abs(p - res)
            Diff_abs = np.array(Diff_abs)
            
            norme_inf = norme_inf + np.sum(np.amax(Diff_abs, axis = 1))
            
            norm = np.array( (p-res)**2 )
            norme_2 = norme_2 + np.sum(np.mean(norm, axis = 1))
            norme_2_classes = norme_2_classes + np.sum(norm, axis = 0)
            
        norme_inf= norme_inf / len(testset)
        
        norme_2 = norme_2 / len(testset)
        RMSE = np.sqrt(norme_2)
        RMSE_classes = np.sqrt( norme_2_classes / len(testset))
        
        supp = good_support / len(testset)

        output_file.write('MMAE : ' + str(norme_inf) + '\n')
        output_file.write('RMSE : ' + str(RMSE) + '\n')
        output_file.write('RRS: ' + str(supp) + '\n' )
        output_file.write('RMSE_classes : ' + str(RMSE_classes[0]) +','+ str(RMSE_classes[1])+','+ str(RMSE_classes[2])+',' + str(RMSE_classes[3]) + '\n' + '\n')
    
        return (norme_inf, RMSE, supp, RMSE_classes)


output = open('Results_XRD.txt', mode = 'w') #File for keep results
nb_training = 1 # Number of training

"""
*** Dirichlet et MSE ***
"""
N_i = np.zeros(nb_training)
Supp = np.zeros(nb_training)
RMSE = np.zeros(nb_training)
RMSE_classes = np.zeros(len(classes))
for i in range(nb_training):
    print('Test')
    model_ft = Net()
    criterion = LossDirichlet_MSE()
    model_ft.load_state_dict(torch.load(NN_trained_database2,map_location=torch.device('cpu')))
    N_i[i], RMSE[i], Supp[i], RMSE_classes = test(model = model_ft, output_file = output, criterion = criterion)

output.close()



