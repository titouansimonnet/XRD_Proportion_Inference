"""
Modules
"""
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


"""
***Classes definition and path to data***
"""
classes = ['Calcite','Gibbsite','Dolomite','Hematite']
path = '~/Databases/Experimental/'

real_datalist = pd.read_csv(path+'Labels_T.csv', sep=';',header=None,names=['file','Calcite','Gibbsite',
                                        'Dolomite','Hematite'],skiprows=1)

Data = torch.zeros(32,2905)
for i in range(32):
    V = pd.read_csv(path+real_datalist['file'][i],header = None, skiprows = 0,
                    names = ['angle','intensity'],sep = ',')
    V_tensor = np.asarray(V['intensity'])
    V_tensor = torch.tensor(V_tensor).float()
    if (len(V_tensor) == 2905):
        Data[i,:] = V_tensor
 
"""
Data pre-processing
"""
Data_m_2 = np.array(Data)
for i in range(len(Data_m_2)):
    m = min(Data_m_2[i,:])
    Data_m_2[i,:] = Data_m_2[i,:] - m
    Data_m_2[i,:] = Data_m_2[i,:]/max(Data_m_2[i,:])
Data_M_2 = torch.tensor(Data_m_2)


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
Function to test the trained neural network on the pixels of the Data
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
        for i in range(len(Data_M_2)):
            #Load the data
            images = Data_M_2[i,:]
            images = torch.unsqueeze(images,0)
            images = torch.unsqueeze(images,0)
            
            #Load the label
            prop_Calcite = real_datalist.loc[i,'Calcite']
            prop_Gibbsite = real_datalist.loc[i,'Gibbsite']
            prop_Dolomite = real_datalist.loc[i,'Dolomite']
            prop_Hematite = real_datalist.loc[i,'Hematite']
            p = torch.tensor([prop_Calcite,prop_Gibbsite,prop_Dolomite,prop_Hematite])
            
            #Results after NN
            y = model_ft(images)
            res = criterion.prediction(y)
            res = res.squeeze(0)
            
            #RRS
            predictive_support = res > eps
            true_support = (p > 0)
            if np.array_equal(true_support,predictive_support):
                good_support = good_support + 1
            
            #MMAE
            Diff_abs = torch.abs(p - res)
            Diff_abs = np.array(Diff_abs)
            norme_inf = norme_inf + np.amax(Diff_abs)
            
            #RMSE
            norm = np.array((p-res)**2)
            norme_2 = norme_2 + np.mean(norm)
            norme_2_classes = norme_2_classes + norm
            
        norme_inf = norme_inf / len(Data)
        norme_2 = norme_2 / len(Data)
        RMSE = np.sqrt(norme_2)
        RMSE_classes = np.sqrt( norme_2_classes / len(Data))
        
        supp = good_support / len(Data)

        output_file.write('Infinite norm : ' + str(norme_inf*100) + '\n')
        output_file.write('RMSE : ' + str(RMSE) + '\n')
        output_file.write('‰ Support : ' + str(supp*100) + '\n' )
        output_file.write('RMSE_classes : ' + str(RMSE_classes[0]) +','+ str(RMSE_classes[1])+','+ str(RMSE_classes[2])+',' + str(RMSE_classes[3]) + '\n' + '\n')
    
        return (norme_inf, RMSE, supp, RMSE_classes)


output = open('Results_exp_data.txt', mode = 'w') #File for keep results
nb_training_by_loss = 1 # Number of training for each loss

"""
*** Dirichlet et MSE ***
"""
N_i = np.zeros(nb_training_by_loss)
Supp = np.zeros(nb_training_by_loss)
RMSE = np.zeros(nb_training_by_loss)
RMSE_classes = np.zeros(len(classes))
for i in range(nb_training_by_loss):
    output.write('Test' + str(i) + ':' + '\n')
    model_ft = Net()
    criterion = LossDirichlet_MSE()
    model_ft.load_state_dict(torch.load('NN_trained_database2' ,map_location=torch.device('cpu')))
    N_i[i], RMSE[i], Supp[i], RMSE_classes = test(model = model_ft, output_file = output, criterion = criterion)


output.close()

