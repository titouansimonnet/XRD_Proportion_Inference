""" Modules """

import numpy as np
import random

"""
***** Function mix *****

This function takes a list of crystals as input parameters and generates a random mixture of their diffraction patterns, with the mixing ratio ranging from 1 to number of mineral phases. It outputs a .txt file containing the names of the crystals in the mixture (including percentages and details) as well as a column representing relative intensity. Additionally, it writes each mixture to a .csv file, which includes the percentage composition of each of the mixtures that have been created.

***Parameters***
- list_cristal (string list) : mineral phases list to be mixed
- N (int)  Maximum number of mixed phases
- data (string) : train, val or test
- a_min & a_max (int) : interval to draw the single mineral phases XRD patterns
- data_size (int) : multiphases XRD patterns length
- num (int) : multiphases XRD pattern number (to write in .txt)
- fichier_label : file to write the labels (i.e mineral phases proportions)
- path : path to put the multiphases XRD files
"""


def mix(list_cristal,N,data,a_min,a_max,data_size,num,fichier_label,path):
    percent = np.zeros(N)
    Intensity = np.zeros(shape = (data_size,N))
    I = np.zeros(data_size)
    num_diffracto = np.zeros(N)
    for i in range(N):
        num_diffracto[i] = np.random.randint(a_min,a_max)
        percent[i] = np.random.randint(1,100)

    percent = percent / sum(percent)
    percent = np.around(percent,2)
    
    composant = random.sample(list_cristal,N)

    num = num
    melange = open(path+'Melange'+str(num)+'.txt',mode='w')
    fichier_label.write('Melange'+str(num)+'.txt,')

    
    for j in range(len(list_cristal)):
        present = 'no'
        for i in range(N):
            if (composant[i] == list_cristal[j]):
                a = int(num_diffracto[i])
                melange.write('Y :' + str(100*percent[i])+'% '+list_cristal[j]+ ', Diffracto n :'+str(a)+'\n')
                fichier_label.write(str(percent[i])+',')
                present = 'yes'
        if (present == 'no'):
            melange.write('N :' + list_cristal[j] + '\n')
            fichier_label.write('0,')
    
    fichier_label.write('\n')
    
    for i in range(N):
        a = int(num_diffracto[i])
        cristal = open("""path_single TO DO""" + composant[i]+str(a)+'.txt', mode='r')
        j = 0
        for line in cristal:
            if ('a' in line):
                titre_a = line
                continue
            Intensity[j,i] = float(line)
            j = j + 1
        
        I = I + Intensity[:,i]*percent[i]
    for i in range(len(Intensity)):
        melange.write(str(I[i]) + '\n')
        


list_cristal = ['Calcite','Gibbsite','Dolomite','Hematite']
path = """TODO"""


#Train
num = 0
label = open(path+'label_train.csv',mode='w')
a_min = 1
a_max = 1000
long = 10000
data = 'train'


for i in range(long):
    N = np.random.randint(1,4) # Mix from 1 to 3 single signals
    mix(list_cristal = list_cristal, N = N,data = data,a_min = a_min,a_max = a_max, data_size = 2905,num = num,fichier_label = label, path = path)
    num = num + 1
        
