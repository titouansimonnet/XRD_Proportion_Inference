import numpy as np
import pandas as pd


"""
***** Function write *****
Rewrite in a new file

PARAMETERS :
- path: path to stock files
- path_or :path to access originals files
- file_name: Original file data to get the parameters of crystal
- data(Matrix): Normalize signal

"""
def write(new_path,old_path,file_name,data):
    f = open(old_path+file_name, 'r')
    fichier = open(new_path+file_name,'w')
    for j in range(1):
        j = f.readline()
        fichier.write(j)
    for i in range(len(data)):
        fichier.write(str(data[i]) + '\n')
    
    


"""
Attenuation coefficient for Calcite, Dolomite and Hematite.
Calculating according to the Gibbsite's one.
"""
coeff_calcite = 0.435215611
coeff_dolomite = 0.677349206
coeff_hematite = 0.176194569


# Calculation of the new intensity for single mineral phases XRD patterns. To do before mixing the single phases signals !!

"""
PARAMETERS TO DO
"""
old_path = '' # TODO
new_path = '' # TODO
N = 1 # Number of single phase XRD patterns for each mineral phases.



for l in range(N):
    C = pd.read_csv(old_path + 'Calcite' + str(l+1) + '.txt',skiprows=1,header=None,names=['I'])
    H = pd.read_csv(old_path + 'Hematite' + str(l+1) + '.txt',skiprows=1,header=None,names=['I'])
    D = pd.read_csv(old_path + 'Dolomite' + str(l+1) + '.txt',skiprows=1,header=None,names=['I'])
    G = pd.read_csv(old_path + 'Gibbsite' + str(l+1) + '.txt',skiprows=1,header=None,names=['I'])
    
    I_C = C['I']*coeff_calcite
    I_D = D['I']*coeff_dolomite
    I_H = H['I']*coeff_hematite
    
    write(new_path = new_path, old_path = old_path, file_name = 'Calcite' + str(l+1) + '.txt',data = I_C)
    write(new_path = new_path, old_path = old_path, file_name = 'Dolomite' + str(l+1) + '.txt',data = I_D)
    write(new_path = new_path, old_path = old_path, file_name = 'Hematite' + str(l+1) + '.txt',data = I_H)
    write(new_path = new_path, old_path = old_path, file_name = 'Gibbsite' + str(l+1) + '.txt',data = G['I'])
    
    
    if (l%100 == 0):
        print(np.round((100*l/N),0), ' % effectué')
