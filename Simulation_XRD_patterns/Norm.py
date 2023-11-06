import numpy as np
import pandas as pd


"""
***** Function norm_1 *****
Normalize the 1D-signal XRD pattern to obtain a maximum = 1.
Divided by the maximum intensity.

Parameter:
- y : original signal we want to normalize
"""
def norm_1(y):
    y = y[:][0]
    y = y / max(y)
    return(y)

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
    for j in range(4):
        j = f.readline()
        fichier.write(j)
    for i in range(len(data)):
        fichier.write(str(data[i]) + '\n')
    
    

old_path = """ TO DO """
new_path = """ TO DO """

for l in range(10000):
    file_name = 'Melange' + str(l) + '.txt'
    data = pd.read_csv(old_path + file_name,skiprows=4,header = None)
    v = norm_1(y = data)
    write(new_path = new_path, old_path = old_path, file_name = file_name,data = v)
    if (l%100 == 0):
        print(np.round((100*l/10000),0), ' % done')
