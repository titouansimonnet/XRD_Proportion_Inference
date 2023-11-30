"""
*** Modules ***
"""
from crystals import Crystal
from crystals import Atom
import numpy as np
import pandas as pd
import CifFile
import time



"""
***** Function periodic_compt *****:
Function to obtain the number of atom for each type in the unit cell and their atomic number

Parameters:
- Pos: Matrix containing position and nature of the atom in the unit-cell. Extract using crystals module
- F: List containing the different atom in the unit-cell.Extract using crystals module

Output:
- compt (np.array): Matrix containing the number of atoms of each type
- num_periodique (np.array): Matrix with the atomic number of each atom.

"""

def periodic_compt(Pos,F):
  compt = np.zeros(len(F))
  num_periodique = np.zeros(len(F))
  k = Pos[0][0]
  num_periodique[0] = k
  j = 0
  c = 0
  for i in range(len(Pos)):
    if (Pos[i][0] == k) :
      c = c + 1
      compt[j] = c
    
    else : 
      c = 1
      j = j+1
      k = Pos[i][0]
      num_periodique[j] = k
  
  compt = compt.astype(int)
  return (compt,num_periodique)

"""
***** Function Debye_w *****
Apply the Debye_waller coeeficient to the atomic factor diffusion

Parameters:
- DW (np.array): Matrix containing the Debye-Waller coefficient
- f (np.array): atomic factor diffusion matrix
- theta (np.array): vector theta (diffraction angles)
- l (float): Emmited X-Ray wavelength

Output:
- No output, just modify the matrix f according to the Debye-Waller coefficient
"""
  
#DW : matrice avec coeff de DW
#f : matrice des facteurs de diffusion atomique
#theta : vecteur des valeurs de theta
# l : longueur d'onde des rayons X

def Debye_w(DW,f,theta,l):
  for i in range(len(DW)):
    f[0:len(f),i] = f[0:len(f),i]*np.exp(-DW[i]*(np.sin(theta*np.pi/180)*np.sin(theta*np.pi/180))/(l*l))
  return (f)
  
  
"""
***FONCTIONS FACTEUR DIFFUSION ATOMIQUE***

* get_coeff_facteur_diffusion_base:
Retourne les coeff a1,b1,..., a5,b5 et c
* facteur_diffusion_atomique_base:
Pour une valeur de d donnée, calcule les valeurs du facteur atomique pour chaque atome du tableau
* matrice_f_diffusion_base:
Calcule la matrice avec facteur diff atomique pour chaque valeur de d
"""

"""
***** Function get_coeff_atomic_factor *****
Return coefficient a1,b1,..., a5,b5 and c to calculate the atomic diffusion factor

Parameters:
- i (int): index the atoms
- factor (pandas data.frame): Dataframe containing the coefficient for a large amount of atom

Outputs:
A list containing all the coefficient to calculate the atomic diffusion factor, first element of the list is a vector with a, second a vector with b and the last is c.
"""

def get_coeff_atomic_factor(factor,i):
  if (i < len(factor)):
    a1 = factor['a1'][i]
    b1 = factor['b1'][i]
    a2 = factor['a2'][i]
    b2 = factor['b2'][i]
    a3 = factor['a3'][i]
    b3 = factor['b3'][i]
    a4 = factor['a4'][i]
    b4 = factor['b4'][i]
    a5 = factor['a5'][i]
    b5 = factor['b5'][i]
    c = factor['c'][i]
    return ([a1,a2,a3,a4,a5],[b1,b2,b3,b4,b5],c)
  else : return (0)

"""
***** Function factor_atomic_diffusion *****
For a given value of d (i.e browse vector theta) return the atomic factor diffusion for all atoms.

Parameters:
- factor (pandas data.frame): Dataframe containing the coefficient for a large amount of atom
- d (float): a ponctual value of the distance between 2 atomic planes (deduce from theta using Bragg's law).

Output:
- F (np.array): Vector containing with the calculated atomic factor for all atom in the factor list.
"""

def factor_atomic_diffusion(d,factor):
  F = np.zeros(len(factor))
  s = 1/(2*d)
  for j in range(len(factor)):
    f = 0
    A,B,c = get_coeff_atomic_factor(factor,j)
    for i in range(len(A)):
      f = f + A[i]*np.exp(-s*s*B[i])
    f = f + c
    F[j] = f
  return(F)
  
  
"""
***** Function matrix_f_atomic *****
Return the atomic diffusion factor for all atoms and all diffraction angles (linked with d).

Parameters:
- factor (pandas data.frame): Dataframe containing the coefficient for a large amount of atom
- d (np.array): vector with all values d of the distance between 2 atomic planes (deduce from theta using Bragg's law).

Ouutput:
f_diffusion (np.array): Matrix with atomic factor for all atoms and all values of d (theta).
"""
def matrix_f_atomic(d,facteur):
  f_diffusion = np.zeros(shape=(len(d)+1,len(facteur)))
  for j in range(len(facteur)):
    f_diffusion[0][j] = facteur['num_tab_periodique'][j]
  for i in range(len(d)):
    f_diffusion[i+1,:] = factor_atomic_diffusion(d[i],facteur)
  return f_diffusion


"""
***** Function  diffractogram *****
Return an XRD patterns using the .CIF

Parameters:
- lamb : X-Ray wavelength / longueur d'onde des rayons X
- LMH (float) : Width at half height / largeur a mi hauteur
- DW (vector,float) : Debye-Waller coefficicent / Coefficient de DebyeWaller
- a_param,b_param,c_param (float) : Lattice parameters / Parametres de norme de la maille
- Gauss (boolean) : TRUE -> Gaussian / Gaussienne / FALSE -> Lorentz / Lorentzienne
- cos/sin (float): cosinus and sinus for angle between lattice parameters (alpha, beta, gamma) / cos et sin des angles des parametres de maille
- d_min (float) : Minimum value of distance between two planes of atoms (according to theta angle) / d minimum déduit de l'angle theta de diffraction le plus petit
- num(vecteur,int) : Vector obtain with 'periodic_compt'  function, give the atomic number of the crystal's atoms / Vecteur obtenu avec fonction 'periodic_compt' qui permet d'avoir les numeros atomiques de atomes du cristal
- facteur : dataframe with coefficient for the atomic factor / table avec coefficient du facteur de diffusion atomique
- compt : (vecteur,int) : Vector obtain with 'periodic_compt'  function, give the number of atoms of each type / Vecteur obtenu avec fonction 'compteur periodique' qui permet d'avoir les nombres d'atomes de differents type du crystal
- Pos(Matrice_float) : Matrix containing the position of the atom in the unit-cell / Matrice avec les coordonnees de chaque atomes dans la maille
- theta_b(vecteur_float) : Diffraction angle range (5 -> 90°) / Plage des angles de diffraction
- M_hkl (Matrice_int) : Matrix containing initial (h,k,l) of the reciprocal space (after max_calculation) / Matrice avec les h,k et l initiaux (après calcul des max)
- v (float) : Numerator to calculate the inter-reticular distance / Numérateur de la distance inter-réticulaire
"""

def diffractogram(d,lamb,inc_theta,LMH,DW,M_hkl,v,a_param,b_param,c_param,cos_alpha,cos_beta,cos_gamma,sin_alpha,sin_beta,sin_gamma,cos_beta_star,d_min,num,facteur,compt,Pos,theta_b,Gauss=True):
  

  #Distance between two planes of atoms according to the matrix M_hkl
  D_inter_ret = np.sqrt(v/
    (((M_hkl[:,0]*M_hkl[:,0])/(a_param*a_param))*(sin_alpha*sin_alpha)
    +((M_hkl[:,1]*M_hkl[:,1])/(b_param*b_param))*(sin_beta*sin_beta)
    +((M_hkl[:,2]*M_hkl[:,2])/(c_param*c_param))*(sin_gamma*sin_gamma)
    - ((2*M_hkl[:,1]*M_hkl[:,2])/(b_param*c_param))*(cos_alpha-cos_beta*cos_gamma)
    - ((2*M_hkl[:,2]*M_hkl[:,0])/(a_param*c_param))*(cos_beta-cos_alpha*cos_gamma)
    - ((2*M_hkl[:,1]*M_hkl[:,0])/(b_param*a_param))*(cos_gamma-cos_beta*cos_alpha))
    )
  D_inter_ret[0] = 0


  #Associating the d values with the h,k,l corresponding values
  D_val_hkl = np.zeros(shape=(len(D_inter_ret),4))
  D_val_hkl[:,0] = D_inter_ret
  D_val_hkl[:,1:4] = M_hkl

  # Sort by descending order according to the d values
  D_val_hkl_tri=D_val_hkl[D_val_hkl[:,0].argsort()][::-1]

  # Remove the values that are not in the d range
  compteur = 0 
  D_hkl = np.zeros(shape=(len(D_val_hkl_tri),4))

  for i in range(len(D_val_hkl_tri)):
    if ( D_val_hkl_tri[i,0] < np.amax(d) and D_val_hkl_tri[i,0] > np.amin(d)):
      D_hkl[compteur] = D_val_hkl_tri[i]
      compteur = compteur + 1 

  D_hkl = D_hkl[0:compteur,:]

  #Find the theta values associated
  th_D_inter_ret = 2*np.arcsin((lamb/(2*D_hkl[:,0])))*180/np.pi


  """ Atomic factor diffusion """
  #Interpolation
  f_dif = np.zeros(shape=(len(D_hkl),len(num)))
  j = 0
  i = 0 
  while (j < len(num)):
    if (num[j] == facteur[0][i]):
      for k in range(len(D_hkl)):
        g1 = np.min(np.where(D_hkl[k][0]-d <= inc_theta))
        g2 = np.max(np.where(D_hkl[k][0]-d <= inc_theta))
        c_dir = (facteur[g1][i]-facteur[g2][i]) / (d[g1]-d[g2])
        b = facteur[g1][i] - c_dir*d[g1]
        f_dif[k,j] = c_dir*D_hkl[k][0]+b 
      j = j + 1
      i = 0
    else : i = i + 1

  """ DEBYE WALLER  """
  f_dif = Debye_w(DW,f_dif,th_D_inter_ret,lamb)

  """ STRUCTUR FACTOR """
  F_s = np.zeros(len(th_D_inter_ret))
  for i in range(len(th_D_inter_ret)):
    theta = th_D_inter_ret[i]
    inc_sfactor_1 = 0
    inc_sfactor_2 = 0
    c = 0

    lp = (1+np.cos(2*(theta*np.pi/360))*np.cos(2*(theta*np.pi/360)))/(np.sin(theta*np.pi/360))

    for z in range(len(compt)):
      for w in range(compt[z]):
        inc_sfactor_1 = inc_sfactor_1 + f_dif[i,z]*np.cos(2*np.pi*(D_hkl[i,1]*Pos[c+w,1]+D_hkl[i,2]*Pos[c+w,2]+D_hkl[i,3]*Pos[c+w,3]))
        inc_sfactor_2 = inc_sfactor_2 + f_dif[i,z]*np.sin(2*np.pi*(D_hkl[i,1]*Pos[c+w,1]+D_hkl[i,2]*Pos[c+w,2]+D_hkl[i,3]*Pos[c+w,3]))
      c = c + compt[z]
    inc_sfactor_1 = inc_sfactor_1 * inc_sfactor_1 
    inc_sfactor_2 = inc_sfactor_2 * inc_sfactor_2
    inc_sfactor = np.sqrt(inc_sfactor_1+inc_sfactor_2)

    inc_sfactor = inc_sfactor*lp
    F_s[i] = inc_sfactor
  
  #Matrix with theta and the corresponding Intensity
  Comb = np.zeros(shape=(len(th_D_inter_ret),2))
  Comb[:,0] = th_D_inter_ret
  Comb[:,1] = F_s
  

  #Sorts repeating occurrences
  compt_same_intensity = np.ones(len(Comb))
  Intensity = np.zeros(len(Comb)) 
  compt_intensity_diff = 0
  Intensity[compt_intensity_diff] = Comb[0,1]*Comb[0,1]
  k = round(Comb[0,0],ndigits=8)
  for i in range(len(Comb)-1):
    if (round(Comb[i+1,0],ndigits=8) == k):
      compt_same_intensity[compt_intensity_diff] = compt_same_intensity[compt_intensity_diff]+1
      Intensity[compt_intensity_diff] = Intensity[compt_intensity_diff] + Comb[i+1,1]*Comb[i+1,1]
    else :
      k = round(Comb[i+1,0],ndigits=8)
      compt_intensity_diff = compt_intensity_diff + 1
      Intensity[compt_intensity_diff] = Comb[i+1,1]*Comb[i+1,1]

  compt_same_intensity = compt_same_intensity[0:compt_intensity_diff+1]
  Intensity = Intensity[0:compt_intensity_diff+1]


  t_bis = np.zeros(shape=(compt_intensity_diff+1,2))
  t_bis[:,1] = Intensity
  k = 0
  for i in range(compt_intensity_diff+1):
    t_bis[i,0] = Comb[k,0]
    k = (int)(k + compt_same_intensity[i])

  
  #Gaussian distribution
  th = t_bis[:,0]
  Inte = t_bis[:,1]

  for i in range(len(theta_b)):
    theta_b[i] = round(theta_b[i],2)
  I = np.zeros(len(theta_b))
  
  if (Gauss == True):
      sigma = LMH/2*np.sqrt(2*np.log(2)) # Variance from Width at half heigth
      begin_peak = th-3*sigma
      end_peak = th+3*sigma
      n = (int)((end_peak[0]-begin_peak[0])/0.01)
      for i in range(len(th)):
        if (begin_peak[i] >= 5 and begin_peak[i] <= np.amax(theta)):
            t=np.min(np.where(np.abs(begin_peak[i]-theta_b)<inc_theta))
            mu = th[i]
            for j in range(n):
                if (t+j >= len(theta_b)):
                    break
                I[t+j] = I[t+j] + (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((theta_b[t+j]-mu)*(theta_b[t+j]-mu))/(2*sigma*sigma))*Inte[i]
 
  #Lorentz distribution
  else :
    begin_peak = th-3*LMH
    end_peak = th+3*LMH
    n = (int)((end_peak[0]-begin_peak[0])/0.01)
    for i in range(len(th)):
      if (begin_peak[i] >= 5 and begin_peak[i] <= np.amax(theta)):
          t=np.min(np.where(np.abs(begin_peak[i]-theta_b)<inc_theta))
          mu = th[i]
          for j in range(n):
            if (t+j >= len(theta_b)):
              break
            I[t+j] = I[t+j] + ((2/np.pi*LMH)/(1+((theta_b[t+j]-mu)/(LMH/2))))*Inte[i]
  
  return(I)
  
"""
*** FUNCTION WRITE ***
Function to write the Intensity in a txt file
PARAMETERS:
- I : Intensity describing the XRD pattern
- path : path
- name (string) : crystal name
- q (int) : to change the name
"""

def write(I,path,name,q,a,b,c,LMH,DW,Gauss):
  fichier = open(path+name+str(q)+'.txt',mode='w')
  fichier.write('a = '+ str(a) +' b = '+ str(b) + ' c = ' + str(c) + ' ,')
  if (Gauss == True):
    fichier.write('Gaussian ,')
  else : 
    fichier.write('Lorentzienne ,')
  fichier.write('LMH = ' + str(LMH) + ' ,')
  fichier.write('DW = ' + str(DW) + '\n')
  #Ecriture des donnees
  for i in range(len(I)):
    fichier.write(str(I[i])+'\n')
  fichier.close()
  return ()
 
 
"""
***** Function all_diffracto_cif *****

Parameters:
- path : path to the folder
- name(string) : crystal name (1st letter in capital)
- lamb(array) : vector with the X-rays wavelengths
- nb_diffracto (int) : number of XRD patterns required
- theta(vecteur_float) : vector with theta values (diffraction angles range)
- inc_LMH : increment width at half height
- inc_DW : increment Debye-Waller
- n_inc_param : increment number for lattice parameters variation (a,b,c)
- contrib (array) : Wavelengths contribution
"""
def all_diffracto_cif(path,name,theta,inc_theta,nb_diffracto,lamb,contrib,n_inc_param=50,inc_LMH=0.04,inc_DW=0.1):
  since = time.time()
  #Param du cristal
  crystal = Crystal.from_cif('AMS_DATA_'+name+'.cif')
  F = crystal.chemical_composition
  Pos = np.array(crystal)
  compt,num = periodic_compt(Pos,F)
  param = crystal.lattice_parameters
  a = param[0]
  b = param[1]
  c = param[2]
  alpha = param[3]
  beta = param[4]
  beta_star = np.pi - beta
  gamma = param[5]
  
  #Load dataframe with atomic diffusion coefficient
  factor = pd.read_csv('Coeff_dif_atomiquev3.txt',header=0,sep=',')

  #Cos/sin for lattice parameters
  cos_alpha = round(np.cos(alpha*np.pi/180),ndigits=8)
  cos_beta = round(np.cos(beta*np.pi/180),ndigits=8)
  cos_gamma = round(np.cos(gamma*np.pi/180),ndigits=8)
  sin_alpha = round(np.sin(alpha*np.pi/180),ndigits=8)
  sin_beta = round(np.sin(beta*np.pi/180),ndigits=8)
  sin_gamma = round(np.sin(gamma*np.pi/180),ndigits=8)
  cos_beta_star = round(np.sin(beta_star*np.pi/180),ndigits=8)

  
  #5% variation for a,b and c
  a_max = a + (a/100)*2
  a_min = a - (a/100)*2
  b_max = b + (b/100)*2
  b_min = b - (b/100)*2
  c_max = c + (c/100)*2
  c_min = c - (c/100)*2

  """
  - Random drawing of params a,b and c, taking into account the fact that in certain cases the mesh params must be equal
  - Random drawing of the DW coeff in a given interval: [0.1,2]
  - Draw width at half height in [0.08,0.50]
  - Pre-processing: atomic diffusion factor calculation and h,k,l values deduce from wavelength values
  """
  
  D = np.zeros(shape = (lamb.shape[0],theta.shape[0]))
  F = np.zeros(shape = (lamb.shape[0],len(theta) + 1,211))
  v = (1 - cos_alpha*cos_alpha - cos_beta*cos_beta - cos_gamma*cos_gamma+2 * cos_alpha * cos_beta *cos_gamma)
  x = 0
  
  for w in lamb:
    D[x,:] = w / (2*np.sin(theta*np.pi/360))
    F[x,:,:] = matrix_f_atomic(D[x,:],factor)
    x += 1
  
    
  #Print time calculation for initial parameters.
  time_elapsed = time.time()-since
  print('Initial parameters completed in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  
  #For each XRD patterns
  since = time.time()
  since_b = time.time()
  q = 1 #To write in .txt file
  for i in range(nb_diffracto):
    x = 0
    I = np.zeros(shape = len(theta))
    """ ADAPT the a,b and c values depending of the crystal """
    var_a = (a_max-a_min)*np.random.random_sample(1)+a_min
    var_b = var_a
    var_c = (c_max-c_min)*np.random.random_sample(1)+c_min

    Dw = (2-0.1)*np.random.random_sample(1)+0.1 #Random Debye-Waller
    DW = np.ones(len(num))*Dw
    LMH = (0.5-0.05)*np.random.random_sample(1)+0.05 #Random Width at half height
    for w in lamb:
        #Calculation: h,k et l_max
        h_m = (int) (np.sqrt(
            (v*var_a*var_a) / ((np.amin(D[x])**2)*(sin_alpha*sin_alpha)))) + 1
        l_m = (int)(np.sqrt(
            (v*var_c*var_c) / ((np.amin(D[x])**2)*(sin_gamma*sin_gamma)))) + 1
        k_m = (int)(np.sqrt(
            (v*var_b*var_b) / ((np.amin(D[x])**2)*(sin_beta*sin_beta)))) + 1
        
        #Matrix h,k,l from h,k et l_max
        M_hkl = np.zeros(shape=((h_m*2+1)*(k_m*2+1)*(l_m*2+1),3))
        z = 0
        for i in np.arange(-h_m,h_m+1):
            for j in np.arange(-k_m,k_m+1):
                for k in np.arange(-l_m,l_m+1):
                    M_hkl[z,:] = [i,j,k]
                    z = z + 1
    
        I1 = diffractogram(D[x],w,inc_theta,LMH,DW,M_hkl,v,var_a,var_b,var_c,cos_alpha,cos_beta,cos_gamma,sin_alpha,sin_beta,sin_gamma,cos_beta_star,np.amin(D[x]),num,F[x],compt,Pos,theta)
        I = I + contrib[x]*I1
        x = x + 1
    # Write the intensity in a file
    write(I,path,name,q,a = var_a,b = var_b ,c = var_c ,LMH = LMH, DW = Dw ,Gauss = True)
          
    if (q % 100 == 0):
              time_elapsed_b = time.time()-since_b
              print(str(q)+' diffractogrammes completed in {:.0f}m {:.0f}s'.format(
                  time_elapsed_b // 60, time_elapsed_b % 60))
    q = q + 1

            
  #Print time
  time_elapsed = time.time()-since
  print(str(q-1)+' diffractogrammes completed in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))

      
 
 
 
""" APPLICATION """

theta_min = 4.0001
theta_max = 90.020055
inc_theta = 0.029611
theta = np.arange(theta_min,theta_max,inc_theta)

path = 'Pur_XRD/'
name ='Calcite'  # ENTER CRYSTAL NAME FIRST CAPITAL LETTER ex: Calcite, Gibbsite, Pyrite, ....
lamb = np.array([1.5348,1.5406,1.5411,1.5444,1.5447,1.3923]) # Adapt to the device -> Wavelength function
contrib = np.array([0.01586,0.56768,0.07601,0.25107,0.08688,0.00249]) # Adapt to the device -> Wavelength function

all_diffracto_cif(path = path ,name = name ,theta = theta ,inc_theta = inc_theta, nb_diffracto = 1500, lamb = lamb, contrib = contrib)

