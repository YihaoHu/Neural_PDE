import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import random
from sklearn.metrics import mean_squared_error
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from model import data_normalize, flatten, DE_Learner, split_final, data_split, train_test_split, multi_heatmap
matplotlib.rcParams.update({'font.size': 14})
output ='./NSCH/'

file_name = './cahn_hilliard_ns/'
phi =[[] for _ in range(4385)]
p =[[] for _ in range(4385)]
u1 = [[] for _ in range(4385)]
u2 = [[] for _ in range(4385)]
data_len = 2000
for frame in range(1, data_len):
    f1 = open(file_name + 'movie_phi_'+str(10000+frame)+'.txt')
    f2 = open(file_name + 'movie_pressuer_'+str(10000+frame)+'.txt')
    f3 = open(file_name + 'movie_u1_'+str(10000+frame)+'.txt')
    f4 = open(file_name + 'movie_u2_'+str(10000+frame)+'.txt')
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()
    lines4 = f4.readlines()
    #print(len(lines1),len(lines2),len(lines3),len(lines4))
    for i in range(len(phi)):
        phi[i].append(float(lines1[i].strip()))
        p[i].append(float(lines2[i].strip()))
        u1[i].append(float(lines3[i].strip()))
        u2[i].append(float(lines4[i].strip()))

system = phi + data_normalize(p)
#system = data_normalize(system)
t1 = 30; t2 = 10

#select last data for prediction plot
print(len(system),len(system[0]))
sys_final_x = [d[-t1-t2:-t2] for d in system]
sys_final_y = [d[-t2:] for d in system]
sys_final_x = np.array(sys_final_x).reshape(1, len(sys_final_x),len(sys_final_x[0]))
sys_final_y = np.array(sys_final_y).reshape(1, len(sys_final_y),len(sys_final_y[0]))
system = [d[:-t1-t2] for d in system]
system = np.array(system)
N = 50
system_py, system_ty, system_err, systemmodel = DE_Learner(system, t1, t2, 1 , .2, N, 'NSCH_p', plot = 1)
systemmodel.save("NSCHmodel")
