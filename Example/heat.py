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
output ='./heat/'
#sns.set(font_scale = 2)
def heat2d(dxdy, X, T, plot = 0):
    w = h = X
    dx = dy = dxdy
    D = 1.
    Tcool, Thot = .1, .9
    nx, ny = int(w/dx), int(h/dy)
    dx2, dy2 = dx*dx, dy*dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
    #D = (dx2 * dy2 / dt) / (2 * (dx2 + dy2))
    print('dt: ', dt)
    u0 = Tcool * np.ones((nx, ny))
    u = u0.copy()

    # Initial conditions 
    r, cx, cy = .5, 1, 1
    r2 = r**2
    heat_dat = [[] for _ in range(int(nx*ny))]
    for i in range(nx):
        for j in range(ny):
            p2 = (i*dx-cx)**2 + (j*dy-cy)**2
            if p2 < r2:
                u0[i,j] = Thot
    k = 0
    for i in range(nx):
        for j in range(ny):
            heat_dat[k].append(u0[i,j])
            k+= 1
    if plot:
        plt.figure()
        plt.imshow(u0)
        plt.title('Initial Condition')
    def do_timestep(u0, u):
        #central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
              (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
              + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

        u0 = u.copy()
        return u0, u
    time_step = int(T / dt)
    for steps in range(time_step+1):
        u0, u = do_timestep(u0, u)
        k = 0
        for i in range(nx):
            for j in range(ny):
                heat_dat[k].append(u0[i,j])
                k+= 1
    if plot:
        plt.figure()
        plt.imshow(u0)
        plt.title('Final State T = : '+ str(dt*time_step))
        print('Time series data size: ', len(heat_dat), len(heat_dat[0]))
    return np.array(heat_dat)
import keras
heatmodel = keras.models.load_model('heatmodel')

T = 0.15
X = 2.0
dx = 0.02
#seq2seq train
t1, t2 = 30, 10
N = 20
heat_dat = heat2d(dx, X, T, 0)
name = 'heat'
heat_py, heat_ty, heat_err, heatmodel = DE_Learner(heat_dat, t1, t2, 1 , .2, N, name, plot = 1)
heatmodel.save("heatmodel")
