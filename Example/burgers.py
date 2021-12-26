import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import random
from sklearn.metrics import mean_squared_error
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from functions import data_normalize, flatten, DE_Learner, split_final, data_split, train_test_split, multi_heatmap
matplotlib.rcParams.update({'font.size': 14})
output ='./burgers/'
matplotlib.use ( 'Agg' )
width, height = 1., 1.
dx = dy = .01
nx = int(width/dx)+1
ny = int(height/dy)+1
nt = 1000
sigma = 10
nu = 0.0
dt = sigma * dx * dy #/ nu
x = np.linspace ( 0.0, width, nx )
y = np.linspace ( 0.0, height, ny )
X, Y = np.meshgrid ( x, y )
u = np.zeros ( ( ny, nx ) )
v = np.zeros ( ( ny, nx ) )
u[nx//4:nx//4*3,ny//4:ny//4*3] = .9
v[nx//4:nx//4*3,ny//4:ny//4*3] = .5
#plot the initial conditions
fig = plt.figure ( figsize = ( 11, 7 ), dpi = 100 )
ax = fig.gca ( projection = '3d' )
wire1 = ax.plot_wireframe ( X, Y, u[:], cmap = cm.coolwarm )
wire2 = ax.plot_wireframe ( X, Y, v[:], cmap = cm.coolwarm )
ax.set_title('Initial Condition')
#save the time series data:
u_dat = [[] for _ in range(nx*ny)]
v_dat = [[] for _ in range(nx*ny)]
k = 0
for i in range(ny):
    for j in range(nx):
        u_dat[k].append(u[i,j])
        v_dat[k].append(v[i,j])
        k+=1

for n in range ( nt + 1 ):
  un = u.copy ( )
  vn = v.copy ( )
  u[1:-1,1:-1] = un[1:-1,1:-1] \
    - dt / dx * un[1:-1,1:-1] * ( un[1:-1,1:-1] - un[1:-1,0:-2] ) \
    - dt / dy * vn[1:-1,1:-1] * ( un[1:-1,1:-1] - un[0:-2,1:-1] ) \
    + nu * dt / dx ** 2 * ( un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2] ) \
    + nu * dt / dy ** 2 * ( un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1] )
  v[1:-1,1:-1] = vn[1:-1,1:-1] \
    - dt / dx * un[1:-1,1:-1] * ( vn[1:-1,1:-1] - vn[1:-1,0:-2] ) \
    - dt / dy * vn[1:-1,1:-1] * ( vn[1:-1,1:-1] - vn[0:-2,1:-1] ) \
    + nu * dt / dx ** 2 * ( vn[1:-1,2:] - 2 * vn[1:-1,1:-1] + vn[1:-1,0:-2] ) \
    + nu * dt / dy ** 2 * ( vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1] )
  u[0,:] = 0
  u[-1,:] = 0
  u[:,0] = 0
  u[:,-1] = 0
  v[0,:] = 0
  v[-1,:] = 0
  v[:,0] = 0
  v[:,-1] = 0
  k = 0
  for i in range(ny):
    for j in range(nx):
        u_dat[k].append(u[i,j])
        v_dat[k].append(v[i,j])
        k+=1
u_dat = np.array(u_dat)
v_dat = np.array(v_dat)
fig = plt.figure ( figsize = ( 11, 7 ), dpi = 100 )
ax = fig.gca ( projection = '3d' )
wire1 = ax.plot_wireframe ( X, Y, u )
wire2 = ax.plot_wireframe ( X, Y, v )
ax.set_title('Final Time step t =: '+str(dt*nt))
plt.show()
plt.figure()
plt.imshow(u)
plt.figure()
plt.imshow(v)
print('dt: ', dt)
print(u_dat.shape, v_dat.shape)
t1=30
t2=10
print('training time period: ', dt*t1)
print('test time period: ', dt*t2)
burgers_data = np.vstack((u_dat, v_dat))
t1, t2 = 30,10
final_x, final_y, burgers_data = split_final(burgers_data,t1,t2)
name = 'burgers'
print(final_x.shape, final_y.shape)
N = 20
burger_py, burger_ty, burger_err, burgermodel = DE_Learner(burgers_data, t1, t2, 1 , .2, N, name, plot = 1)
burgermodel.save("burgersmodel")