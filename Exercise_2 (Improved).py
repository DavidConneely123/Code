# Importing the standard libraries and defining the required matrix reps. 

import math 
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from datetime import datetime
startTime = datetime.now()

i2 = np.array([[1 + 0j,0 + 0j] , [0 + 0j, 1 + 0j]])
sz = np.array([[0.5 + 0j,0 + 0j] , [0 + 0j, -0.5 + 0j]])
sy = np.array([[0 + 0j, 0 - 0.5j] , [0 + 0.5j, 0 + 0j]])
sx = np.array([[0 + 0j,0.5 + 0j] , [0.5 + 0j, 0 + 0j]])

def matrix_rep(electronA, electronB, nucleus):
    return np.kron(electronA, np.kron(electronB, nucleus))

S_az = matrix_rep(sz,i2,i2)
S_bz = matrix_rep(i2,sz,i2)

S_ay = matrix_rep(sy,i2,i2)
S_by = matrix_rep(i2,sy,i2)

S_ax = matrix_rep(sx,i2,i2)
S_bx = matrix_rep(i2,sx,i2)

S_axI_x = matrix_rep(sx,i2,sx)
S_ayI_y = matrix_rep(sy,i2,sy)
S_azI_z = matrix_rep(sz,i2,sz)

S_aI = S_axI_x + S_ayI_y + S_azI_z

S_axS_bx = matrix_rep(sx,sx,i2)
S_ayS_by = matrix_rep(sy,sy,i2)
S_azS_bz = matrix_rep(sz,sz,i2)

i8 = np.kron(i2, np.kron(i2,i2))

Ps = 0.25*i8 - S_axS_bx - S_ayS_by - S_azS_bz

# Expressing everything in frequency units

a = 1 # NB a=1 mT
gyromag = 1.76e+8 # s^-1 mT^-1

# Defining the Zeeman Hamiltonian

def H_zee(field_strength, theta, phi=0):   #NB in s^-1 , field_strength should be in mT
      return gyromag*field_strength*((S_ax + S_bx)*math.sin(theta)*math.cos(phi) + (S_ay + S_by)*math.sin(theta)*math.sin(phi) + (S_az+S_bz)*math.cos(theta))

# Defining the Axiality and Rhombicity and the Hyperfine Hamiltonian


def H_hf(axiality, rhombicity):
    
    T_xx = gyromag*a*axiality*(1-rhombicity)
    T_yy = gyromag*a*axiality*(1+rhombicity)
    T_zz = -2*gyromag*a*axiality                     #T_ij are no in s^-1
    
    return (gyromag*a+T_xx)*S_axI_x + (gyromag*a+T_yy)*S_ayI_y + (gyromag*a+T_zz)*S_azI_z #Now completely in s^-1

def H_total(field_strength, theta, axiality, rhombicity, phi=0):
    return H_zee(field_strength, theta, phi=0) + H_hf(axiality,rhombicity)


# Calculating the singlet yield

def singlet_yield(rate_constant, field_strength, theta, axiality, rhombicity, phi=0):
    Hamiltonian = H_total(field_strength, theta, axiality, rhombicity, phi=0)
    eigenvalues, eigenvectors_A = np.linalg.eig(Hamiltonian)
    A_hermitian_conjugate = np.linalg.inv(eigenvectors_A)
    Ps_transformed = np.dot(A_hermitian_conjugate , np.dot(Ps,eigenvectors_A ))
    
    sum = 0
    
    for m in range(8):
        for j in range(8):
            sum += 1/2 *((rate_constant**2) / (rate_constant**2 + (eigenvalues[m] - eigenvalues[j])**2))*(Ps_transformed[j,m]**2)
            
    return sum.real

# Producing Figure 1

import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

pl.figure()
ax = pl.subplot(projection='3d')

x = np.linspace(0, 2, 1000)
y = np.zeros(x.size)
 
for theta in np.linspace(0, math.pi / 2, 17):
    x = np.linspace(0,2,1000)
    y = y + (math.pi/16)*np.ones(x.size)
    z = [singlet_yield(2.801e6, i, theta, -0.3, 0) for i in x]

    ax.plot(x,y,z)
    
    ax.set_xlabel('w / a ')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Singlet yield')
    ax.view_init(elev=10, azim=-130)

plt.show(block=False)
plt.pause(0.001)
#   Calculating the rotationally averaged singlet yield
def rotational_average(rate_constant, field_strength, axiality, rhombicity, phi=0):
    return (integrate.quad(lambda theta: math.sin(theta)*singlet_yield(rate_constant, field_strength, theta, axiality, rhombicity, phi=0), 0, math.pi)[0]) / 2

# Reaction yield anisotropy

def singlet_yield_norm(rate_constant , field_strength, theta, axiality, rhombicity, phi =0):

     return singlet_yield(rate_constant, field_strength, theta, axiality, rhombicity, phi=0) - rotational_average(rate_constant, field_strength, axiality, rhombicity, phi=0)

#Converting to spherical coordinates and plotting
from matplotlib import cm

B_o =4.9/1000 # in mT

def sph2cart(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

mesh_size = 30

phi = np.linspace(0, 2 * math.pi, mesh_size)
theta = np.linspace(0, math.pi, mesh_size)
phi, theta = np.meshgrid(phi, theta)

myfunc_vec = np.vectorize(singlet_yield_norm)
A = myfunc_vec(2e5, B_o, theta, -0.02, 0)

x, y, z = sph2cart(np.abs(A), phi, theta)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1, specs = [[{'is_3d': True}]])

fig.add_trace(go.Surface(x=x , y=y, z=z, surfacecolor=A), 1, 1)
fig.update_layout(title_text=f'Magnetic field strength = {B_o} mT')
fig.show()


print(datetime.now() - startTime)
plt.show()

