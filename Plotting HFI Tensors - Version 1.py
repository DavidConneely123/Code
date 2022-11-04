import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import cm
import plotly.graph_objects as go

TrpH_HFITs = np.load('TrpH_HFITS_MHz.npy')   # NB: the HFIT for nucleus 1 is TrpH_HFIT[0] and so on... hence for N9 want TrpH_HFITs[8] and so on
atomicpos = pd.read_csv('TrpH Atomic Positions.csv') # Standard Orientation

def Surface(N, resolution,scale):

    # Create a mesh-grid of the desired resolution

    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2*np.pi, resolution)
    thetaGrid, phiGrid = np.meshgrid(theta, phi)

    # Calculate the surface for a given Nuclei's HFI

    q = np.array([np.sin(thetaGrid)*np.cos(phiGrid), np.sin(thetaGrid)*np.cos(phiGrid), np.cos(thetaGrid)])

    vectorarray = np.empty((resolution, resolution, 3))

    for i in range(resolution):
        for j in range(resolution):
            vectorarray[i,j] = np.array([q[0,i,j], q[1,i,j], q[2,i,j]])


    r_array = np.empty((resolution,resolution))

    for i in range(resolution):
        for j in range(resolution):
            r_array[i,j] = scale*np.dot(vectorarray[i,j], np.dot(TrpH_HFITs[N-1], vectorarray[i,j]))


    X = r_array*np.sin(thetaGrid)*np.cos(phiGrid)
    Y = r_array*np.sin(thetaGrid)*np.sin(phiGrid)
    Z = r_array*np.cos(thetaGrid)

    return X, Y, Z

def Plot_Surface(N, resolution, scale, color = 'green'):
    X = Surface(N,resolution,scale)[0]
    Y = Surface(N,resolution,scale)[1]
    Z = Surface(N,resolution,scale)[2]

    ax.plot_surface(X + atomicpos['x'][N-1], Y + atomicpos['y'][N-1], Z + atomicpos['z'][N-1], color = color)

def Plot_Carbons():
    for N in [8,7,11,10,12,13,14,15, 5, 4, 2]:
        Plot_Surface(N, 20, 0.02, color = 'blue')

def Plot_Hydrogens():
    for N in [24, 27, 26, 25, 23, 22, 19, 18, 21, 20, 17, 16]:
        Plot_Surface(N, 20, 0.02, color = 'green')

def Plot_Nitrogens():
    for N in [9,6]:
        Plot_Surface(N, 20, 0.02, color = 'red')

def Plot_Oxygens():
    for N in [1,3]:
        Plot_Surface(N, 20, 0.02, color = 'magenta')

def Draw_Line(N1, N2, color1 = 'black', color2 = 'black'):
    midpoint_x = (atomicpos['x'][N2 - 1] + atomicpos['x'][N1-1]) / 2.0
    midpoint_y = (atomicpos['y'][N2 - 1] + atomicpos['y'][N1 - 1]) / 2.0
    midpoint_z = (atomicpos['z'][N2 - 1] + atomicpos['z'][N1 - 1]) / 2.0


    x = np.array([atomicpos['x'][N1-1], atomicpos['x'][N2-1]])
    y = np.array([atomicpos['y'][N1-1], atomicpos['y'][N2 - 1]])
    z = np.array([atomicpos['z'][N1-1], atomicpos['z'][N2 - 1]])

    x1 = np.array([atomicpos['x'][N1-1], midpoint_x])
    y1 = np.array([atomicpos['y'][N1-1], midpoint_y])
    z1 = np.array([atomicpos['z'][N1-1], midpoint_z])

    x2 = np.array([midpoint_x, atomicpos['x'][N2-1]])
    y2 = np.array([midpoint_y, atomicpos['y'][N2-1]])
    z2 = np.array([midpoint_z, atomicpos['z'][N2-1]])

    ax.plot3D(x1, y1, z1, color = color1)
    ax.plot3D(x2, y2, z2, color = color2)


def Draw_Skeleton():
    # Drawing C-C Bonds

    for bond in [[10, 11], [11, 12], [12, 15], [15, 14], [14, 13], [13, 10], [8, 7], [7, 11], [7, 5], [5, 4], [4, 2]]:
        Draw_Line(bond[0], bond[1], color1='blue', color2='blue')

    # Drawing C-N Bonds
    for bond in [[10, 9], [8, 9], [4, 6]]:
        Draw_Line(bond[0], bond[1], color1='blue', color2='red')

    # Drawing C-H Bonds
    for bond in [[12, 24], [15, 27], [14, 26], [13, 25], [8, 22], [5, 19], [5, 18], [4, 17]]:
        Draw_Line(bond[0], bond[1], color1='blue', color2='green')

    # Drawing N-H Bonds
    for bond in [[9, 23], [6, 21], [6, 20]]:
        Draw_Line(bond[0], bond[1], color1='red', color2='green')

    # Drawing C-O Bonds
    for bond in [[2, 1], [2, 3]]:
        Draw_Line(bond[0], bond[1], color1='blue', color2='magenta')

    # Drawing O-H Bonds
    for bond in [[1, 16]]:
        Draw_Line(bond[0], bond[1], color1='magenta', color2='green')


fig = plt.figure()
ax = fig.gca(projection='3d')


Draw_Skeleton()

Plot_Carbons()
#Plot_Hydrogens()
Plot_Nitrogens()
#Plot_Oxygens()

plt.axis('off')
plt.show()