import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import cm
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None

 # NB: the HFIT for nucleus 1 is TrpH_HFIT[0] and so on... hence for N9 want TrpH_HFITs[8] and so on

TrpH_HFITs = np.load('TrpH_HFITS_MHz.npy')

# NB: Use the 'Standard Orientations' from the Gaussian file, not the 'Input Orientations'

atomicpos = pd.read_csv('TrpH Atomic Positions.csv')

# We first carry out some rotations

# Define a vector containing the coordinates of atom N

def Coordinate_Vector(N):
    return np.array([atomicpos['x'][N-1], atomicpos['y'][N-1], atomicpos['z'][N-1]])

# Find a vector to the center of the benzene ring

Center_Vector = 0
for N in [12,11,10,14,14,13]:
    Center_Vector += Coordinate_Vector(N)
Center_Vector = Center_Vector/6

# Making the rotation matrix to make the z-axis perpendicular to the benzenoid ring:

# A function to normalize a vector
def Normalize(v):
    Norm = np.linalg.norm(v)
    if Norm == 0:
       return v
    return v / Norm

# Make a (normalized) vector connecting atoms N1 and N2
def Bond_Vector(N1,N2):
    v = Coordinate_Vector(N1) - Coordinate_Vector(N2)

    return Normalize(v)

# Find a vector perpendicular to two ring-diagonals

fgz = Normalize(np.cross(Bond_Vector(11,14), Bond_Vector(10,15)))

# Making our new basis vectors...
fgx = Normalize(Bond_Vector(13,12))
fgy = np.cross(fgz,fgx)
fgx = np.cross(fgy,fgz)

# Thus our rotation matrix is:
Rfg = np.array([fgx, fgy, fgz])
Rfg_transpose = np.transpose(Rfg)

# Transforming our coordinates such that the benzene ring is perpendicular to the z-axis and (0,0,0) is the center of the ring

def Coordinate_Vector_Rotated(N):
    return np.dot(Rfg, (Coordinate_Vector(N) - Center_Vector))

def Bond_Vector_Rotated(N1, N2):
    return Coordinate_Vector_Rotated(N1) - Coordinate_Vector_Rotated(N2)

# Defining the surface of a given Nuclei's HFI

def Surface(N, resolution, scale):

    # Create a mesh-grid of the desired resolution

    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2*np.pi, resolution)
    thetaGrid, phiGrid = np.meshgrid(theta, phi)

    # Calculate the surface for a given Nuclei's HFI

    # First we make a (resolution x resolution) array of q-vectors for each value of (theta, phi) (with {x,y,z} in three sheets)

    q = np.array([np.sin(thetaGrid)*np.cos(phiGrid), np.sin(thetaGrid)*np.sin(phiGrid), np.cos(thetaGrid)])

    vectorarray = np.empty((resolution, resolution, 3))

    for i in range(resolution):
        for j in range(resolution):
            vectorarray[i,j] = np.array([q[0,i,j], q[1,i,j], q[2,i,j]])


    r_array = np.empty((resolution,resolution))

    for i in range(resolution):
        for j in range(resolution):
            r_array[i,j] = scale*np.dot(vectorarray[i,j], np.dot(Rfg, np.dot(TrpH_HFITs[N-1], np.dot(Rfg_transpose,  vectorarray[i,j]))))


    X = r_array*np.sin(thetaGrid)*np.cos(phiGrid)
    Y = r_array*np.sin(thetaGrid)*np.sin(phiGrid)
    Z = r_array*np.cos(thetaGrid)

    return X, Y, Z

def Plot_Surface(N, resolution, scale, color = 'green', x_manual = 0, y_manual = 0, z_manual = 0):
    X = Surface(N,resolution,scale)[0]
    Y = Surface(N,resolution,scale)[1]
    Z = Surface(N,resolution,scale)[2]

    ax.plot_surface(X + Coordinate_Vector_Rotated(N)[0] + x_manual, Y + Coordinate_Vector_Rotated(N)[1] + y_manual, Z + Coordinate_Vector_Rotated(N)[2] + z_manual, color = color)


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

# We draw a line between atoms N1 and N2 (actually drawing two lines so can show what type of atoms are connected by that bond)

def Draw_Line(N1, N2, color1 = 'black', color2 = 'black'):
    midpoint_x = (Coordinate_Vector_Rotated(N1)[0] + Coordinate_Vector_Rotated(N2)[0]) / 2.0
    midpoint_y = (Coordinate_Vector_Rotated(N1)[1] + Coordinate_Vector_Rotated(N2)[1]) / 2.0
    midpoint_z = (Coordinate_Vector_Rotated(N1)[2] + Coordinate_Vector_Rotated(N2)[2]) / 2.0


    x = np.array([Coordinate_Vector_Rotated(N1)[0], Coordinate_Vector_Rotated(N2)[0]])
    y = np.array([Coordinate_Vector_Rotated(N1)[1], Coordinate_Vector_Rotated(N2)[1]])
    z = np.array([Coordinate_Vector_Rotated(N1)[2], Coordinate_Vector_Rotated(N2)[2]])

    x1 = np.array([Coordinate_Vector_Rotated(N1)[0], midpoint_x])
    y1 = np.array([Coordinate_Vector_Rotated(N1)[1], midpoint_y])
    z1 = np.array([Coordinate_Vector_Rotated(N1)[2], midpoint_z])

    x2 = np.array([midpoint_x, Coordinate_Vector_Rotated(N2)[0]])
    y2 = np.array([midpoint_y, Coordinate_Vector_Rotated(N2)[1]])
    z2 = np.array([midpoint_z, Coordinate_Vector_Rotated(N2)[2]])

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
ax.set_box_aspect((np.ptp([Coordinate_Vector_Rotated(N)[0] for N in range(1,28)]), np.ptp([Coordinate_Vector_Rotated(N)[1] for N in range(1,28)]), np.ptp([Coordinate_Vector_Rotated(N)[2] for N in range(1,28)]))) # This fixes the aspect ratio (note: np.ptp(array) returns max(array) - min(array))



Draw_Skeleton()

Plot_Carbons()
Plot_Hydrogens()
Plot_Nitrogens()
Plot_Oxygens()


#plt.axis('off')
plt.show()