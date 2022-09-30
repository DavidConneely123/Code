
# Importing modules

import math
import numpy as np
import matplotlib.pyplot as plt


# Defining the basic (2x2) matrix representations:
i2 = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])
sz = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])
sy = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, -0 + 0j]])
sx = np.array([[0 + 0j, 0.5 - 0j], [0.5 + 0j, 0 + 0j]])



# Calculate the required matrix representations of the operators (8x8 matrices)

def matrix_rep(electronA, electronB, nucleus):
    return np.kron(electronA, np.kron(electronB, nucleus))


S_az = matrix_rep(sz, i2, i2)
S_bz = matrix_rep(i2, sz, i2)

S_ay = matrix_rep(sy, i2, i2)
S_by = matrix_rep(i2, sy, i2)

S_ax = matrix_rep(sx, i2, i2)
S_bx = matrix_rep(i2, sx, i2)

S_axI_x = matrix_rep(sx, i2, sx)
S_ayI_y = matrix_rep(sy, i2, sy)
S_azI_z = matrix_rep(sz, i2, sz)

S_aI = S_axI_x + S_ayI_y + S_azI_z

S_axS_bx = matrix_rep(sx, sx, i2)
S_ayS_by = matrix_rep(sy, sy, i2)
S_azS_bz = matrix_rep(sz, sz, i2)

 i8 = np.kron(i2, np.kron(i2, i2))

# Matrix representation of the initial (singlet) state of the radical pair

Ps = 0.25 * i8 - S_axS_bx - S_ayS_by - S_azS_bz



# Define our Zeeman Hamiltonian (NB: beta=0 so we set phi=0 as there is no dependence, could just not include it
# and use simplified form given on sheet but might be nice to have later if we introduce rhombicity)

def H_zee(w, theta, phi=0):
    return w * ((S_ax + S_bx) * math.sin(theta) * math.cos(phi) + (S_ay + S_by) * math.sin(theta) * math.sin(phi) + (
                S_az + S_bz) * math.cos(theta))



# Defining the Hyperfine Hamiltonian:

def H_hf(axiality, rhombicity):
    T_xx = a * axiality * (1 - rhombicity)
    T_yy = a * axiality * (1 + rhombicity)
    T_zz = -2 * a * axiality

    return (a + T_xx) * S_axI_x + (a + T_yy) * S_ayI_y + (a + T_zz) * S_azI_z



# Defining the Total Hamiltonian

def H_total(w, theta, axiality, rhombicity, phi=0):
    return H_zee(w, theta, phi=0) + H_hf(axiality, rhombicity)



# Calculating the Singlet yield using the same method as in exercise 1

a = 1
k = a / (20 * math.pi)


def singlet_yield(k, w, theta, axiality, rhombicity, phi=0):
    Hamiltonian = H_total(w, theta, axiality, rhombicity, phi=0)
    eigenvalues, eigenvectors_A = np.linalg.eig(Hamiltonian)
    A_hermitian_conjugate = np.linalg.inv(eigenvectors_A)
    Ps_transformed = np.dot(A_hermitian_conjugate, np.dot(Ps, eigenvectors_A))

    sum = 0

    for m in range(8):
        for j in range(8):
            sum += 1 / 2 * ((k ** 2) / (k ** 2 + (eigenvalues[m] - eigenvalues[j]) ** 2)) * (Ps_transformed[j, m] ** 2)

    return sum.real



# Conversion factors etc... NB: conversion is 1 mT => 28 MHz (Hence for P2, k = a/140) and we have axiality = -0.02
# and rhombicity = 0 (from the HFI tensor)

# Calculating the rotationally averaged singlet yield:

import scipy.integrate as integrate


def rotational_average(k, w, axiality, rhombicity, phi=0):
    return (integrate.quad(lambda theta: math.sin(theta) * singlet_yield(k, w, theta, axiality, rhombicity, phi=0), 0,
                           math.pi)[0]) / 2


# Calculating normalised directional singlet yield:

def singlet_yield_norm(k, w, theta, axiality, rhombicity, phi=0):
    return singlet_yield(k, w, theta, axiality, rhombicity, phi=0) - rotational_average(k, w, axiality, rhombicity,
                                                                                        phi=0)

# Plotting the yield anisotropies


def sph2cart(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


phi = np.linspace(0, 2 * math.pi, 25)
theta = np.linspace(0, math.pi, 25)

phi, theta = np.meshgrid(phi, theta)

myfunc_vec = np.vectorize(singlet_yield_norm)
Y = myfunc_vec(a / 500,  58 / 1000 * a, theta, -0.02, 0)

Y1 = Y[0]
Y2 = Y1[0]

x, y, z = sph2cart(np.abs(Y), phi, theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


from matplotlib import cm

ax.plot_surface(x, y, z, linewidth=0.5, facecolors=cm.jet(abs(Y) * 1 / Y2), edgecolors='red')

print('done')
plt.show()













