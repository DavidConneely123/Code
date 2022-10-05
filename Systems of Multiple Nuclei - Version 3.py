import numpy as np
import scipy
import scipy.linalg
from scipy import sparse
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.sparse import linalg

startTime = datetime.now()

#Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:

i2 = np.identity(2)
sx_spinhalf = np.array([[0 + 0j,0.5 + 0j] , [0.5 + 0j, 0 + 0j]]).astype(complex)
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j] , [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j,0 + 0j] , [0 + 0j, -0.5 + 0j]])

i3 = np.identity(3)
sx_spin1 = 1/2*np.sqrt(2)*np.array([[0,1,0] , [1,0,1], [0,1,0]])
sy_spin1 = 1/2*np.sqrt(2)*np.array([[0, -1j,0] , [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = 1/2*np.sqrt(2)*np.array([[1,0,0], [0,0,0] , [0,0,-1]])

#Defining a function that allows us to take an arbitrary number of Kronecker products
#NB: Creates sparse matrices

def matrix_representation(spins):
    for index, spin in enumerate(reversed(spins)):

        if index <= 1:
            current_loop = scipy.sparse.kron(spins[-2], spins[-1])

        else:
            current_loop = scipy.sparse.kron(spin, current_loop, format='csr')

    return current_loop


#Some useful accounting and indexing (electronA, electronB, NA1,...)

number_of_nuclei_included_A = 10
number_of_spins_included_A = number_of_nuclei_included_A + 2

spin_half_or_spin_1_list_ALLSPINS = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
spin_half_or_spin_1_list_ACTIVE = spin_half_or_spin_1_list_ALLSPINS[0:number_of_spins_included_A]

identity_list=[]
for index, value in enumerate(spin_half_or_spin_1_list_ACTIVE):
    if value == 1:
        identity_list.append(i3)
    else:
        identity_list.append(i2)

matrix_size = 1
for index, value in enumerate(spin_half_or_spin_1_list_ACTIVE):
    if value == 1:
        matrix_size *= 3
    else:
        matrix_size *= 2

# Inputting the values required to scale the matrix representations before adding them to the Hamiltonian

gyromag = 1.76e+8 # s^-1 mT^-1
field_strength = 0.05 #in mT (50 uT)

Zeeman_layerA = gyromag*field_strength*np.array([[0,0,0],[0,0,0], [1/3, 1/3, 1/3]]) #NB NOT SURE ABOUT THIS.... SHOULD IT BE 1/3 (ADDING THREE TIMES!)
Zeeman_layerB = gyromag*field_strength*np.array([[0,0,0],[0,0,0], [1/3, 1/3, 1/3]])
A1_layer = np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]])*1e6 #N5
A2_layer = np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00],[0.00, 0.00, 16.94]])*1e6 #N10
A3_layer = np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]])*1e6 #H8_1
A4_layer = np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]])*1e6 #H8_2
A5_layer = np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]])*1e6 #H8_3
A6_layer = np.array([[11.41, 0.00, 0.00], [0.00, 11.41, 0.00], [0.00, 0.00, 11.41]])*1e6 #H1'
A7_layer = np.array([[11.41, 0.00, 0.00], [0.00, 11.41, 0.00], [0.00, 0.00, 11.41]])*1e6 #H1''
A8_layer = np.array([[-5.63, 0.92, 0.00], [0.92, -14.77, 0.00], [0.00, 0.00, -12.15]])*1e6 #H6
A9_layer = np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6 #H7_1
A10_layer = np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6 #H7_2
A11_layer = np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6 #H7_3
A12_layer = np.array([[1.88, -0.71, 0.00], [-0.71, 3.01, 0.00], [0.00, 0.00, -0.14]])*1e6 #H9
A13_layer = np.array([[-1.21, 0.00, 0.00], [0.00, -0.92, 0.00], [0.00, 0.00, -1.09]])*1e6 #N3
A14_layer = np.array([[-0.84, -0.07, 0.00], [-0.07, 1.06, 0.00], [0.00,0.00,-1.81]])*1e6 #H3
A15_layer = np.array([[-0.72, 0.08, 0.00], [0.08, -0.64, 0.00], [0.00, 0.00, 1.07]])*1e6 #N1

final_layer = np.array([Zeeman_layerA, Zeeman_layerB, A1_layer, A2_layer, A3_layer, A4_layer, A5_layer, A6_layer, A7_layer, A8_layer, A9_layer, A10_layer, A11_layer, A12_layer, A13_layer, A14_layer, A15_layer])


#Calculates all the required matrix products and then scales them by the appropriate quantity as defined by the array above, adding them to the Hamiltonian

list_of_spin_half_operators = [sx_spinhalf, sy_spinhalf, sz_spinhalf]
list_of_spin_one_operators = [sx_spin1, sy_spin1, sz_spin1]

Sparse_Hamiltonian = csr_matrix((matrix_size, matrix_size))
for k in range(number_of_spins_included_A):
    for i in range(3):
        for j in range(3):
            if k==0 or 1 < k < 2 + number_of_nuclei_included_A:
                x = [identities for identities in identity_list]  # x becomes [i2, i3, i3, i2, ...]
                x[0] = list_of_spin_half_operators[j]  # x becomes [sx_spinhalf, i3, i3, i2,...] or [sy_spinhalf, i3, i3, i2, ...] ...

                if spin_half_or_spin_1_list_ACTIVE[k] == 1:
                    x[k] = list_of_spin_one_operators[i] #x becomes [sx_spinhalf, sx_spinone, i3, i2, ...]

                else:
                    x[k] = list_of_spin_half_operators[i] #x becomes e.g. [sx_spinhalf, i3, i3, sx_spinhalf]...

            else:
                x = [identities for identities in identity_list]
                x[1] = list_of_spin_half_operators[j]

            matrix_rep = matrix_representation(x)
            Sparse_Hamiltonian += matrix_rep*final_layer[k,i,j]

print(f'Sparse Semi-Definite Hamiltonian created in {datetime.now() - startTime}s')

valmax=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None, mode='normal')
valmin=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None, mode='normal')
Vmax = valmax - valmin
Vmax = Vmax[0]/1e6


print(f'Vmax for FAD-Z with {number_of_nuclei_included_A} nuclei = {Vmax} MHz')
print(f'Time taken is {datetime.now() - startTime}')


