import numpy as np

#Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:

i2 = np.identity(2)
sx_spinhalf = np.array([[0 + 0j,0.5 + 0j] , [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j] , [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j,0 + 0j] , [0 + 0j, -0.5 + 0j]])

i3 = np.identity(3)
sx_spin1 = 1/2*np.sqrt(2)*np.array([[0,1,0] , [1,0,1], [0,1,0]])
sy_spin1 = 1/2*np.sqrt(2)*np.array([[0, -1j,0] , [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = 1/2*np.sqrt(2)*np.array([[1,0,0], [0,0,0] , [0,0,-1]])

#Defining a function that allows us to take an arbitrary number of Kronecker products


def matrix_representation(spins):
    for index, spin in enumerate(reversed(spins)):

        if index <= 1:
            current_loop = np.kron(spins[-2], spins[-1])

        else:
            current_loop = np.kron(spin, current_loop)

    return current_loop


#Setting up the matrix representations in radical A (FAD) (electronA, N5, N10, H8_1, H8_2, H8_3, H1', H1'', H6, H7_1, H7_2, H7_3, H9, N3, H3, N1)

number_of_spins_included_A = 4

spin_half_or_spin_1_list_ALLSPINS = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
spin_half_or_spin_1_list_ACTIVE = spin_half_or_spin_1_list_ALLSPINS[0:number_of_spins_included_A]

identity_list=[]
for index, value in enumerate(spin_half_or_spin_1_list_ACTIVE):
    if value == 1:
        identity_list.append(i3)
    else:
        identity_list.append(i2)

print(identity_list)


S_AxI_Aix_dict = {}
for i in range(number_of_spins_included_A):
    


    if i == 1:
        x.clear

    else:
        x[i] = sx_spinhalf

    key = i
    value = matrix_representation(x)
    S_AxI_Aix_dict[key] = value

# The matrix representations for S_AxI_Aix are stored in this dictionary (i.e. the matrix representation of
# S_AxI_A1x (i.e. the term which involves the nucleus N5 is stored as S_AxI_Aix_dict[1])

S_AyI_Aiy_dict = {}

for i in range(number_of_spins_included_A):
    x = [sy_spinhalf, i3, i3, i2, i2, i2]

    if i == 1 or i == 2:
        x[i] = sy_spin1

    else:
        x[i] = sy_spinhalf

    key = i
    value = matrix_representation(x)
    S_AyI_Aiy_dict[key] = value

S_AzI_Aiz_dict = {}

for i in range(number_of_spins_included_A):
    x = [sz_spinhalf, i3, i3, i2, i2, i2]

    if i == 1 or i == 2:
        x[i] = sz_spin1

    else:
        x[i] = sz_spinhalf

    key = i
    value = matrix_representation(x)
    S_AzI_Aiz_dict[key] = value








