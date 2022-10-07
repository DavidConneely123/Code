import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from datetime import datetime
import matplotlib.pyplot as plt

startTime = datetime.now()

data = np.load('Fw _B-half_Calculations.zip')
N1_tensor = np.load('FN1_mT.npy')
N3_tensor = np.load('FN3_mT.npy')

# Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:

i2 = scipy.sparse.identity(2)
sx_spinhalf = np.array([[0 + 0j, 0.5 + 0j], [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])

i3 = scipy.sparse.identity(3)
sx_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = 1 / 2 * np.sqrt(2) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

gyromag = 1.76085e8
i4 = scipy.sparse.identity(4)

# Defining our Nucleus class

class Nucleus:
    # Class attributes

    is_nucleus = True
    nuclei_included_in_simulation = []
    all = []

    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, state=None):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.state = i2 if self.spin == 1 / 2 else i3
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list nuclei_included_in_simulation
        # and also to the 'all' list

        Nucleus.nuclei_included_in_simulation.append(self)
        Nucleus.all.append(self)

    # This simply changes how the instance objects appear when we print Nucleus.all to the terminal

    def __repr__(self):
        return f"Nucleus('{self.name}', 'spin-{self.spin}') "

    def remove_from_simulation(self):
        Nucleus.nuclei_included_in_simulation.remove(self)

    def remove_all_from_simulation(self):
        for nucleus in Nucleus.all:
            Nucleus.nuclei_included_in_simulation.remove(nucleus)

    def add_to_simulation(self):
        Nucleus.nuclei_included_in_simulation.append(self)

    def identity_matrix(self):
        if self.spin == 1 / 2:
            return i2
        if self.spin == 1:
            return i3

    def sx(self):
        if self.spin == 1 / 2:
            return sx_spinhalf
        if self.spin == 1:
            return sx_spin1

    def sy(self):
        if self.spin == 1 / 2:
            return sy_spinhalf
        if self.spin == 1:
            return sy_spin1

    def sz(self):
        if self.spin == 1 / 2:
            return sz_spinhalf
        if self.spin == 1:
            return sz_spin1

    # We find the dimensions of the nuclear eigenbasis before and after the nucleus of interest

    def nuclear_dimensions_before(self):
        length = 1

        for nucleus in Nucleus.nuclei_included_in_simulation[0: Nucleus.nuclei_included_in_simulation.index(self)]:
            if nucleus.spin == 1:
                length *= 3
            if nucleus.spin == 1/2:
                length *= 2

        return length

    def nuclear_dimensions_after(self):
        length = 1

        for nucleus in Nucleus.nuclei_included_in_simulation[Nucleus.nuclei_included_in_simulation.index(self) + 1:]:
            if nucleus.spin == 1:
                length *= 3
            if nucleus.spin == 1/2:
                length *= 2

        return length

    # This allows us to easily define matrix representations required for the calculation of the hyperfine interaction term in the Hamiltonian

    def SAxIix(self):
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAxIiy(self):
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAxIiz(self):
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAyIix(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAyIiy(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAyIiz(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAzIix(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAzIiy(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

    def SAzIiz(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.nuclear_dimensions_after()), format='csr'),format='csr'),format='csr'),format='csr')

# Nuclei included in the simulation (NB HFI Tensor always in s^-1)

N5 = Nucleus('N5', 1, data['FN5_mT']*gyromag/(2*np.pi))
N10 = Nucleus('N10', 1, data['FN10_mT']*gyromag/(2*np.pi))
H8_1 = Nucleus('H8_1', 1/2, data['FH81_mT']*gyromag/(2*np.pi))
H8_2 = Nucleus('H8_2', 1/2, data['FH82_mT']*gyromag/(2*np.pi))
H8_3 = Nucleus('H8_3', 1/2, data['FH83_mT']*gyromag/(2*np.pi))
H1prime = Nucleus('H1prime', 1/2, data["FH1'_mT"]*gyromag/(2*np.pi))
H1dprime = Nucleus('H1dprime', 1/2, data["FH1''_mT"]*gyromag/(2*np.pi))
H6 = Nucleus('H6', 1/2, data['FH6_mT']*gyromag/(2*np.pi))
H7_1 = Nucleus('H7_1', 1/2, data['FH71_mT']*gyromag/(2*np.pi))
H7_2 = Nucleus('H7_2', 1/2, data['FH72_mT']*gyromag/(2*np.pi))
H7_3 = Nucleus('H7_3', 1/2, data['FH73_mT']*gyromag/(2*np.pi))
H9 = Nucleus('H9', 1/2, data['FH9_mT']*gyromag/(2*np.pi))
N3 = Nucleus('N3', 1, N3_tensor*gyromag/(2*np.pi))
H3 = Nucleus('H3', 1/2, data['FH3_mT']*gyromag/(2*np.pi))
N1 = Nucleus('N3', 1, N1_tensor*gyromag/(2*np.pi))


# We calculate an identity matrix of the correct size for the nuclear basis

def Identity_Matrix_of_Nuclear_Spins():
    size = 1
    for nucleus in Nucleus.nuclei_included_in_simulation:
        if nucleus.spin == 1 / 2:
            size *= 2
        if nucleus.spin == 1:
            size *= 3

    return scipy.sparse.identity(size)


# This can then be used to easily construct all of the matrix representations for the electronic operators

def SAx():
    return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SAy():
    return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SAz():
    return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SBx():
    return scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SBy():
    return scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SBz():
    return scipy.sparse.kron(i2, scipy.sparse.kron(sz_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SAxSBx():
    return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(sx_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SAySBy():
    return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(sy_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

def SAzSBz():
    return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(sz_spinhalf, Identity_Matrix_of_Nuclear_Spins(),format='csr'),format='csr')

#This allows us to define the singlet (and triplet) projection operators for the system:

def Ps():
    return 1/4 * scipy.sparse.kron(i4, Identity_Matrix_of_Nuclear_Spins()) - SAxSBx() - SAySBy() - SAzSBz()


# Can then define the Zeeman Hamiltonian:
def H_zee(field_strength, theta, phi):
    return gyromag * field_strength * (np.sin(theta) * np.cos(phi) * (SAx() + SBx()) + np.sin(theta) * np.sin(phi) * (SAy() + SBy()) + np.cos(theta) * (SAz() + 0*SBz()))



# Using the matrix representations defined in the class we can now also define the hyperfine interaction term in the hamiltonian:

def H_hyperfine():
    sum = 0
    for nucleus in Nucleus.nuclei_included_in_simulation:
        sum += (nucleus.hyperfine_interaction_tensor[0,0]*nucleus.SAxIix() + nucleus.hyperfine_interaction_tensor[0,1]*nucleus.SAxIiy() + nucleus.hyperfine_interaction_tensor[0,2]*nucleus.SAxIiz() + nucleus.hyperfine_interaction_tensor[1,0]*nucleus.SAyIix() + nucleus.hyperfine_interaction_tensor[1,1]*nucleus.SAyIiy() + nucleus.hyperfine_interaction_tensor[1,2]*nucleus.SAyIiz() + nucleus.hyperfine_interaction_tensor[2,0]*nucleus.SAzIix() + nucleus.hyperfine_interaction_tensor[2,1]*nucleus.SAzIiy() + nucleus.hyperfine_interaction_tensor[2,2]*nucleus.SAzIiz())

    return sum

# Vmax can now be calculated as before...

def Sparse_Hamiltonian(field_strength, theta, phi):
    return H_hyperfine() + H_zee(field_strength, theta, phi)

def Dense_Hamiltonian(field_strength, theta,phi):
    return Sparse_Hamiltonian(field_strength, theta, phi).todense()


def Vmax(field_strength, theta, phi, display = False):
    sp = Sparse_Hamiltonian(field_strength, theta, phi)

    if display:
        print(f'Sparse Hamiltonian created in {datetime.now() - startTime}s')

    valmax=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian(field_strength,theta, phi), k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian(field_strength,theta, phi), k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6

    print(f'Maximum Eigenvalue = {valmax}, Minimum Eigenvalue = {valmin}')

    if display:
        print(f'Vmax for FAD-Z with {len(Nucleus.nuclei_included_in_simulation)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {datetime.now()-startTime}')
    return Vmax




def Singlet_yield(rate_constant, field_strength, theta, phi):
    Hamiltonian = Dense_Hamiltonian(field_strength,theta,phi)
    eigenvalues, A = np.linalg.eig(Hamiltonian)
    A_hermitian_conjugate = np.linalg.inv(A)
    Singlet_projection = Ps().todense()
    Singlet_projection_transformed = np.dot(A_hermitian_conjugate, np.dot(Singlet_projection, A))

    sum = 0

    for m in range(8):
        for j in range(8):
            sum += 1 / 2 * ((rate_constant ** 2) / (rate_constant ** 2 + (eigenvalues[m] - eigenvalues[j]) ** 2)) * (Singlet_projection_transformed[j, m] ** 2)

    return sum


def Sequentially_adding_in_nuclei_to_simulation():
    x1 = np.linspace(1, len(Nucleus.all), len(Nucleus.all))
    x2 = []

    N5.remove_all_from_simulation()

    for k in range(len(Nucleus.all)):
        x = Nucleus.all[k]
        x.add_to_simulation()
        print(Nucleus.nuclei_included_in_simulation)
        x2.append(Vmax(0.05, 0, 0))


    print(x1)
    print(x2)

    plt.plot(x1,x2)
    plt.xlabel('Number of Nuclei Included in Simulation')
    plt.ylabel('Vmax')
    plt.show()

Sequentially_adding_in_nuclei_to_simulation()







