import numpy as np
import scipy
import scipy.sparse
from datetime import datetime

startTime = datetime.now()

# Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:

i2 = np.identity(2)
sx_spinhalf = np.array([[0 + 0j, 0.5 + 0j], [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])

i3 = np.identity(3)
sx_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = 1 / 2 * np.sqrt(2) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

gyromag = 1.76e8
i4 = np.identity(4)

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
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), np.identity(self.nuclear_dimensions_after())))))

    def SAxIiy(self):
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), np.identity(self.nuclear_dimensions_after())))))

    def SAxIiz(self):
        return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), np.identity(self.nuclear_dimensions_after())))))

    def SAyIix(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), np.identity(self.nuclear_dimensions_after())))))

    def SAyIiy(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), np.identity(self.nuclear_dimensions_after())))))

    def SAyIiz(self):
        return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), np.identity(self.nuclear_dimensions_after())))))

    def SAzIix(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sx(), np.identity(self.nuclear_dimensions_after())))))

    def SAzIiy(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sy(), np.identity(self.nuclear_dimensions_after())))))

    def SAzIiz(self):
        return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.kron(np.identity(self.nuclear_dimensions_before()), scipy.sparse.kron(self.sz(), np.identity(self.nuclear_dimensions_after())))))

# Nuclei included in the simulation

N5 = Nucleus('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
N10 = Nucleus('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
H8_1 = Nucleus('H8_1', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
H8_2 = Nucleus('H8_2', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
H8_3 = Nucleus('H8_3', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
H1_prime = Nucleus('H1_prime', 1/2, np.array([[11.41, 0.00, 0.00], [0.00, 11.41, 0.00], [0.00, 0.00, 11.41]])*1e6)
H1_doubleprime = Nucleus('H1_doubleprime', 1/2, np.array([[11.41, 0.00, 0.00], [0.00, 11.41, 0.00], [0.00, 0.00, 11.41]])*1e6)
H6 = Nucleus('H6', 1/2, np.array([[-5.63, 0.92, 0.00], [0.92, -14.77, 0.00], [0.00, 0.00, -12.15]])*1e6)
H7_1 = Nucleus('H7_1', 1/2, np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6)
H7_2 = Nucleus('H7_2', 1/2, np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6)
H7_3 = Nucleus('H7_3', 1/2, np.array([[-3.97, 0.00, 0.00], [0.00, -3.97, 0.00], [0.00, 0.00, -3.97]])*1e6)
H9 = Nucleus('H9', 1/2, np.array([[1.88, -0.71, 0.00], [-0.71, 3.01, 0.00], [0.00, 0.00, -0.14]])*1e6)
N3 = Nucleus('N3', 1, np.array([[-1.21, 0.00, 0.00], [0.00, -0.92, 0.00], [0.00, 0.00, -1.09]])*1e6)
H3 = Nucleus('H3', 1/2, np.array([[-0.84, -0.07, 0.00], [-0.07, 1.06, 0.00], [0.00,0.00,-1.81]])*1e6)
N1 = Nucleus('N1', 1, np.array([[-0.72, 0.08, 0.00], [0.08, -0.64, 0.00], [0.00, 0.00, 1.07]])*1e6)

print(H1_prime.nuclear_dimensions_after())
print(Nucleus.nuclei_included_in_simulation)


N1.remove_from_simulation()
H3.remove_from_simulation()
N3.remove_from_simulation()



print(H1_prime.nuclear_dimensions_after())

# We calculate an identity matrix of the correct size for the nuclear basis

def Identity_Matrix_of_Nuclear_Spins():
    size = 1
    for nucleus in Nucleus.nuclei_included_in_simulation:
        if nucleus.spin == 1 / 2:
            size *= 2
        if nucleus.spin == 1:
            size *= 3

    return np.identity(size)

Identity_Matrix_of_Nuclear_Spins()

# This can then be used to easily construct all of the matrix representations for the electronic operators

def SAx():
    return scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SAy():
    return scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SAz():
    return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SBx():
    return scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, Identity_Matrix_of_Nuclear_Spins()))


def SBy():
    return scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, Identity_Matrix_of_Nuclear_Spins()))


def SBz():
    return scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(sz_spinhalf, Identity_Matrix_of_Nuclear_Spins()))

# Can then define the Zeeman Hamiltonian:
def H_zee(field_strength, theta, phi):
    return gyromag * field_strength * (np.sin(theta) * np.cos(phi) * (SAx() + SBx()) + np.sin(theta) * np.sin(phi) * (SAx() + SBx()) + np.cos(theta) * (SAz() + SBz()))



# Using the matrix representations defined in the class we can now also define the hyperfine interaction term in the hamiltonian:

def H_hyperfine():
    for nucleus in Nucleus.nuclei_included_in_simulation:
        return  nucleus.hyperfine_interaction_tensor[0,0]*nucleus.SAxIix() + nucleus.hyperfine_interaction_tensor[0,1]*nucleus.SAxIiy() + nucleus.hyperfine_interaction_tensor[0,2]*nucleus.SAxIiz() + nucleus.hyperfine_interaction_tensor[1,0]*nucleus.SAyIix() + nucleus.hyperfine_interaction_tensor[1,1]*nucleus.SAyIiy() + nucleus.hyperfine_interaction_tensor[1,2]*nucleus.SAyIiz() + nucleus.hyperfine_interaction_tensor[2,0]*nucleus.SAzIix() + nucleus.hyperfine_interaction_tensor[2,1]*nucleus.SAzIiy() + nucleus.hyperfine_interaction_tensor[2,2]*nucleus.SAzIiz()

# Vmax can now be calculated as before...

Sparse_Hamiltonian = H_hyperfine() + H_zee(0.05,0,0)

valmax=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
valmin=scipy.sparse.linalg.eigsh(Sparse_Hamiltonian, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
Vmax = valmax - valmin
Vmax = Vmax[0]/1e6

print(Sparse_Hamiltonian.todense)

print(f'Vmax for FAD-Z with 6 nuclei = {Vmax} MHz')