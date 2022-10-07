import numpy as np

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


# Defining our Nucleus class

class Nucleus:
    # Class attributes

    is_nucleus = True
    nuclei_included_in_simulation = []

    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, state=None, included_in_simulation=True):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.included_in_simulation = included_in_simulation
        self.state = i2 if self.spin == 1 / 2 else i3
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor

        # Actions to execute

        Nucleus.nuclei_included_in_simulation.append(self)

    # This simply changes how the instance objects appear when we print Nucleus.all to the terminal

    def __repr__(self):
        return f"Nucleus('{self.name}', 'spin-{self.spin}', '{self.included_in_simulation}') "

    def remove_from_simulation(self):
        self.included_in_simulation = False
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




# Nuclei included in the simulation

N5 = Nucleus('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
N10 = Nucleus('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
H8_1 = Nucleus('H8_1', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
H8_2 = Nucleus('H8_2', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
H8_3 = Nucleus('H8_3', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)


N5.remove_from_simulation()
H8_3.remove_from_simulation()



print(Nucleus.nuclei_included_in_simulation)















# We calculate an identity matrix of the correct size for the nuclear basis

def Identity_Matrix_of_Nuclear_Spins():
    size = 1
    for nucleus in Nucleus.nuclei_included_in_simulation:
        if nucleus.spin == 1 / 2:
            size *= 2
        if nucleus.spin == 1:
            size *= 3

    return np.identity(size)


# This can then be used to easily construct all of the matrix representations for the electronic operators

def SAx():
    return np.kron(sx_spinhalf, np.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SAy():
    return np.kron(sy_spinhalf, np.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SAz():
    return np.kron(sz_spinhalf, np.kron(i2, Identity_Matrix_of_Nuclear_Spins()))


def SBx():
    return np.kron(i2, np.kron(sx_spinhalf, Identity_Matrix_of_Nuclear_Spins()))


def SBy():
    return np.kron(i2, np.kron(sy_spinhalf, Identity_Matrix_of_Nuclear_Spins()))


def SBz():
    return np.kron(sz_spinhalf, np.kron(sz_spinhalf, Identity_Matrix_of_Nuclear_Spins()))


# Can then define the Zeeman Hamiltonian:
def H_zee(field_strength, theta, phi):
    return gyromag * field_strength * (
                np.sin(theta) * np.cos(phi) * (SAx() + SBx()) + np.sin(theta) * np.sin(phi) * (SAx() + SBx()) + np.cos(
            theta) * (SAz() + SBz()))

# To calculate the various SAxI_kx... for the hyperfine interactions we need to:
# 1. define some sort of 'order' to the nuclei
# 2. calculate the size of the identity matrix that will come before
# 3. calculate the size of the identity matrix that will come after
