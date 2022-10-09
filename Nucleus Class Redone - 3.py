import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.constants import physical_constants
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import cProfile
import pstats
warnings.filterwarnings('ignore')

startTime = datetime.now()

# Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:

i2 = scipy.sparse.identity(2)
sx_spinhalf = np.array([[0 + 0j, 0.5 + 0j], [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])

i3 = scipy.sparse.identity(3)
sx_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

gyromag = 1.760859e8   # NB ! in rad s^-1 mT^-1
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

    @classmethod
    def remove_all_from_simulation(cls):
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


    # We can make matrix representations for the nuclear operators

    def Ix(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()) , scipy.sparse.kron( self.sx() , scipy.identity(self.nuclear_dimensions_after()))))

    def Iy(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()) ,scipy.sparse.kron(self.sy(), scipy.identity(self.nuclear_dimensions_after()))))

    def Iz(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before()) ,scipy.sparse.kron(self.sz(), scipy.identity(self.nuclear_dimensions_after()))))

    # We then make the representations for the SApIiq operators via matrix multiplication

    def SAxIix(self):
        return SAx * self.Ix()

    def SAxIiy(self):
        return SAx * self.Iy()

    def SAxIiz(self):
        return SAx * self.Iz()

    def SAyIix(self):
        return SAy * self.Ix()

    def SAyIiy(self):
        return SAy * self.Iy()

    def SAyIiz(self):
        return SAy * self.Iz()

    def SAzIix(self):
        return SAz * self.Ix()

    def SAzIiy(self):
        return SAz * self.Iy()

    def SAzIiz(self):
        return SAz * self.Iz()

# Nuclei included in the simulation (NB HFI Tensor always in s^-1)

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




# Calculating the size of the nuclear Hilbert space

Nuclear_Dimensions = 1
for nucleus in Nucleus.nuclei_included_in_simulation:
    if nucleus.spin == 1 / 2:
        Nuclear_Dimensions *= 2
    if nucleus.spin == 1:
        Nuclear_Dimensions *= 3

Identity_Matrix_of_Nuclear_Spins = scipy.sparse.identity(Nuclear_Dimensions)

# Creating matrix representations for the electronic operators

SAx = scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')
SAy = scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')
SAz = scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')

SBx = scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')
SBy = scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')
SBz = scipy.sparse.kron(i2, scipy.sparse.kron(sz_spinhalf, Identity_Matrix_of_Nuclear_Spins, format='csr'), format='csr')


# We can define the singlet projection operator

def Ps():
    return 1/4 * scipy.sparse.kron(i4, Identity_Matrix_of_Nuclear_Spins) - Nucleus.SAxSBx() - Nucleus.SAySBy() - Nucleus.SAzSBz()


# Can also define the Zeeman Hamiltonian

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (SAx + SBx) + np.sin(theta) * np.sin(phi) * (SAy + SBy) + np.cos(theta) * (SAz + 0*SBz))



# Using the matrix representations defined in the class we can now also define the hyperfine interaction term in the hamiltonian:

def H_hyperfine():
    sum = 0
    for nucleus in Nucleus.nuclei_included_in_simulation:

        val = nucleus.hyperfine_interaction_tensor[0,0]*nucleus.SAxIix() + nucleus.hyperfine_interaction_tensor[0,1]*nucleus.SAxIiy() + nucleus.hyperfine_interaction_tensor[0,2]*nucleus.SAxIiz() + nucleus.hyperfine_interaction_tensor[1,0]*nucleus.SAyIix() + nucleus.hyperfine_interaction_tensor[1,1]*nucleus.SAyIiy() + nucleus.hyperfine_interaction_tensor[1,2]*nucleus.SAyIiz() + nucleus.hyperfine_interaction_tensor[2,0]*nucleus.SAzIix() + nucleus.hyperfine_interaction_tensor[2,1]*nucleus.SAzIiy() + nucleus.hyperfine_interaction_tensor[2,2]*nucleus.SAzIiz()
        sum += val

    return sum

# Vmax can now be calculated as before...

def Sparse_Hamiltonian(field_strength, theta, phi):
    return H_hyperfine() + H_zee(field_strength, theta, phi)

def Dense_Hamiltonian(field_strength, theta,phi):
    return Sparse_Hamiltonian(field_strength, theta, phi).todense()


def Vmax(field_strength, theta, phi, display = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian(field_strength, theta, phi)

    if display:
        print(f'Sparse Hamiltonian created in {datetime.now() - startTime}s')

    valmax=scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin=scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6

    if display:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}')

    if display:
        print(f'Vmax with {len(Nucleus.nuclei_included_in_simulation)} nuclei = {Vmax} MHz')
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


def Sequentially_adding_in_nuclei_to_simulation(field_strength,theta,phi):

    print(Nucleus.all)
    Nucleus.remove_all_from_simulation()

    import numpy as np
    import time
    import matplotlib.pyplot as plt

    # creating initial data values
    # of x and y
    x1 = []
    y1 = []

    # to run GUI event loop
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x1, y1)

    # setting title
    plt.title("Vmax", fontsize=20)

    # setting x-axis label and y-axis label
    plt.xlabel("Number of Nuclei")
    plt.ylabel("Vmax")

    # Loop
    for k in range(len(Nucleus.all)):
        x1 = np.linspace(1,k+1, k+1)

        # creating new Y values
        x = Nucleus.all[k]
        x.add_to_simulation()
        y1.append(Vmax(field_strength, theta, phi))

        # updating data values
        line1.set_xdata(x1)
        line1.set_ydata(y1)

        plt.plot(x1,y1)

        # drawing updated values
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        time.sleep(0.1)


    print(y1)
    plt.show()


with cProfile.Profile() as pr:
    Vmax(0.05,1.055732, 4.167529,display=True)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()




