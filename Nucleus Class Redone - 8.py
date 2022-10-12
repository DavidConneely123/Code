import time

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.constants import physical_constants
import scipy.stats
from scipy.stats import linregress
from datetime import datetime
import random
import matplotlib.pyplot as plt
import warnings
import cProfile
import pstats
from scipy.optimize import curve_fit
import pandas as pd
import operator

startTime = time.perf_counter()

# Defining the basic matrix representations for both spin-half(1H) and spin-1(14N) nuclei:
i2 = scipy.sparse.identity(2)
sx_spinhalf = np.array([[0 + 0j, 0.5 + 0j], [0.5 + 0j, 0 + 0j]])
sy_spinhalf = np.array([[0 + 0j, 0 - 0.5j], [0 + 0.5j, 0 + 0j]])
sz_spinhalf = np.array([[0.5 + 0j, 0 + 0j], [0 + 0j, -0.5 + 0j]])

i3 = scipy.sparse.identity(3)
sx_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_spin1 = 1 / 2 * np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
sz_spin1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

gyromag = scipy.constants.physical_constants['electron gyromagn. ratio'][0] / 1000   # NB ! in rad s^-1 mT^-1
i4 = scipy.sparse.identity(4)

# Defining our Nucleus class

class Nucleus:
    # Class attributes

    is_nucleus = True
    nuclei_included_in_simulation = []
    all = []
    isotopologues = []

    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, is_Isotopologue=False):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.is_Isotopologue = is_Isotopologue
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list nuclei_included_in_simulation
        # and also to the 'all' list


        Nucleus.all.append(self)

        if self.is_Isotopologue:
            Nucleus.isotopologues.append(self)

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

    @classmethod
    def add_all_to_simulation(cls):
        for nucleus in Nucleus.all:
            Nucleus.nuclei_included_in_simulation.append(nucleus)

    @classmethod
    def reset_simulation(cls):
        Nucleus.nuclei_included_in_simulation = []
        for nucleus in Nucleus.all:
            Nucleus.nuclei_included_in_simulation.append(nucleus)


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
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format = 'coo') , scipy.sparse.kron( self.sx() , scipy.sparse.identity(self.nuclear_dimensions_after()))))

    def Iy(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format = 'coo') ,scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.nuclear_dimensions_after()))))

    def Iz(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format = 'coo') ,scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.nuclear_dimensions_after()))))

    # We then make the representations for the SApIiq operators via matrix multiplication

    def SAxIix(self):
        return Nucleus.SAx() * self.Ix()

    def SAxIiy(self):
        return Nucleus.SAx() * self.Iy()

    def SAxIiz(self):
        return Nucleus.SAx() * self.Iz()

    def SAyIix(self):
        return Nucleus.SAy() * self.Ix()

    def SAyIiy(self):
        return Nucleus.SAy() * self.Iy()

    def SAyIiz(self):
        return Nucleus.SAy() * self.Iz()

    def SAzIix(self):
        return Nucleus.SAz() * self.Ix()

    def SAzIiy(self):
        return Nucleus.SAz() * self.Iy()

    def SAzIiz(self):
        return Nucleus.SAz() * self.Iz()

    # Calculating the size of the nuclear Hilbert space
    @classmethod
    def Identity_Matrix_of_Nuclear_Spins(cls):
        Nuclear_Dimensions = 1
        for nucleus in Nucleus.nuclei_included_in_simulation:
            if nucleus.spin == 1 / 2:
                Nuclear_Dimensions *= 2
            if nucleus.spin == 1:
                Nuclear_Dimensions *= 3

        Identity_Matrix_of_Nuclear_Spins = scipy.sparse.identity(Nuclear_Dimensions, format = 'coo')
        return Identity_Matrix_of_Nuclear_Spins

    # Creating matrix representations for the electronic operators
    @classmethod
    def SAx(cls):
        SAx = scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'), format='csr')
        return SAx

    @classmethod
    def SAy(cls):
        SAy = scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'), format='csr')
        return SAy

    @classmethod
    def SAz(cls):
        SAz = scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'), format='csr')
        return SAz

    @classmethod
    def SBx(cls):
        SBx = scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'), format='csr')
        return SBx

    @classmethod
    def SBy(cls):
        SBy = scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'),  format='csr')
        return SBy

    @classmethod
    def SBz(cls):
        SBz = scipy.sparse.kron(i2, scipy.sparse.kron(sz_spinhalf, Nucleus.Identity_Matrix_of_Nuclear_Spins(), format='csr'),  format='csr')
        return SBz

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
H3 = Nucleus('H3', 1/2, np.array([[-0.84, -0.07, 0.00], [-0.07, 1.06, 0.00], [0.00,0.00,-1.81]])*1e6)
N1 = Nucleus('N1', 1, np.array([[-0.72, 0.08, 0.00], [0.08, -0.64, 0.00], [0.00, 0.00, 1.07]])*1e6)


C8 = Nucleus('C8', 1/2, np.array([[1.10324799, -0.34094734, 0.00], [-0.34094734, 1.55501846, 0.00], [0.00, 0.00, 46.49118]])*1e6, is_Isotopologue=True)
C5A = Nucleus('C5A', 1/2, np.array([[-16.39801673, -1.7214494, 0.00], [-1.7214494, -13.97962665, 0.00], [0.00, 0.00, -34.8079]])*1e6 , is_Isotopologue=True)
C6 = Nucleus('C6', 1/2, np.array([[0.570117858, 0.0107777895, 0], [0.0107777895, 0.703398326, 0], [0, 0, 38.0512000]])*1e6, is_Isotopologue=True)
C4A = Nucleus('C4A', 1/2, np.array([[-16.80154633, 1.29779775, 0.00], [1.29779775, -15.64680962, 0.00], [0.00, 0.00, 18.27357]])*1e6, is_Isotopologue=True)
C7 = Nucleus('C7', 1/2, np.array([[-9.1508731, -0.46094791, 0.00], [-0.46094791, -9.363066042, 0.00], [0.00, 0.00, -22.43097]])*1e6, is_Isotopologue=True)
C9A = Nucleus('C9A', 1/2, np.array([[-1.99205207, -0.20734695, 0.00], [-0.20734695, -1.41081737, 0.00], [0.00, 0.00, -18.58458]])*1e6, is_Isotopologue=True)
C9 = Nucleus('C9', 1/2, np.array([[-7.97712849, -0.043130833, 0.00], [-0.04310833, -6.91664501, 0.00], [0.00, 0.00, -13.66601]])*1e6, is_Isotopologue=True)
C10 = Nucleus('C10', 1/2, np.array([[-9.01294238, -0.22105705, 0.00], [-0.22105705, -10.25776244, 0.00], [0.00, 0.00, 3.70115]])*1e6, is_Isotopologue = True)
C4 = Nucleus('C4', 1/2, np.array([[-9.01473609, -0.84701422, 0.00], [-0.84701422, -9.19059253, 0.00], [0.00, 0.00, 1.57954]])*1e6, is_Isotopologue = True)
C8M = Nucleus('C8M', 1/2, np.array([[-7.06096989, -0.3739835, 0.00], [-0.3739835, -6.67867194, 0.00], [0.00, 0.00, -7.03779]])*1e6, is_Isotopologue = True)
C1prime = Nucleus('C1prime', 1/2, np.array([[-3.61055802, -0.0455604, 0.00], [-0.0455604, -4.78196682, 0.00], [0.00, 0.00, -5.09795]])*1e6, is_Isotopologue = True)
C2 = Nucleus('C2', 1/2, np.array([[-2.28627732, 0.52152737, 0.00], [0.52152737, -1.49717727, 0.00], [0.00, 0.00, 1.65115]])*1e6, is_Isotopologue = True)
C7M = Nucleus('C7M', 1/2, np.array([[1.86995394, 0.17078518, 0.00], [0.17078518, 2.054022, 0.00], [0.00, 0.00, 1.48894]])*1e6, is_Isotopologue = True)


# We can define the singlet projection operator

def Ps():
    return 1/4 * scipy.sparse.kron(i4, Nucleus.Identity_Matrix_of_Nuclear_Spins) - Nucleus.SAxSBx() - Nucleus.SAySBy() - Nucleus.SAzSBz()

# Can also define the Zeeman Hamiltonian

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 0*Nucleus.SBz()))

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
        print(f'Sparse Hamiltonian created in {time.perf_counter() - startTime}s')
        print(f'Nuclei Included in Simulation = {Nucleus.nuclei_included_in_simulation}')

    valmax=scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin=scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6

    if display:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}')

    if display:
        print(f'Vmax with {len(Nucleus.nuclei_included_in_simulation)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {time.perf_counter()-startTime}')
    return Vmax

def Sum_of_Squares_HFI():
    sum_of_squares_of_HFI_tensor = 0
    for nucleus in Nucleus.nuclei_included_in_simulation:
        sum_of_squares_of_HFI_tensor += np.sum(nucleus.hyperfine_interaction_tensor ** 2)
    print(sum_of_squares_of_HFI_tensor)

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


# Running a variety of different simulations

def Sequentially_adding_in_nuclei_to_simulation(field_strength,theta,phi):
    Nucleus.reset_simulation()
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
    plt.title("Variation in Vmax with Number of Nuclei Included in Simulation : FAD-Z", fontsize=15)

    # setting x-axis label and y-axis label
    plt.xlabel("Number of Nuclei Inlcuded in Simulation")
    plt.ylabel("Vmax (MHz)")

    # Loop
    for k in range(len(Nucleus.all)):
        x1 = np.linspace(1,k+1, k+1)

        # creating new Y values
        x = Nucleus.all[k]
        x.add_to_simulation()
        y1.append(Vmax(field_strength, theta, phi, display=True))
        print(y1)

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
    plt.show(block = True)


def Iterating_Over_Selections_of_13C(number_of_iterations, number_of_13C, file_name = None):
    x1 =[]
    y1 =[]

    iteration_counter = 0

    while iteration_counter < number_of_iterations:
        Nucleus.reset_simulation()
        random_list = random.sample(range(len(Nucleus.isotopologues)) ,  len(Nucleus.isotopologues) - number_of_13C)  # Choose the N-7 13C to remove from the simulation

        for index, nucleus in enumerate(Nucleus.isotopologues):
            if index in random_list:
                nucleus.remove_from_simulation()  # Remove from simulation

        H7_1.remove_from_simulation()
        H7_2.remove_from_simulation()
        H7_3.remove_from_simulation()
        H9.remove_from_simulation()
        N3.remove_from_simulation()
        H3.remove_from_simulation()
        N1.remove_from_simulation()

        sum_of_squares_of_HFI_tensor = 0
        for nucleus in Nucleus.nuclei_included_in_simulation:
            sum_of_squares_of_HFI_tensor += np.sum(nucleus.hyperfine_interaction_tensor ** 2)

        x1.append(sum_of_squares_of_HFI_tensor)
        y1.append(Vmax(0.05,0,0))

        iteration_counter += 1
        print(iteration_counter)

    df = pd.DataFrame({'x': x1 , 'y': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)


    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x1,y1)
    print(r_value)

    coefficents_of_fitting = np.polyfit(x1,y1,1)
    fitting = np.poly1d(coefficents_of_fitting)
    print(fitting)

    plt.plot(x1,y1, 'yo', x1 ,fitting(x1), '--k')

    plt.xlabel('Sum of Squares of HFI Tensors')
    plt.ylabel('Vmax')
    plt.show()


def Iterating_Over_Selections_All_Nuclei_SS(number_of_iterations, number_of_nuclei, file_name = None):
    x1 =[]
    y1 =[]

    iteration_counter = 0
    Combinations_of_Nuclei_Tested = []

    while iteration_counter < number_of_iterations:
        Nucleus.reset_simulation()
        random_list = random.sample(range(len(Nucleus.all)) ,  len(Nucleus.all) - number_of_nuclei)  # Choose the N-7 13C to remove from the simulation

        for index, nucleus in enumerate(Nucleus.all):
            if index in random_list:
                nucleus.remove_from_simulation()  # Remove from simulation



        if Nucleus.nuclei_included_in_simulation in Combinations_of_Nuclei_Tested:
            print('already tested that combination of nuclei')
            iteration_counter += 0
            continue

        print(Nucleus.nuclei_included_in_simulation)
        sum_of_squares_of_HFI_tensor = 0
        for nucleus in Nucleus.nuclei_included_in_simulation:
            sum_of_squares_of_HFI_tensor += np.sum(nucleus.hyperfine_interaction_tensor ** 2)

        x1.append(sum_of_squares_of_HFI_tensor)
        y1.append(Vmax(0.05,0,0))

        iteration_counter += 1
        print(iteration_counter)

        Combinations_of_Nuclei_Tested.append(Nucleus.nuclei_included_in_simulation)

    df = pd.DataFrame({'x': x1, 'y': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x1,y1)
    print(r_value)

    coefficents_of_fitting = np.polyfit(x1,y1,1)
    fitting = np.poly1d(coefficents_of_fitting)
    print(fitting)

    plt.plot(x1,y1, 'yo', x1 ,fitting(x1), '--k')

    print(len(x1))

    plt.xlabel('Sum of Squares of HFI Tensors')
    plt.ylabel('Vmax')
    plt.show()


def Iterating_Over_Selections_All_Nuclei_SS2(number_of_iterations, number_of_nuclei, file_name = None):
    x1 =[]
    y1 =[]

    iteration_counter = 0
    Combinations_of_Nuclei_Tested = []

    while iteration_counter < number_of_iterations:
        Nucleus.reset_simulation()
        random_list = random.sample(range(len(Nucleus.all)) ,  len(Nucleus.all) - number_of_nuclei)  # Choose the N-7 13C to remove from the simulation

        for index, nucleus in enumerate(Nucleus.all):
            if index in random_list:
                nucleus.remove_from_simulation()  # Remove from simulation



        if Nucleus.nuclei_included_in_simulation in Combinations_of_Nuclei_Tested:
            print('already tested that combination of nuclei')
            iteration_counter += 0
            continue

        print(Nucleus.nuclei_included_in_simulation)
        sum_of_squares_of_HFI_tensor = 0
        for nucleus in Nucleus.nuclei_included_in_simulation:
            sum_of_squares_of_HFI_tensor += np.sum(np.matmul(nucleus.hyperfine_interaction_tensor, nucleus.hyperfine_interaction_tensor))

        x1.append(sum_of_squares_of_HFI_tensor)
        y1.append(Vmax(0.05,0,0))

        iteration_counter += 1
        print(iteration_counter)

        Combinations_of_Nuclei_Tested.append(Nucleus.nuclei_included_in_simulation)

    df = pd.DataFrame({'HFI Scale': x1, 'Vmax': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x1,y1)
    print(r_value)

    coefficents_of_fitting = np.polyfit(x1,y1,1)
    fitting = np.poly1d(coefficents_of_fitting)
    print(fitting)

    plt.plot(x1,y1, 'yo', x1 ,fitting(x1), '--k')

    print(len(x1))

    plt.xlabel('Sum of A.A')
    plt.ylabel('Vmax')
    plt.show()


def Iterating_Over_Selections_All_Nuclei_SS_Quadratic_Fit(number_of_iterations, number_of_nuclei, file_name = None):
    x1 =[]
    y1 =[]

    iteration_counter = 0
    Combinations_of_Nuclei_Tested = []

    while iteration_counter < number_of_iterations:
        Nucleus.reset_simulation()
        random_list = random.sample(range(len(Nucleus.all)) ,  len(Nucleus.all) - number_of_nuclei)  # Choose the N-7 13C to remove from the simulation

        for index, nucleus in enumerate(Nucleus.all):
            if index in random_list:
                nucleus.remove_from_simulation()  # Remove from simulation



        if Nucleus.nuclei_included_in_simulation in Combinations_of_Nuclei_Tested:
            iteration_counter += 0
            continue


        sum_of_squares_of_HFI_tensor = 0
        for nucleus in Nucleus.nuclei_included_in_simulation:
            sum_of_squares_of_HFI_tensor += np.sum(nucleus.hyperfine_interaction_tensor ** 2)

        x1.append(sum_of_squares_of_HFI_tensor)
        y1.append(Vmax(0.05,0,0))

        iteration_counter += 1
        print(iteration_counter)
        print(f'Time Running = {time.perf_counter() -startTime}')

        Combinations_of_Nuclei_Tested.append(Nucleus.nuclei_included_in_simulation)

    df = pd.DataFrame({'x': x1, 'y': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)

    def objective_function(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    popt, _ = curve_fit(objective_function, x1, y1)
    a, b , c, d = popt
    print(f'{a}x^3 + {b}x^2 + {c}x + {d}')


    plt.scatter(x1,y1)
    x_line = np.arange(min(x1), max(x1), 1e13)
    y_line = objective_function(x_line, a, b, c, d)

    plt.plot(x_line, y_line, '--', color = 'red')


    plt.xlabel('Sum of Squares of HFI Tensors')
    plt.ylabel('Vmax')
    plt.show()


def Iterating_Over_Selections_All_Nuclei_Amn_Higher_Powers(number_of_iterations, number_of_nuclei, file_name = None):
    x1 =[]
    y1 =[]

    iteration_counter = 0
    Combinations_of_Nuclei_Tested = []

    while iteration_counter < number_of_iterations:
        Nucleus.reset_simulation()
        random_list = random.sample(range(len(Nucleus.all)) ,  len(Nucleus.all) - number_of_nuclei)  # Choose the N-7 13C to remove from the simulation

        for index, nucleus in enumerate(Nucleus.all):
            if index in random_list:
                nucleus.remove_from_simulation()  # Remove from simulation



        if Nucleus.nuclei_included_in_simulation in Combinations_of_Nuclei_Tested:
            iteration_counter += 0
            continue


        sum_of_squares_of_HFI_tensor = 0
        for nucleus in Nucleus.nuclei_included_in_simulation:
            sum_of_squares_of_HFI_tensor += np.sum(nucleus.hyperfine_interaction_tensor ** 3)

        x1.append(sum_of_squares_of_HFI_tensor)
        y1.append(Vmax(0.05,0,0))

        iteration_counter += 1
        print(iteration_counter)
        print(f'Time Running = {time.perf_counter() -startTime}')

        Combinations_of_Nuclei_Tested.append(Nucleus.nuclei_included_in_simulation)

    df = pd.DataFrame({'x': x1, 'y': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)


    plt.scatter(x1,y1)

    plt.ylabel('Vmax')
    plt.show()


# This creates a dictionary which gives Vmax (NB: with the component due to the Zeeman Interaction of the electron with the field of 1.4MHz subtracted off) for each nucleus by itself

def Vmax_dictionary():
    Vmax_dictionary = {}
    for nucleus in Nucleus.all:
        Nucleus.reset_simulation()
        Nucleus.remove_all_from_simulation()
        Vmax_no_nuclei = Vmax(0.05,0,0)
        nucleus.add_to_simulation()
        Vmax_dictionary[nucleus] = Vmax(0.05,0,0) - Vmax_no_nuclei

    print(Vmax_dictionary)

# This is used to make a dictionary {Nucleus: Sum of Squares of HFI Tensor} which is sorted in descending order

def Sum_of_Squares_Dictionary():
    Nucleus.reset_simulation()
    Sum_of_Squares_Dictionary = {}

    for nucleus in Nucleus.all:
        Nucleus.reset_simulation()
        Nucleus.remove_all_from_simulation()
        nucleus.add_to_simulation()
        Sum_of_Squares_Dictionary[nucleus] = np.sum(nucleus.hyperfine_interaction_tensor ** 2)
        Descending_Order = sorted(Sum_of_Squares_Dictionary.items(), key=operator.itemgetter(1),reverse=True)

    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()


    return Descending_Order

def Vmax_largest_nuclei():
    # Creating a dictionary that orders the nuclei in descending order of Sum of Squares of their HFI Tensors

    Descending_Order = Sum_of_Squares_Dictionary()

    # Completely resetting the simulation

    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()


    for i in range(1):
      Descending_Order[i][0].add_to_simulation()
      print(Nucleus.nuclei_included_in_simulation)


    Vmax(0.05,0,0, display = True)


# This is a method which adds nuclei sequentially to the simulation in order of decreasing sum of squares of HFI Tensor and then plots Vmax as the number of nuclei included increases

def Vmax_largest_nuclei_sequentially(number_of_nuclei, field_strength, theta, phi, file_name = None):
    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()

    # creating initial data values of x and y
    x1 = []
    y1 = []

    # to run GUI event loop
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x1, y1)

    # setting title
    plt.title("Variation in Vmax with Number of Nuclei Included in Simulation : FAD-Z", fontsize=15)

    # setting x-axis label and y-axis label
    plt.xlabel("Number of Nuclei Inlcuded in Simulation")
    plt.ylabel("Vmax (MHz)")


    #
    for iterations in range(0, number_of_nuclei+1):
        x1 = [i for i in range(0, iterations+1)]
        Descending_Order = Sum_of_Squares_Dictionary()

        Nucleus.reset_simulation()
        Nucleus.remove_all_from_simulation()

        for i in range(iterations):
            x = Descending_Order[i][0]
            x.add_to_simulation()

        y1.append(Vmax(field_strength, theta, phi))

        # updating data values
        line1.set_xdata(x1)
        line1.set_ydata(y1)

        plt.plot(x1, y1)

        # drawing updated values
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        time.sleep(0.1)

    print(x1)
    print(y1)

    df = pd.DataFrame({'Number of Nuclei': x1, 'Vmax': y1})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name, index=False)

    plt.show(block=True)

Vmax_largest_nuclei_sequentially(20, 0.05, 0, 0, 'Building up Vmax with Nuclei - Decreasing SS - 15 Nuclei')



