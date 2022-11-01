import itertools
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.constants import physical_constants
import scipy.stats
import pandas as pd
import operator
import plotly.graph_objects as go

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

    nuclei_included_in_simulation = []
    all = []
    isotopologues = []


    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, is_isotopologue=False):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.is_isotopologue = is_isotopologue
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list 'all' and if it is a 13C it is
        # also added to the 'Isotopologues' list

        # NB! Initially no nuclei are included in the simulation

        Nucleus.all.append(self)

        if self.is_isotopologue:
            Nucleus.isotopologues.append(self)


    # This simply changes how the instance objects appear when we print Nucleus.all to the terminal


    def __repr__(self):
        return f"Nucleus('{self.name}', 'spin-{self.spin}') "

    ######################################################## Simulation Methods ##############################################

    def remove_from_simulation(self):
        try:
            Nucleus.nuclei_included_in_simulation.remove(self)

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    @classmethod
    def remove_all_from_simulation(cls):
        try:
            for nucleus in Nucleus.all:
                Nucleus.nuclei_included_in_simulation.remove(nucleus)

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

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
            Nucleus.nuclei_included_in_simulation.append(nucleus) # This adds all the nuclei to the simulation

    def deuterate(self):
        self.hyperfine_interaction_tensor = 0.154*self.hyperfine_interaction_tensor
        self.spin = 1
        self.name = self.name + '_DEUTERATED'

    def nitrogen_label(self):
        self.hyperfine_interaction_tensor = -1.402*self.hyperfine_interaction_tensor
        self.spin = 1/2
        self.name = self.name + '_LABELLED_15N'


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
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format='coo'), scipy.sparse.kron(self.sx(), scipy.sparse.identity(self.nuclear_dimensions_after()))))

    def Iy(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format='coo'), scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.nuclear_dimensions_after()))))

    def Iz(self):
        return scipy.sparse.kron(i4, scipy.sparse.kron(scipy.sparse.identity(self.nuclear_dimensions_before(), format='coo'), scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.nuclear_dimensions_after()))))


    # Calculating the size of the nuclear Hilbert space
    @classmethod
    def Nuclear_Dimensions(cls):
        Nuclear_Dimensions = 1
        for nucleus in Nucleus.nuclei_included_in_simulation:
            if nucleus.spin == 1 / 2:
                Nuclear_Dimensions *= 2
            if nucleus.spin == 1:
                Nuclear_Dimensions *= 3
        return Nuclear_Dimensions

    @classmethod
    def Identity_Matrix_of_Nuclear_Spins(cls):
        Nuclear_Dimensions = 1
        for nucleus in Nucleus.nuclei_included_in_simulation:
            if nucleus.spin == 1 / 2:
                Nuclear_Dimensions *= 2
            if nucleus.spin == 1:
                Nuclear_Dimensions *= 3

        Identity_Matrix_of_Nuclear_Spins = scipy.sparse.identity(Nuclear_Dimensions, format='coo')
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
def tensor_list(number_of_tensors):
    list_of_tensors = []

    while list_of_tensors == []:
        for _ in range(number_of_tensors):                         # An Axial HFIT #
            tensor = np.zeros((3,3))                             #-----------------------#
            tensor[0,0] = np.random.uniform(-1,1)                # small #   0   #   0   #
            tensor[1,1] = np.random.uniform(-1,1)                #   0   # small #   0   #
            tensor[2,2] = np.random.uniform(-50/3, 50/3)         #   0   #   0   # large #
                                                                 #-----------------------#
            list_of_tensors.append(tensor)

        """
        sum = 0
        for tensor in list_of_tensors:
            sum += (1 / 3) * (np.sum(tensor ** 2)) * (1/2) * (1/2 + 1)   # Sigma (NB with only spin-1/2 nuclei)

        sigma = np.sqrt(1/3 * sum)
        

        if not 28 < 200*sigma < 29:
            list_of_tensors = []  # If the sigma value is out of the desired range the procedure is repeated...

        """

    return list_of_tensors



# Can also define the Zeeman Hamiltonian (NB in s^-1)

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 0*Nucleus.SBz()))

# Can also define a term for the interaction of our electron spins with an rf field perpendicular to the geomagnetic field
# NB: note that the field strength of this rf field will not in general be the same as the geomagnetic field strenght !!!!

# Can also define a dipolar coupling Hamiltonian

def H_dip():
    D = np.array([[1.09, -12.12, 4.91], [-12.12, -7.80, 7.02], [4.91, 7.02, 6.71]])*1e6
    sax, say, saz = Nucleus.SAx(), Nucleus.SAy(), Nucleus.SAz()
    sbx, sby, sbz = Nucleus.SBx(), Nucleus.SBy(), Nucleus.SBz()

    return D[0,0]*sax*sbx +  D[0,1]*sax*sby +  D[0,2]*sax*sbz +  D[1,0]*say*sbx +  D[1,1]*say*sby +  D[1,2]*say*sbz +  D[2,0]*saz*sbx +  D[2,1]*saz*sby +  D[2,2]*saz*sbz

# Using the matrix representations defined in the class we can now also define the hyperfine interaction term in the hamiltonian:

def H_hyperfine_new():
    sax, say, saz = Nucleus.SAx(), Nucleus.SAy(), Nucleus.SAz()

    sum = 0
    for nucleus in Nucleus.nuclei_included_in_simulation:
        ix, iy, iz = nucleus.Ix(), nucleus.Iy(), nucleus.Iz()
        val = nucleus.hyperfine_interaction_tensor[0,0]*sax*ix + nucleus.hyperfine_interaction_tensor[0,1]*sax*iy + nucleus.hyperfine_interaction_tensor[0,2]*sax*iy + nucleus.hyperfine_interaction_tensor[1,0]*say*ix + nucleus.hyperfine_interaction_tensor[1,1]*say*iy + nucleus.hyperfine_interaction_tensor[1,2]*say*iz + nucleus.hyperfine_interaction_tensor[2,0]*saz*ix + nucleus.hyperfine_interaction_tensor[2,1]*saz*iy + nucleus.hyperfine_interaction_tensor[2,2]*saz*iz
        sum += val
    return sum

# Can then create (sparse or dense) representations of the total hamiltionian (containing only electron Zeeman and hyperfine terms by default...)
def Sparse_Hamiltonian(field_strength, theta, phi, dipolar = False):
    H_tot = H_hyperfine_new() + H_zee(field_strength, theta, phi)

    if dipolar:
        H_tot += H_dip()

    return H_tot

def Dense_Hamiltonian(field_strength, theta,phi, dipolar = False):
    return Sparse_Hamiltonian(field_strength, theta, phi, dipolar).todense()

# Vmax can now be calculated using the method below

def Vmax(field_strength, theta, phi, display = False, display_eigenvalues = False, dipolar = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian(field_strength, theta, phi, dipolar)

    if display:
        print(f'Sparse Hamiltonian created in {time.perf_counter() - startTime}s')
        print(f'Nuclei Included in Simulation = {Nucleus.nuclei_included_in_simulation}')

    valmax = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='LA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    valmin = scipy.sparse.linalg.eigsh(Hspar, k=1, M=None, sigma=None, which='SA', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=False, Minv=None, OPinv=None)
    Vmax = valmax - valmin
    Vmax = Vmax[0]/1e6    # Converting Vmax from Hz to MHz

    if display_eigenvalues:
      print(f'Maximum Eigenvalue = {valmax * 2 *np.pi}, Minimum Eigenvalue = {valmin * 2 * np.pi}') # Showing the eigenvalues in rad s^-1

    if display:
        print(f'Vmax with {len(Nucleus.nuclei_included_in_simulation)} nuclei = {Vmax} MHz')
        print(f'Time Taken = {time.perf_counter()-startTime}')
    return Vmax

# This method creates a dictionary (Nucleus ; Sum of Squares of Elements of HFIT) for reference

def Sum_of_Squares_Dictionary():
    Nucleus.reset_simulation()    # Add all nuclei to the simulation

    Sum_of_Squares_Dictionary = {}

    for nucleus in Nucleus.all:
        Nucleus.reset_simulation()
        Sum_of_Squares_Dictionary[nucleus] = np.sum(nucleus.hyperfine_interaction_tensor ** 2)
        Sorted = sorted(Sum_of_Squares_Dictionary.items(), key=operator.itemgetter(1),reverse=True)


    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()
    return dict(Sorted)

# This is a method which calculates Vmax for the first k nuclei in the system (based on decreasing SS)
def Vmax_Decreasing_SS(number_of_nuclei, field_strength, theta, phi, file_name = None):
    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()
    Sorted_Dictionary = list(Sum_of_Squares_Dictionary().keys())
    print(Sorted_Dictionary)

    for i in range(number_of_nuclei):
        x = Sorted_Dictionary[i]

        x.add_to_simulation()

    Vmax_value = Vmax(field_strength, theta, phi, display=True)

    df = pd.DataFrame({'Number of Nuclei': [number_of_nuclei], 'Vmax': [Vmax_value]})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name)

    return Vmax_value


def Overestimation_of_Vmax_by_Summing(number_of_nuclei, subsystem_dividing, display = False):
    list_of_tensors = tensor_list(number_of_nuclei)

    for i in range(number_of_nuclei):
        name = f'Nuc{i}'
        Nucleus(name = name, spin = 1/2, is_isotopologue = False, hyperfine_interaction_tensor = list_of_tensors[i]*1e6) # Generate a set of nuclei with the HFIT's created above !

    for nucleus in Nucleus.all[:math.ceil(number_of_nuclei/subsystem_dividing)]:
        nucleus.add_to_simulation()

    Vmax_firsthalf = Vmax(0.05,0,0, display)    # Add the first subsystem of nuclei to the system and calculate Vmax

    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()

    for nucleus in Nucleus.all[math.ceil(number_of_nuclei/subsystem_dividing):]:
        nucleus.add_to_simulation()

    Vmax_secondhalf = Vmax(0,0,0, display)      # Add the second subsystem of nuclei to the system and calculate Vmax


    Nucleus.reset_simulation()

    Vmax_total = Vmax(0.05,0,0,display)         # Add all the nuclei to the system and calculate Vmax

    if display:
         print(f'Overestimation is { (((Vmax_firsthalf + Vmax_secondhalf) - Vmax_total) / Vmax_total)*100}%')

    Nucleus.nuclei_included_in_simulation = []
    my_nuclei={}                  # Completely reset the simulation
    return(((Vmax_firsthalf + Vmax_secondhalf) - Vmax_total) / Vmax_total)*100          # Return the overestimation


def Mean_overestimation(number_of_nuclei, subsystem_dividing, number_of_iterations):
    i = 0
    Overestimation_Sum = 0
    while i < number_of_iterations:
        Overestimation_Sum += Overestimation_of_Vmax_by_Summing(number_of_nuclei, subsystem_dividing)
        i += 1
        Nucleus.all = []


    Overestimation_mean = Overestimation_Sum / number_of_iterations
    Overestimation_mean = round(Overestimation_mean, 2)
    print(f'Mean overestimation with {number_of_nuclei} nuclei split into sub-systems of {math.ceil(number_of_nuclei/subsystem_dividing)} nuclei and {math.ceil(number_of_nuclei -(number_of_nuclei/subsystem_dividing))} nuclei = {Overestimation_mean}%')
    return Overestimation_mean



