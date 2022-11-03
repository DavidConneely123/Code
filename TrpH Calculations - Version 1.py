import itertools
import random
import time

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
TrpH_HFITs = np.load('TrpH_HFITS_MHz.npy')   # NB: the HFIT for nucleus 1 is TrpH_HFIT[0] and so on... hence for N9 want TrpH_HFITs[8] and so on

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


N9 = Nucleus('N9', 1, TrpH_HFITs[8]*1e6)
H23 = Nucleus('H23', 1/2, TrpH_HFITs[22]*1e6)
H22 = Nucleus('H22', 1/2, TrpH_HFITs[21]*1e6)
H25 = Nucleus('H25', 1/2, TrpH_HFITs[24]*1e6)
H26 = Nucleus('H26', 1/2, TrpH_HFITs[25]*1e6)
H27 = Nucleus('H27', 1/2, TrpH_HFITs[26]*1e6)
H24 = Nucleus('H24', 1/2, TrpH_HFITs[23]*1e6)
H18 = Nucleus('H18', 1/2, TrpH_HFITs[17]*1e6)
H19 = Nucleus('H19', 1/2, TrpH_HFITs[18]*1e6)
H17 = Nucleus('H17', 1/2, TrpH_HFITs[16]*1e6)
N6 = Nucleus('N6', 1, TrpH_HFITs[5]*1e6)
H21 = Nucleus('H21', 1/2, TrpH_HFITs[20]*1e6)
H20 = Nucleus('H20', 1/2, TrpH_HFITs[19]*1e6)
H16 = Nucleus('H16', 1/2, TrpH_HFITs[15]*1e6)   # NB: This is the carboxylic acid proton, do we include ? (note: has <0.5 MHZ HFI)

C2 = Nucleus('C2', 1/2, TrpH_HFITs[1]*1e6, is_isotopologue=True)
C4 = Nucleus('C4', 1/2, TrpH_HFITs[3]*1e6, is_isotopologue=True)
C5 = Nucleus('C5', 1/2, TrpH_HFITs[4]*1e6, is_isotopologue=True)
C7 = Nucleus('C7', 1/2, TrpH_HFITs[6]*1e6, is_isotopologue=True)
C8 = Nucleus('C8', 1/2, TrpH_HFITs[7]*1e6, is_isotopologue=True)
C10 = Nucleus('C10', 1/2, TrpH_HFITs[9]*1e6, is_isotopologue=True)
C11 = Nucleus('C11', 1/2, TrpH_HFITs[10]*1e6, is_isotopologue=True)
C12 = Nucleus('C12', 1/2, TrpH_HFITs[11]*1e6, is_isotopologue=True)
C13 = Nucleus('C13', 1/2, TrpH_HFITs[12]*1e6, is_isotopologue=True)
C14 = Nucleus('C14', 1/2, TrpH_HFITs[13]*1e6, is_isotopologue=True)
C15 = Nucleus('C15', 1/2, TrpH_HFITs[14]*1e6, is_isotopologue=True)
C2 = Nucleus('C2', 1/2, TrpH_HFITs[1]*1e6, is_isotopologue=True)

#O1 = Nucleus('O1', 5/2, TrpH_HFITs[0]*1e6, is_isotopologue=True)
#O3 = Nucleus('O3', 5/2, TrpH_HFITs[0]*1e6, is_isotopologue=True)   # NB, should we include these, if we do have to input the spin-5/2 matrices etc...


# We now set up the desired nuclei involved in our simulation


# We can define the singlet projection operator

def Ps():
    return 1/4 * scipy.sparse.kron(i4, Nucleus.Identity_Matrix_of_Nuclear_Spins()) - Nucleus.SAx()*Nucleus.SBx() - Nucleus.SAy()*Nucleus.SBy() - Nucleus.SAz()*Nucleus.SBz()

# Can also define the Zeeman Hamiltonian (NB in s^-1)

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 0*Nucleus.SBz()))

# Can also define a term for the interaction of our electron spins with an rf field perpendicular to the geomagnetic field
# NB: note that the field strength of this rf field will not in general be the same as the geomagnetic field strenght !!!!

def H_perp(field_strength, theta, phi):
    return (gyromag / (2 * np.pi)) * field_strength * (np.cos(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.cos(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) - np.sin(theta) * (Nucleus.SAz() + 0 * Nucleus.SBz()))

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

#

def Vmax_Decreasing_SS(number_of_nuclei, field_strength, theta, phi, file_name = None):
    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()

    Nuclei_list = list(Sum_of_Squares_Dictionary().keys())
    print(Nuclei_list)

    for i in range(number_of_nuclei):
        x = Nuclei_list[i]

        x.add_to_simulation()     # Add the first (number_of_nuclei) nuclei to the system, in decreasing order of SS !

    Vmax_value = Vmax(field_strength, theta, phi, display=True)

    df = pd.DataFrame({'Number of Nuclei': [number_of_nuclei], 'Vmax': [Vmax_value]})

    if file_name is not None:
        print(f'File name: {file_name}')
        df.to_csv(file_name)

    return Vmax_value





print(TrpH_HFITs[8])