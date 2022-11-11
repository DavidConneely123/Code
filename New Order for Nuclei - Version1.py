import itertools
import random
import time
from multiprocessing import Pool
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
    subsystem_b = []
    subsystem_a = []


    # Initialise an instance of the class (i.e. a single nucleus)

    def __init__(self, name: str, spin: float, hyperfine_interaction_tensor, is_isotopologue=False, in_subsystem_b = False):
        # Run validations to the received arguments
        assert spin % 0.5 == 0, f'Spin must be an integer or half-integer value !'

        # Assigning properties to self object

        self.name = name
        self.spin = spin
        self.is_isotopologue = is_isotopologue
        self.hyperfine_interaction_tensor = hyperfine_interaction_tensor
        self.in_subsystem_b = in_subsystem_b

        # Actions to execute

        # Every time we initialise a new instance of the class it is added to the list 'all' and if it is a 13C it is
        # also added to the 'Isotopologues' list

        # NB! Initially no nuclei are included in the simulation

        Nucleus.all.append(self)

        # If the nucleus is an isotope or is in subsystem_b they are added to the relevant list

        if self.is_isotopologue:
            Nucleus.isotopologues.append(self)

        if self.in_subsystem_b:
            Nucleus.subsystem_b.append(self)

        if not self.in_subsystem_b:
            Nucleus.subsystem_a.append(self)


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

    @classmethod
    def Ib_dimension(cls):
        Ib_dimension = 1
        for nucleus in Nucleus.subsystem_b:
            if nucleus in Nucleus.nuclei_included_in_simulation:
                if nucleus.spin == 1 / 2:
                    Ib_dimension *= 2
                if nucleus.spin == 1:
                    Ib_dimension *= 3

        return Ib_dimension

    @classmethod
    def Ia_dimension(cls):
        Ia_dimension = 1
        for nucleus in Nucleus.subsystem_a:
            if nucleus in Nucleus.nuclei_included_in_simulation:
                if nucleus.spin == 1 / 2:
                    Ia_dimension *= 2
                if nucleus.spin == 1:
                    Ia_dimension *= 3

        return Ia_dimension

    def Ib_before_dimension(self):
        if self in Nucleus.subsystem_a:
            return Nucleus.Ib_dimension()

        else:
            Ib_before_length = 1
            for nucleus in Nucleus.nuclei_included_in_simulation[0: Nucleus.nuclei_included_in_simulation.index(self)]:
                if nucleus in Nucleus.subsystem_b:
                    if nucleus.spin == 1:
                        Ib_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_before_length *= 2

            return Ib_before_length

    def Ib_after_dimension(self):
        if self in Nucleus.subsystem_a:
            return 1

        else:
            Ib_after_length = 1
            for nucleus in Nucleus.nuclei_included_in_simulation[Nucleus.nuclei_included_in_simulation.index(self) + 1:]:
                if nucleus in Nucleus.subsystem_b:
                    if nucleus.spin == 1:
                        Ib_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ib_after_length *= 2

            return Ib_after_length

    def Ia_before_dimension(self):
        if self in Nucleus.subsystem_b:
            return 1

        else:
            Ia_before_length = 1
            for nucleus in Nucleus.nuclei_included_in_simulation[0: Nucleus.nuclei_included_in_simulation.index(self)]:
                if nucleus in Nucleus.subsystem_a:
                    if nucleus.spin == 1:
                        Ia_before_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_before_length *= 2

            return Ia_before_length

    def Ia_after_dimension(self):
        if self in Nucleus.subsystem_b:
            return Nucleus.Ia_dimension()

        else:
            Ia_after_length = 1
            for nucleus in Nucleus.nuclei_included_in_simulation[Nucleus.nuclei_included_in_simulation.index(self) + 1:]:
                if nucleus in Nucleus.subsystem_a:
                    if nucleus.spin == 1:
                        Ia_after_length *= 3
                    if nucleus.spin == 1 / 2:
                        Ia_after_length *= 2

            return Ia_after_length



    # We can make matrix representations for the nuclear operators

    def Ix(self):
        if self.in_subsystem_b:
            return scipy.sparse.kron(scipy.sparse.identity(self.Ib_before_dimension(), format='coo'), scipy.sparse.kron(self.sx(), scipy.sparse.kron(scipy.sparse.identity(self.Ib_after_dimension(), format='coo'), scipy.sparse.identity(self.Ia_dimension() * 4, format='coo'))))

        else:
            return scipy.sparse.kron(scipy.sparse.identity(self.Ib_dimension() * 4, format='coo'), scipy.sparse.kron(scipy.sparse.identity(self.Ia_before_dimension(), format='coo'), scipy.sparse.kron(self.sx(), scipy.sparse.identity(self.Ia_after_dimension(), format='coo'))))

    def Iy(self):
        if self.in_subsystem_b:
            return scipy.sparse.kron(scipy.sparse.identity(self.Ib_before_dimension(), format='coo'), scipy.sparse.kron(self.sy(), scipy.sparse.kron(scipy.sparse.identity(self.Ib_after_dimension(), format='coo'), scipy.sparse.identity(self.Ia_dimension() * 4, format='coo'))))

        else:
            return scipy.sparse.kron(scipy.sparse.identity(self.Ib_dimension() * 4, format='coo'), scipy.sparse.kron(scipy.sparse.identity(self.Ia_before_dimension(), format='coo'), scipy.sparse.kron(self.sy(), scipy.sparse.identity(self.Ia_after_dimension(), format='coo'))))

    def Iz(self):
        if self.in_subsystem_b:
            return scipy.sparse.kron(scipy.sparse.identity(self.Ib_before_dimension(), format='coo'), scipy.sparse.kron(self.sz(), scipy.sparse.kron(scipy.sparse.identity(self.Ib_after_dimension(), format='coo'), scipy.sparse.identity(self.Ia_dimension() * 4, format='coo'))))

        else:
            return scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension() * 4, format='coo'), scipy.sparse.kron(scipy.sparse.identity(self.Ia_before_dimension(), format='coo'), scipy.sparse.kron(self.sz(), scipy.sparse.identity(self.Ia_after_dimension(), format='coo'))))


    # Creating matrix representations for the electronic operators
    @classmethod
    def SAx(cls):
        SAx = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'), scipy.sparse.kron(sx_spinhalf, scipy.sparse.kron(i2, scipy.sparse.identity(Nucleus.Ia_dimension(), format ='coo'))))
        return SAx

    @classmethod
    def SAy(cls):
        SAy = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(sy_spinhalf, scipy.sparse.kron(i2, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SAy

    @classmethod
    def SAz(cls):
        SAz = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(sz_spinhalf, scipy.sparse.kron(i2, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SAz

    @classmethod
    def SBx(cls):
        SAx = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SAx

    @classmethod
    def SBy(cls):
        SAy = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SAy

    @classmethod
    def SBz(cls):
        SAz = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sz_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SAz


# Nuclei included in the simulation (NB HFI Tensor always in s^-1)


# We can define the singlet projection operator

def Ps():
    return 0.25 * scipy.sparse.identity(Nucleus.Ib_dimension() * 4 * Nucleus. Ia_dimension()) - Nucleus.SAx()*Nucleus.SBx() - Nucleus.SAy()*Nucleus.SBy() - Nucleus.SAz()*Nucleus.SBz()

# Can also define the Zeeman Hamiltonian (NB in s^-1)

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (1*Nucleus.SAz() + 0*Nucleus.SBz()))

# Can also define a term for the interaction of our electron spins with an rf field perpendicular to the geomagnetic field
# NB: note that the field strength of this rf field will not in general be the same as the geomagnetic field strenght !!!!

def H_perp(field_strength, theta, phi):
    return (gyromag / (2 * np.pi)) * field_strength * (np.cos(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.cos(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) - np.sin(theta) * (Nucleus.SAz() + 1 * Nucleus.SBz()))

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
        val = nucleus.hyperfine_interaction_tensor[0,0]*sax*ix + nucleus.hyperfine_interaction_tensor[0,1]*sax*iy + nucleus.hyperfine_interaction_tensor[0,2]*sax*iz + nucleus.hyperfine_interaction_tensor[1,0]*say*ix + nucleus.hyperfine_interaction_tensor[1,1]*say*iy + nucleus.hyperfine_interaction_tensor[1,2]*say*iz + nucleus.hyperfine_interaction_tensor[2,0]*saz*ix + nucleus.hyperfine_interaction_tensor[2,1]*saz*iy + nucleus.hyperfine_interaction_tensor[2,2]*saz*iz
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


def tensor_list(number_of_tensors, axiality, off_diagonals = False, not_random = False):

    list_of_tensors = []

    for _ in range(number_of_tensors):
        tensor = np.zeros((3,3))
        tensor[0,0] = random.uniform(-(1-axiality), (1-axiality))*10
        tensor[1,1] = random.uniform(-(1-axiality), (1-axiality))*10
        tensor[2,2] = random.uniform(-1,1)*10

        if off_diagonals:
            tensor[0,1] = random.uniform(-(1-axiality), (1-axiality))*0.2
            tensor[1,0] = random.uniform(-(1-axiality), (1-axiality))*0.2

            tensor[1, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 1] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

            tensor[0, 2] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2
            tensor[2, 0] = random.uniform(-(1 - axiality), (1 - axiality)) * 0.2

        if not_random:
            tensor[0,0] = 1-axiality
            tensor[1,1] = 1-axiality
            tensor[2,2] = axiality


        list_of_tensors.append(tensor)

    return list_of_tensors

# Here we define the nuclei involved in our simulation, either by creating some artificial nuclei

'''
Nb1 = Nucleus('Nb1', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), in_subsystem_b= True)
Nb2 = Nucleus('Nb2', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), in_subsystem_b= True)
Nb3 = Nucleus('Nb3', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), in_subsystem_b= True)

Na1 = Nucleus('Na1', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
Na2 = Nucleus('Na2', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
Na3 = Nucleus('Na2', 1/2, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
'''

def Rotation_Matrix(axis, theta):
    if axis == 'x':
        return np.array([[1, 0, 0] , [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    if axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)] , [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    if axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])

Rotation_Matrix('x', 0.5*np.pi)




N5 = Nucleus('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6, in_subsystem_b=True)
N10 = Nucleus('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6, in_subsystem_b=True)
C8 = Nucleus('C8', 1/2, np.array([[1.10324799, -0.34094734, 0.00], [-0.34094734, 1.55501846, 0.00], [0.00, 0.00, 46.49118]])*1e6, is_isotopologue=True, in_subsystem_b=True)
C5A = Nucleus('C5A', 1/2, np.array([[-16.39801673, -1.7214494, 0.00], [-1.7214494, -13.97962665, 0.00], [0.00, 0.00, -34.8079]])*1e6 , is_isotopologue=True)
C6 = Nucleus('C6', 1/2, np.array([[0.570117858, 0.0107777895, 0], [0.0107777895, 0.703398326, 0], [0, 0, 38.0512000]])*1e6, is_isotopologue=True)
C4A = Nucleus('C4A', 1/2, np.array([[-16.80154633, 1.29779775, 0.00], [1.29779775, -15.64680962, 0.00], [0.00, 0.00, 18.27357]])*1e6, is_isotopologue=True)



for nucleus in [C4A]:
    nucleus.hyperfine_interaction_tensor[0,0], nucleus.hyperfine_interaction_tensor[2,2] = nucleus.hyperfine_interaction_tensor[2,2] , nucleus.hyperfine_interaction_tensor[0,0]

    print(np.divide(nucleus.hyperfine_interaction_tensor,1e6), '\n')


for nucleus in Nucleus.subsystem_a:
    nucleus.add_to_simulation()

ha = Dense_Hamiltonian(0.05,0,0)


Nucleus.reset_simulation()
Nucleus.remove_all_from_simulation()

for nucleus in Nucleus.subsystem_b:
    nucleus.add_to_simulation()

hb = Dense_Hamiltonian(0.0,0,0)     #NB: ha, hb are the hamiltonians in the reduced bases of the two subsystems !! (dimensionalities do NOT match!!!)

Nucleus.reset_simulation()
Nucleus.remove_all_from_simulation()
Nucleus.add_all_to_simulation()


Ia = np.identity(Nucleus.Ia_dimension())
Ib = np.identity(Nucleus.Ib_dimension())

HbHa = np.kron(hb, Ia) * np.kron(Ib, ha)            # NB: H =  ha ⊕ hb  =  Ib⊗ha + hb⊗Ia = Ha + Hb

HaHb = np.kron(Ib, ha) * np.kron(hb, Ia)

Norm_of_Commutator = np.linalg.norm(HbHa-HaHb, ord='fro')

Nucleus.reset_simulation()
Nucleus.remove_all_from_simulation()

print(Norm_of_Commutator/1e14)




