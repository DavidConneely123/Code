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


# We can define the singlet projection operator

def Ps():
    return 1/4 * scipy.sparse.kron(i4, Nucleus.Identity_Matrix_of_Nuclear_Spins()) - Nucleus.SAx()*Nucleus.SBx() - Nucleus.SAy()*Nucleus.SBy() - Nucleus.SAz()*Nucleus.SBz()

# Can also define the Zeeman Hamiltonian (NB in s^-1)

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 1*Nucleus.SBz()))

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

# Creating some blank nucleus objects

"""
for i in range(20):
    name = f'Nuc{i}'
    Nucleus(name=name, spin=1 / 2, is_isotopologue=False, hyperfine_interaction_tensor= np.zeros((3,3)))
"""

def tensor_list(number_of_tensors, axiality, off_diagonals = False):

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



        list_of_tensors.append(tensor)

    return list_of_tensors

number_of_nuclei = 8

N5 = Nucleus('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
N10 = Nucleus('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
C8 = Nucleus('C8', 1/2, np.array([[1.10324799, -0.34094734, 0.00], [-0.34094734, 1.55501846, 0.00], [0.00, 0.00, 46.49118]])*1e6, is_isotopologue=True)
C5A = Nucleus('C5A', 1/2, np.array([[-16.39801673, -1.7214494, 0.00], [-1.7214494, -13.97962665, 0.00], [0.00, 0.00, -34.8079]])*1e6 , is_isotopologue=True)
C6 = Nucleus('C6', 1/2, np.array([[0.570117858, 0.0107777895, 0], [0.0107777895, 0.703398326, 0], [0, 0, 38.0512000]])*1e6, is_isotopologue=True)
C4A = Nucleus('C4A', 1/2, np.array([[-16.80154633, 1.29779775, 0.00], [1.29779775, -15.64680962, 0.00], [0.00, 0.00, 18.27357]])*1e6, is_isotopologue=True)
C7 = Nucleus('C7', 1/2, np.array([[-9.1508731, -0.46094791, 0.00], [-0.46094791, -9.363066042, 0.00], [0.00, 0.00, -22.43097]])*1e6, is_isotopologue=True)
C9A = Nucleus('C9A', 1/2, np.array([[-1.99205207, -0.20734695, 0.00], [-0.20734695, -1.41081737, 0.00], [0.00, 0.00, 18.58458]])*1e6, is_isotopologue=True)



"""
list_of_tensors = tensor_list(number_of_nuclei, 0.80)

#for tensor in list_of_tensors:
#   print(tensor, '\n \n')

for i in range(number_of_nuclei):
        Nucleus.all[i].hyperfine_interaction_tensor = list_of_tensors[i] * 1e6
"""



for nucleus in Nucleus.all[0:number_of_nuclei]:
    nucleus.add_to_simulation()

eigenvalues_total, eigenvectors_total = np.linalg.eigh(Dense_Hamiltonian(0.05,0,0))
idx = eigenvalues_total.argsort()[::-1]
eigenvalues_total = eigenvalues_total[idx]



Nucleus.reset_simulation()
Nucleus.remove_all_from_simulation()


for nucleus in Nucleus.all[0:int(number_of_nuclei/2)]:
    nucleus.add_to_simulation()

eigenvalues_1, eigenvectors_1 = np.linalg.eig(Dense_Hamiltonian(0.05,0,0))
idx = eigenvalues_1.argsort()[::-1]
eigenvalues_1 = eigenvalues_1[idx]

Nucleus.reset_simulation()
Nucleus.remove_all_from_simulation()


for nucleus in Nucleus.all[int(number_of_nuclei/2) : number_of_nuclei]:
    nucleus.add_to_simulation()

eigenvalues_2, eigenvectors_2 = np.linalg.eig(Dense_Hamiltonian(0,0,0))
idx = eigenvalues_2.argsort()[::-1]
eigenvalues_2 = eigenvalues_2[idx]

eigenvalues_total = np.ndarray.tolist(eigenvalues_total)



fig1, ax1 = plt.subplots()

for eigenvalue in eigenvalues_total:
    real_eigenvalue = np.real(eigenvalue)
    x = np.linspace(1,2.5,100)
    y = np.linspace(real_eigenvalue,real_eigenvalue,100)
    ax1.plot(x,y, color = 'red')


sums_of_eigenvalues = []
for eigenvalue2 in eigenvalues_2:
    for eigenvalue1 in eigenvalues_1:
        sums_of_eigenvalues.append(eigenvalue2 + eigenvalue1)

sums_of_eigenvalues.sort(reverse=True)
sums_of_eigenvalues = sums_of_eigenvalues[0::4]


for eigenvalue in sums_of_eigenvalues:
    real_eigenvalue = np.real(eigenvalue)
    x = np.linspace(2.5,4,100)
    y = np.linspace(real_eigenvalue,real_eigenvalue,100)
    ax1.plot(x,y, color = 'black')


deviation_list = [ '%.2f' %np.real(100*((eigenvalues_total[i] - sums_of_eigenvalues[i])/eigenvalues_total[i])) for i in range(len(eigenvalues_total)) ]

print(deviation_list)

print('\n \n')



eigenvalues_total = [i/1e6 for i in eigenvalues_total] # Changing to MHz for convenience
sums_of_eigenvalues = [i/1e6 for i in sums_of_eigenvalues]

fig2, ax2 = plt.subplots()

def Histogram_Height_Exact(v, dv=0.5):  # NB v in MHz
    sum = 0
    for i in range(len(eigenvalues_total)):
        for j in range(i,len(eigenvalues_total)):
            if v - dv / 2 < eigenvalues_total[i] - eigenvalues_total[j] < v + dv / 2:  # Find states i,j with a coherence in the range v +/- dv MHz
                sum+=1

    return sum.real

def Action_Spectrum_Exact(vrange):
        Histogram_Heights_Exact = []
        for v in range(1, vrange):
            print(Histogram_Height_Exact(v), v)
            Histogram_Heights_Exact.append(Histogram_Height_Exact(v))

        ax2.bar(np.linspace(1, len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)), Histogram_Heights_Exact, color = 'red')


def Histogram_Height_Sum(v, dv=0.5):  # NB v in MHz
    sum = 0
    for i in range(len(sums_of_eigenvalues)):
        for j in range(i,len(sums_of_eigenvalues)):
            if v - dv / 2 < sums_of_eigenvalues[i] - sums_of_eigenvalues[j] < v + dv / 2:  # Find states i,j with a coherence in the range v +/- dv MHz
                sum += 1

    return sum.real

def Action_Spectrum_Sum(vrange):
        Histogram_Heights_Sum = []
        for v in range(1, vrange):
            print(Histogram_Height_Sum(v), v)
            Histogram_Heights_Sum.append(Histogram_Height_Sum(v))

        ax2.bar(np.linspace(1, len(Histogram_Heights_Sum), len(Histogram_Heights_Sum)), Histogram_Heights_Sum, color = 'black')


def Action_Spectrum_Difference(vrange):
    fig3, ax3 = plt.subplots()
    Histogram_Heights_Differences = []
    for v in range(1, vrange):
        if Histogram_Height_Exact(v) != 0:
            Histogram_Heights_Differences.append(np.abs((Histogram_Height_Sum(v) - Histogram_Height_Exact(v))/Histogram_Height_Exact(v))*100)
        else:
            Histogram_Heights_Differences.append(0)

    ax3.bar(np.linspace(1, len(Histogram_Heights_Differences), len(Histogram_Heights_Differences)), Histogram_Heights_Differences, color='blue')
    ax3.set_ylabel('Percentage Error in Bar Height')

def All_Action_Spectra(vrange):
    Histogram_Heights_Exact = []
    for v in range(1, vrange):
        Histogram_Height_Exact_v = Histogram_Height_Exact(v)
        print(Histogram_Height_Exact_v, v)
        Histogram_Heights_Exact.append(Histogram_Height_Exact_v)

    ax2.bar(np.linspace(1, len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)), Histogram_Heights_Exact, width=-0.3, align = 'edge', color='red')

    Histogram_Heights_Sum = []
    for v in range(1, vrange):
        Histogram_Heights_Sum_v = Histogram_Height_Sum(v)
        print(Histogram_Heights_Sum_v, v)
        Histogram_Heights_Sum.append(Histogram_Heights_Sum_v)

    ax2.bar(np.linspace(1, len(Histogram_Heights_Sum), len(Histogram_Heights_Sum)), Histogram_Heights_Sum, width = 0.3, align = 'edge',color='black')

    """
    Histogram_Heights_Differences = []
    for v in range(1, vrange):
        if Histogram_Heights_Exact[v-1] != 0:
            Histogram_Heights_Differences.append(np.abs(( (Histogram_Heights_Sum[v-1] - Histogram_Heights_Exact[v-1]) / Histogram_Heights_Exact[v-1]) * 100))
        else:
            Histogram_Heights_Differences.append(0)

    ax3.bar(np.linspace(1, len(Histogram_Heights_Differences), len(Histogram_Heights_Differences)),
            Histogram_Heights_Differences, color='blue')
    ax3.set_ylabel('Percentage Error in Bar Height')
    """

All_Action_Spectra(170)


plt.show()