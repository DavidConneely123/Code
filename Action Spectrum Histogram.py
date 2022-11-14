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

TrpH_HFITs = np.load('TrpH_HFITS_MHz_ROTATED.npy')

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

        # If the nucleus is an isotope add to the list
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
                Nucleus.subsystem_a = []
                Nucleus.subsystem_b = []

        except:
            raise Exception('The nucleus you tried to remove is not currently part of the simulation !')

    def add_to_simulation(self):
        Nucleus.nuclei_included_in_simulation.append(self)

        if self.in_subsystem_b:
            Nucleus.subsystem_b.append(self)

        if not self.in_subsystem_b:
            Nucleus.subsystem_a.append(self)

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

    def move_to_subsystem_b(self):
        self.in_subsystem_b = True


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
        SBx = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sx_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SBx

    @classmethod
    def SBy(cls):
        SBy = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sy_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SBy

    @classmethod
    def SBz(cls):
        SBz = scipy.sparse.kron(scipy.sparse.identity(Nucleus.Ib_dimension(), format='coo'),scipy.sparse.kron(i2, scipy.sparse.kron(sz_spinhalf, scipy.sparse.identity(Nucleus.Ia_dimension(), format='coo'))))
        return SBz


# Nuclei included in the simulation (NB HFI Tensor always in s^-1)


# We can define the singlet projection operator

def Ps():
    return 0.25 * scipy.sparse.identity(Nucleus.Ib_dimension() * 4 * Nucleus. Ia_dimension()) - Nucleus.SAx()@Nucleus.SBx() - Nucleus.SAy()@Nucleus.SBy() - Nucleus.SAz()@Nucleus.SBz()

# Can also define the Zeeman Hamiltonian (NB in s^-1)

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 0*Nucleus.SBz()))

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

N9 = Nucleus('N9', 1, TrpH_HFITs[8]*1e6)
H23 = Nucleus('H23', 1/2, TrpH_HFITs[22]*1e6)
H18 = Nucleus('H18', 1/2, TrpH_HFITs[17]*1e6)
H24 = Nucleus('H24', 1/2, TrpH_HFITs[23]*1e6)

for nucleus in Nucleus.all:
    nucleus.add_to_simulation()




H_dense = Dense_Hamiltonian(0.05,0,0)
eigenvalues_total, eigenvectors_total = np.linalg.eigh(H_dense)
idx = eigenvalues_total.argsort()[::-1]
eigenvalues_total = eigenvalues_total[idx]
eigenvectors_total = eigenvectors_total[idx]


ps_tot = Ps()
#ps_tot = eigenvectors_total.conj().T @ ps_tot @ eigenvectors_total
ps_tot = np.linalg.inv(eigenvectors_total) @ ps_tot @ eigenvectors_total

H_perp = H_perp(1e-5,0,0)
H_perp = eigenvectors_total.conj().T @ H_perp @ eigenvectors_total

eigenvalues_total = [i/1e6 for i in eigenvalues_total]


dv = 0.5
def Histogram_Height_Exact(v):  # NB v in MHz
    sum = 0
    for i in range(len(eigenvalues_total)):
        for j in range(i,len(eigenvalues_total)):
            if v - dv < eigenvalues_total[i] - eigenvalues_total[j] < v + dv :  # Find states i,j with a coherence in the range v +/- dv MHz
                t1 = np.abs(ps_tot[i,i] - ps_tot[j,j])
                t2 = (np.abs(H_perp[i,j])**2)
                sum += t1


    return sum.real

def Histogram_Height_Transition_Probability(v):
    sum = 0
    for i in range(len(eigenvalues_total)):
        for j in range(i, len(eigenvalues_total)):
            if v - dv < eigenvalues_total[i] - eigenvalues_total[j] < v + dv:  # Find states i,j with a coherence in the range v +/- dv MHz
                t1 = np.abs(ps_tot[i, i] - ps_tot[j, j])
                t2 = (np.abs(H_perp[i, j]) ** 2)
                sum += t2

    return sum.real

fig1, ax1 = plt.subplots()

def Action_Spectrum(vrange):
    # We can use multiprocessing to calculate the different histogram heights all in parallel, giving large speed up to computation

    if __name__ == '__main__':
        with Pool() as p:
            Histogram_Heights_Exact = p.map(Histogram_Height_Exact, [v for v in np.linspace(1, vrange, int(vrange/dv))]) # Multiprocessing so can run all these calculations in parallel !


    ax1.bar(np.linspace(1, dv*len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)), Histogram_Heights_Exact, width=-0.5, align = 'edge', color='red')
    ax1.scatter(np.linspace(dv, dv*len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)), Histogram_Heights_Exact)


def Action_Spectrum_Transition_Probability(vrange):
    if __name__ == '__main__':
        with Pool() as p:
            Histogram_Heights_Exact = p.map(Histogram_Height_Transition_Probability, [v for v in np.linspace(1, vrange,
                                                                                            int(vrange / dv))])  # Multiprocessing so can run all these calculations in parallel !

    ax1.bar(np.linspace(1, dv * len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)), Histogram_Heights_Exact,width=0.5, align='edge', color='green')
    ax1.scatter(np.linspace(dv, dv * len(Histogram_Heights_Exact), len(Histogram_Heights_Exact)),Histogram_Heights_Exact)


Action_Spectrum(100)
#Action_Spectrum_Transition_Probability(100)


plt.show()
