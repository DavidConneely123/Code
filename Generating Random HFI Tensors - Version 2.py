import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.constants import physical_constants
import time

startTime = time.perf_counter()

# We wish to generate a set of 20 random, symmetric HFI tensors with a uniform distribution of values between +/- 1 mT
# which corresponds to ~ +/- 30 MHz

def tensor_list(number_of_tensors):

    list_of_tensors = []
    for i in range(number_of_tensors):
        random_tensor = np.random.uniform(-30,30, size = (3,3))
        symmetric_random_tensors = (random_tensor+random_tensor.T)/2
        list_of_tensors.append(symmetric_random_tensors)


    sum = 0
    for tensor in list_of_tensors:
        sum += (1 / 3 * np.sum(tensor ** 2)) * (1/2) * (1/2 + 1)   # Sigma (NB with only spin-1/2 nuclei)

    sigma = np.sqrt(1/3 * sum)

    scaling = 28.85778/sigma                    # We scale the HFI tensors so that they match the unlabelled FAD (so Vmax should be rougly 115 MHz)

    list_of_tensors = np.multiply(list_of_tensors, scaling)

    return list_of_tensors


# Defining the basic matrix representations and constants involved

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

# Importing our Nucleus Class

class Nucleus:
    # Class attributes

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

        # Every time we initialise a new instance of the class it is added to the list 'all' and if it is a 13C it is
        # also added to the 'Isotopologues' list

        # NB! Initially no nuclei are included in the simulation

        Nucleus.all.append(self)

        if self.is_Isotopologue:
            Nucleus.isotopologues.append(self)

    # This simply changes how the instance objects appear when we print Nucleus.all to the terminal

    def __repr__(self):
        return f"Nucleus('{self.name}', 'spin-{self.spin}') "


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

# We initiate as many instances of the Nucleus class as we desire, named Nuc{i} and with HFI tensors taken from the list created above
# NB; note we here convert our HFI Tensors from MHz (as above) to Hz

def H_zee(field_strength, theta, phi):
    return (gyromag/(2*np.pi)) * field_strength * (np.sin(theta) * np.cos(phi) * (Nucleus.SAx() + Nucleus.SBx()) + np.sin(theta) * np.sin(phi) * (Nucleus.SAy() + Nucleus.SBy()) + np.cos(theta) * (Nucleus.SAz() + 0*Nucleus.SBz()))

# Using the matrix representations defined in the class we can now also define the hyperfine interaction term in the hamiltonian:

def H_hyperfine():
    sum = 0
    for nucleus in Nucleus.nuclei_included_in_simulation:

        val = nucleus.hyperfine_interaction_tensor[0,0]*nucleus.SAxIix() + nucleus.hyperfine_interaction_tensor[0,1]*nucleus.SAxIiy() + nucleus.hyperfine_interaction_tensor[0,2]*nucleus.SAxIiz() + nucleus.hyperfine_interaction_tensor[1,0]*nucleus.SAyIix() + nucleus.hyperfine_interaction_tensor[1,1]*nucleus.SAyIiy() + nucleus.hyperfine_interaction_tensor[1,2]*nucleus.SAyIiz() + nucleus.hyperfine_interaction_tensor[2,0]*nucleus.SAzIix() + nucleus.hyperfine_interaction_tensor[2,1]*nucleus.SAzIiy() + nucleus.hyperfine_interaction_tensor[2,2]*nucleus.SAzIiz()
        sum += val

    return sum

def Sparse_Hamiltonian(field_strength, theta, phi):
    return H_hyperfine() + H_zee(field_strength, theta, phi)

def Vmax(field_strength, theta, phi, display = False, display_eigenvalues = False):
    if display:
        print(f' \n Field strength = {field_strength} mT , theta = {theta}, phi = {phi} \n __________________________________________________________________')

    Hspar = Sparse_Hamiltonian(field_strength, theta, phi)

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


def Overestimation_of_Vmax_by_Summing(number_of_nuclei, display = False):
    list_of_tensors = tensor_list(number_of_nuclei)

    for i in range(number_of_nuclei):
        name = f'Nuc{i}'
        Nucleus(name = name, spin = 1/2, is_Isotopologue = False, hyperfine_interaction_tensor = list_of_tensors[i]*1e6)

    for nucleus in Nucleus.all[:int(number_of_nuclei/2)]:
        nucleus.add_to_simulation()

    Vmax_firsthalf = Vmax(0,0,0, display=True)

    Nucleus.reset_simulation()
    Nucleus.remove_all_from_simulation()

    for nucleus in Nucleus.all[int(number_of_nuclei/2):]:
        nucleus.add_to_simulation()

    Vmax_secondhalf = Vmax(0,0,0, display = True)


    Nucleus.reset_simulation()

    Vmax_total = Vmax(0,0,0)

    if display:
         print(f'Overestimation is { (((Vmax_firsthalf + Vmax_secondhalf) - Vmax_total) / Vmax_total)*100}%')

    Nucleus.nuclei_included_in_simulation = []
    my_nuclei={}
    return(((Vmax_firsthalf + Vmax_secondhalf) - Vmax_total) / Vmax_total)*100


def Mean_overestimation(number_of_nuclei, number_of_iterations):
    i=0
    Overestimation_Sum = 0
    while i<number_of_iterations:
        Overestimation_Sum += Overestimation_of_Vmax_by_Summing(number_of_nuclei)
        i += 1
        Nucleus.all = []


    Overestimation_mean = Overestimation_Sum/number_of_iterations
    Overestimation_mean = round(Overestimation_mean, 2)
    print(f'Mean overestimation with {number_of_nuclei} nuclei = {Overestimation_mean}%')
    return Overestimation_mean



print(tensor_list(1))