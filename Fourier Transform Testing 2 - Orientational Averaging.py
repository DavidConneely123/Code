import pandas as pd

from RadicalPair import *
from Programs import *
import matplotlib.pyplot as plt
from scipy.fft import fftfreq


# Setting up and defining the nuclei included


N5 = RadicalA('N5', 1, np.array([[-2.79, -0.08, 0], [-0.08, -2.45, 0], [0, 0, 49.24]]) * 1e6)
#N10 = RadicalA('N10', 1, np.array([[-0.42, -0.06, 0.00], [-0.06, -0.66, 0.00], [0.00, 0.00, 16.94]]) * 1e6)
#H8_1 = RadicalA('H8_1', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)
#H8_2 = RadicalA('H8_2', 1 / 2, np.array([[12.33, 0.00, 0.00], [0.00, 12.33, 0.00], [0.00, 0.00, 12.33]]) * 1e6)


for nucleus in RadicalA.all_A: nucleus.add_to_simulation()


yf = 0
iteration = 0
for field_direction in list_of_field_directions[0:10]:
    Theta, Phi = field_direction
    Magnetic_Field_Strength = 0.05

    Field = (Magnetic_Field_Strength, Theta, Phi)

    # Here we are calculating Ps(t) - but in theory we would be using the SU(Z) sampling in order to do this!

    Hamiltonian = Sparse_Hamiltonian_total_basis(Magnetic_Field_Strength, Theta, Phi)
    eigenvalues, eigenvectors = np.linalg.eigh(Densify(Hamiltonian))
    Singlet_Projection_Matrix = eigenvectors.conj().T @ Singlet_Projection_total_basis() @ eigenvectors

    eigenvalues = eigenvalues*2*np.pi   # NB: don't forget about this in order to get peaks to appear at the right place !!!
    M = RadicalA.I_radical_dimension()/2


    def Ps(t):
        sum = 0
        for i in range(int(M*4)):
            for j in range(int(M*4)):
                    sum += np.cos((eigenvalues[j] - eigenvalues[i])*t)*(np.square(np.abs(Singlet_Projection_Matrix[i,j])))

        return sum / M

    # Here we set the time interval which we are propagating over (good to use about 1us - 10us usually...)

    time_interval = 10e-6

    N = 4000                     # This is the number of sample points
    T = time_interval/N          # This is the time-step between each sample point


    x = np.linspace(0, N*T, N)
    y = [np.real(Ps(t)) for t in x]

    # Now we Fourier Transform our Ps(t) signal and plot the spectrum given (NB: we only need to consider the first N//2 terms
    # as the rest will give the negative frequencies, which is just a mirror image of the spectrum as Ps(t) is real)

    yf_current = scipy.fft(y)
    #np.save(f'FT Signal - 4 Nuclei (N5, N10, H8_1, H8_2) - Field = {Field}', yf_current)

    yf += yf_current
    xf = fftfreq(N, T)[0:N//2]

    print(f'Completed Iteration {iteration}')
    iteration += 1


Vmax = Vmax(Magnetic_Field_Strength, Theta, Phi)
Cut_off_line_y = np.linspace(-50,50,100)
Cut_off_line_x = [Vmax for v in range(len(Cut_off_line_y))]

fig1, ax1 = plt.subplots()
ax1.plot(xf/1e6, np.abs(yf)[0:N//2])  #NB; dividing by 1e6 so that the x-axis is in MHz !!!!
#ax1.plot(Cut_off_line_x, Cut_off_line_y, color = 'red')
plt.show()

