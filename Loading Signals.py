import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fftfreq

# We load the relevant zip file of .npy files and convert this into a list of arrays containing the relevant signals

zip = np.load('Fourier Signals/6SPIN_100000.zip')
file_names = [file_name for file_name in zip.files[1:]]          #NB, first element in zip.files is just the directory name, not a file itself...
ps_signals = [zip[file_name] for file_name in file_names]


# NB!!! Our signal is real, so could potentially replace fft with rfft and then wouldn't have to the [0:N//2] slicing
# and would also be faster, something to consider...

yf=0
for y in ps_signals:
    yf_current = scipy.fft(y) # Calculate the FT signal for a given time-domain signal
    yf += yf_current  # Add up all the FT signals

# Here we make the correct frequency-binning for the Fourier Transform #NB: need to make this match the time interval and time step used in the given signal !!!


time_interval = 10e-6           # Usually in the range 1-10us

N = 100000                     # This is the number of sample points
T = time_interval/N          # This is the time-step between each sample point


t = np.linspace(0, N*T, N)

# Now we Fourier Transform our Ps(t) signal and plot the spectrum given (NB: we only need to consider the first N//2 terms
# as the rest will give the negative frequencies, which is just a mirror image of the spectrum as Ps(t) is real)



tf = fftfreq(N, T)[0:N//2]


fig1, ax1 = plt.subplots()
ax1.plot(tf/1e6, np.abs(yf)[0:N//2])
plt.show()
