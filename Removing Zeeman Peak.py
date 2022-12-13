import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fftfreq
from matplotlib.widgets import Slider, Button


def FT_signal(zip_file_directory, number_sample_points, time_interval):

    # We load the relevant zip file of .npy files and convert this into a list of arrays containing the relevant signals

    zip = np.load(zip_file_directory)
    file_names = [file_name for file_name in zip.files[1:]]          #NB, first element in zip.files is just the directory name, not a file itself...
    ps_signals = [zip[file_name] for file_name in file_names]


    # NB!!! Our signal is real, so could potentially replace fft with rfft and then wouldn't have to the [0:N//2] slicing
    # and would also be faster, something to consider...

    yf=0
    for y in ps_signals:
        yf_current = scipy.fft(y - np.mean(y)) # Calculate the FT signal for a given time-domain signal, NB: subtracting off np.mean(y) removes the DC bias !!
        yf += yf_current  # Add up all the FT signals




    # Here we make the correct frequency-binning for the Fourier Transform #NB: need to make this match the time interval and time step used in the given signal !!!

    time_interval = time_interval           # Usually in the range 1-10us

    N = number_sample_points                     # This is the number of sample points
    T = time_interval/N          # This is the time-step between each sample point

    t = np.linspace(0, N*T, N)

    # Now we Fourier Transform our Ps(t) signal and plot the spectrum given (NB: we only need to consider the first N//2 terms
    # as the rest will give the negative frequencies, which is just a mirror image of the spectrum as Ps(t) is real)

    tf = fftfreq(N, T)[0:N//2]
    yf = np.abs(yf)[0:N//2]
    yf = yf/sum(yf[1:])

    #ax1.plot(tf/1e6, yf)   # NB; converting frequency axis (x-axis, tf) to MHz






signals = np.load('Fourier Signals/6SPIN_100000.zip')
file_names = [file_name for file_name in signals.files[1:]]          #NB, first element in zip.files is just the directory name, not a file itself...
ps_signals = [signals[file_name] for file_name in file_names]

signal = sum(signal for signal in ps_signals)


# The parametrized function to be plotted
def f(t, amplitude, frequency, midpoint):
    return signal - (midpoint + amplitude * np.cos(2 * np.pi * frequency * t))

def g(t, amplitude, frequency, midpoint):
    return midpoint + amplitude * np.cos(2 * np.pi * frequency * t)

def h(t, amplitude, frequency, midpoint):
    return signal



t = np.linspace(0,10e-6,100000)

# Define initial parameters
init_amplitude = 20
init_frequency = 1.4e6
init_midpoint = 50

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line1, = ax.plot(t, f(t, init_amplitude, init_frequency, init_midpoint), lw=2)
line2, = ax.plot(t, g(t, init_amplitude, init_frequency, init_midpoint), lw=2)
line3, = ax.plot(t, h(t, init_amplitude, init_frequency, init_midpoint), lw=2)
ax.set_xlabel('Time [microseconds]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(ax=axfreq,label='Frequency [MHz]',valmin=1.3705e6,valmax=1.373872e6,valinit=init_frequency,)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(ax=axamp,label="Amp",valmin=22.307,valmax=26,valinit=init_amplitude,orientation="vertical")

# Make a vertically oriented slider to control the midpoint
axmid = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
mid_slider = Slider(ax=axmid,label="Mid",valmin=0,valmax=100,valinit=init_midpoint,orientation="vertical")

# The function to be called anytime a slider's value changes
def update(val):
    line1.set_ydata(f(t, amp_slider.val, freq_slider.val, mid_slider.val))
    line2.set_ydata(g(t, amp_slider.val, freq_slider.val, mid_slider.val))
    line3.set_ydata(h(t, amp_slider.val, freq_slider.val, mid_slider.val))
    fig.canvas.draw_idle()

# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)
mid_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

#plt.show()



best_zeeman_offset = 63.2 + 23.347* np.cos(2 * np.pi * 1.373716e6 * t)






signal = signal - best_zeeman_offset

yf = scipy.fft(signal - np.mean(signal))


time_interval = 10e-6           # Usually in the range 1-10us

N = 100000                     # This is the number of sample points
T = time_interval/N          # This is the time-step between each sample point

t = np.linspace(0, N*T, N)

# Now we Fourier Transform our Ps(t) signal and plot the spectrum given (NB: we only need to consider the first N//2 terms
# as the rest will give the negative frequencies, which is just a mirror image of the spectrum as Ps(t) is real)

tf = fftfreq(N, T)[0:N//2]
yf = np.abs(yf)[0:N//2]
yf = yf/sum(yf[1:])


fig2, ax2 = plt.subplots()
ax2.plot(tf/1e6, yf)   # NB; converting frequency axis (x-axis, tf) to MHz

plt.show()
