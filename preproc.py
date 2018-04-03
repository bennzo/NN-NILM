import numpy as np

def fft(I_t):
    n = len(I_t)  # length of the signal
    I_fft = np.fft.fft(I_t)/(n/2)   # fft computing and normalization
    I_fft = I_fft[range(int(n/2))]  # one side frequency range
    return I_fft

def fft_amp_phase(I_fft, threshold=0.1):
    I_fft_amp = np.abs(I_fft)                   # calculate amplitude
    I_fft_phase = np.angle(I_fft)               # calculate phase
    I_fft_phase[I_fft_amp < threshold] = 0      # non maximum supression
    I_fft_amp[I_fft_amp < threshold] = 0
    I_fft_phase = I_fft_phase * (360 / (2 * np.pi))
    return I_fft_amp, I_fft_phase

