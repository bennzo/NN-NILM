import numpy as np
import itertools
import random
import utilities as util


def fft(I_t, noise=False):
    n = len(I_t)  # length of the signal
    if noise:
        I_t += np.random.normal(0,1,n)*(I_t*util.preproc_config['noise_percentage'])
    I_fft = np.fft.fft(I_t)/(n/2)                                                       # fft computing and normalization
    I_fft = I_fft[range(int(n/2))]                                                      # one side frequency range
    return I_fft


def fft_amp_phase(I_fft):
    I_fft_amp = np.abs(I_fft)                                                           # calculate amplitude
    I_fft_phase = np.angle(I_fft)                                                       # calculate phase
    I_fft_phase[I_fft_amp < util.preproc_config['threshold']] = 0                       # non maximum supression
    I_fft_amp[I_fft_amp < util.preproc_config['threshold']] = 0
    I_fft_phase = I_fft_phase * (360 / (2 * np.pi))
    return I_fft_amp, I_fft_phase


def fft2input(I_fft_amp, I_fft_phase, Fs):
    harmonies = np.array([1, 3, 5, 7, 9, 11, 13])*round((len(I_fft_amp)*2)/(Fs/50))
    harmonies = harmonies.astype(int)
    IN = np.concatenate((I_fft_amp[harmonies],I_fft_phase[harmonies]))
    return IN


