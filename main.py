import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc

#utilities.gen_sum()

I, I_label = utilities.load_sum('data\lab-noiseless\\')

# F, A, P, I = utilities.load_signal(6, '')
# print(F)
# print(A)
# print(P)

I_t = I[0:int(preproc.opt['Fs']*preproc.opt['sample_time'])]
I_fft = preproc.fft(I_t)
I_fft_amp, I_fft_phase  = preproc.fft_amp_phase(I_fft)


Fs = preproc.opt['Fs']
n = len(I_t)                # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T                   # two sides frequency range
frq = frq[range(int(n/2))]  # one side frequency range

#print(preproc.fft2input(I_fft_amp, I_fft_phase, Fs))

fig1 = plt.figure()
p1 = fig1.add_subplot(111)
p1.plot(range(n), I_t, 'b')

fig2 = plt.figure()
p2 = fig2.add_subplot(121)
p2.plot(frq, I_fft_amp, 'r.')
p2.vlines(frq,[0],I_fft_amp)
p2.grid(True)

p3 = fig2.add_subplot(122)
p3.plot(frq, I_fft_phase, 'r.')
# p3.vlines(frq,[0],I_fft_amp)
p3.grid(True)

plt.show()
