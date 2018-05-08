import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc

#utilities.gen_sum()

I, I_label = utilities.load_sum('data\lab-noise\\')

# F, A, P, I = utilities.load_signal(1, 'data\lab-noise\\')
# print(F)
# print(A)
# print(P)

I_t = I[0:int(preproc.opt['Fs']*preproc.opt['sample_time'])]
I_fft = preproc.fft(I_t, True)
I_fft_amp, I_fft_phase  = preproc.fft_amp_phase(I_fft)


Fs = preproc.opt['Fs']
n = len(I_t)                # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T                   # two sides frequency range
frq = frq[range(int(n/2))]  # one side frequency range

print(preproc.fft2input(I_fft_amp, I_fft_phase, Fs))

# # Plot Signal
# fig1 = plt.figure()
# p1 = fig1.add_subplot(111)
# p1.plot(range(n), I_t, 'b')
# p1.set_title('Current Signal (Sum)')
# p1.set_xlabel('time')
# p1.set_ylabel('I')
#
# # Plot Amplitudes and Phase
# fig2 = plt.figure()
# p2 = fig2.add_subplot(121)
# p2.plot(frq, I_fft_amp, 'r.')
# p2.vlines(frq,[0],I_fft_amp)
# p2.grid(True)
# p2.set_title('Amplitude')
# p2.set_xlabel('Freq')
# p2.set_ylabel('A')
#
# p3 = fig2.add_subplot(122)
# p3.plot(frq, I_fft_phase, 'r.')
# # p3.vlines(frq,[0],I_fft_amp)
# p3.grid(True)
# p3.set_title('Phase')
# p3.set_xlabel('Freq')
# p3.set_ylabel('deg')

# Plot NN Input
# nn_input = preproc.fft2input(I_fft_amp, I_fft_phase, Fs)
# fig4 = plt.figure()
# p4 = fig4.add_subplot(111)
# #p4.plot(nn_input, 'r.')
# p4.stem(nn_input)
# p4.grid(True)
# #plt.axvline(10, color='k', linestyle='solid')
# p4.set_title('NN Input - Noise 50%')
#
# plt.show()
