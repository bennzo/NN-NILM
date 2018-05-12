import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc


utilities.plot_signal('data/signal_5_val.txt',260000, noise=False)


# F, A, P, I = utilities.load_signal(1, 'data\lab-noise\\')
# print(F)
# print(A)
# print(P)

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
