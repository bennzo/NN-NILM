import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc
from pathlib import Path

# load='power_thyristor_DB.txt'
# utilities.plot_signal('measured_loads\\tested\\edited\\signal_sum_val.txt',36000, noise=False)


# new_path = 'measured_loads\\tested\\edited\\'
# utilities.gen_sum_measured()
# pathlist = Path('measured_loads\\tested').glob('**/*.txt')
# c = 0
# for path in pathlist:
#     if c == 8:
#         break
#     path = str(path)
#     print(path)
#     I = np.loadtxt(path)
#     if not '1PH' in path:
#         I = np.tile(I,10)
#
#     on = np.ones(len(I), dtype=int)
#     i = step = 0
#     curr = np.random.randint(0, 2)
#     samples = len(I)
#     while i < samples:
#         step = np.random.randint(samples // 16, samples // 3)
#         on[i:i + step] = int(curr)
#         curr = not curr  # toggle
#         i += step  # step increment
#
#     label = on
#     I = I * on
#     np.savetxt(new_path +'val_'+path.split('\\', 2)[-1] , I[:200000], fmt='%.7f', delimiter='\n')
#     np.savetxt(new_path +'label_'+path.split('\\', 2)[-1] , label[:200000], fmt='%i', delimiter='\n')
#     c += 1
#     out = 'measured_loads\\' + (path.split('\\', 1)[-1]).replace('.mat','.txt')
#     utilities.mat2txt(path, out)









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
