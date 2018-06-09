import numpy as np
import matplotlib.pyplot as plt
import utilities
import preproc
import network
import data
from pathlib import Path

# ----- Ben testing ----- #
#utilities.gen_sum_measured_syn('final_loads\\')
#-------------------------#

# load='power_thyristor_DB.txt'
# utilities.plot_signal('measured_loads\\tested\\edited\\signal_sum_val.txt',36000, noise=False)

# new_path = 'measured_loads\\tested\\edited\\'
# utilities.gen_sum_measured('measured_loads')
# pathlist = Path('measured_loads').glob('**/*.mat')      # Set prefix (.mat/.txt) to list intended files
# c = 0
# for path in pathlist:
#     if c == 8:
#         break
#     path = str(path)
#     print(path)
#
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
#
#     c += 1
#
#     Read mat files
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

# --------------------- Noise Plot ------------------ #
# noise_acc = np.zeros(11)
# for i in range(11):
#     utilities.preproc_config['noise_percentage'] = i/10
#     noise_acc[i] = network.train('data//lab-noise//')
# fig1 = plt.figure()
# p1 = fig1.add_subplot(111)
# p1.plot(range(0,110,10), noise_acc, 'b')
# p1.grid(True)
# p1.set_title('Exact Match Accuracy vs Noise')
# p1.set_xlabel('Noise Percentage')
# p1.set_ylabel('Accuracy')
# plt.show()

# --------------------- Num of Loads Plot ------------------ #
# loads_acc = np.zeros(9)
# for i in range(2,11):
#     path = 'data//plot//'  + str(i) + '//'
#     utilities.nn_config['num_classes'] = i
#     utilities.gen_sum(path, i)
#     loads_acc[i-2] = network.train(path)
# fig1 = plt.figure()
# p1 = fig1.add_subplot(111)
# p1.plot(range(2,11), loads_acc, 'b')
# p1.grid(True)
# p1.set_title('Exact Match Accuracy vs Number of Loads')
# p1.set_xlabel('No. Loads')
# p1.set_ylabel('Accuracy')
# plt.show()


# Signal Plotting for book
#utilities.plot_signal('data\lab-noise\signal_4_val.txt', index=0)

#network.train('data\\real_world_new\\', data.data_init_comb)

# Plot Accuracy for 4,8,16,32 permutations against test permutations

#network.train('data\\real_world_new\\', lambda path: data.data_init_comb_test(path, train_perm=32, test_perm=2))

if (True):
    models = ['32','16','8','4']
    accs = np.zeros((4,28))
    for i in range(0,28):
        test_data_unscaled, test_labels = data.data_init_for_plot('data\\real_world_new\\', i+4)
        for j in range(4):
            accs[j,i] = network.train_plot(test_data_unscaled,test_labels,'model'+str(models[j])+'.pkl')
    np.savetxt('stats.txt',accs, fmt='%.7f')
