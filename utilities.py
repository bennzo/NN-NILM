import numpy as np
import matplotlib.pyplot as plt
import preproc

preproc_config = {
    'Fs': 2*650+50,
    'sample_time': 0.1,
    'noise': True,
    'noise_percentage': 0.1,
    'train_test_ratio': 0.9,
    'threshold': 0.5
}

nn_config = {
    'input_size': 14,
    'output_size': 5,
    'num_classes': 5,
    'num_epochs': 100,
    'batch_size': 30,
    'learning_rate': 1e-3
}


def gen_I(path='', n=1):
    harmony_n = np.random.randint(2,8)
    f = np.array([1,3,5,7,9,11,13]) * 50  # frequency space

    # ------ Signal properties ------ #
    F = np.sort(np.random.choice(f, harmony_n, replace=False))                                # frequency vector
    P = np.random.uniform(-1 / 2 * np.pi, 1 / 2 * np.pi, harmony_n)                           # phase vector
    A = np.sort(np.random.randint(1, 10, size=harmony_n))[::-1]                               # amplitude vector

    # ------ Generate Signal ------ #
    Fs = preproc_config['Fs']                                                                 # sampling rate
    Ts = 1.0 / Fs                                                                             # sampling interval
    T_fin = 200                                                                               # total time
    t = np.arange(0, T_fin, Ts)                                                               # time vector
    I_t = np.array([np.sin(2 * np.pi * F * t[i] + P) for i in range(0, np.size(t))]).dot(A)   # signal

    # ------ Determine on/off ------ #
    #on = np.sort(np.random.choice(T_fin*Fs-1, 16, replace=False))
    # label = np.ones((T_fin*Fs))
    # for i in range(0,np.size(on),2):
    #     I_t[on[i]:on[i+1]] = 0
    #     label[on[i]:on[i+1]] = 0

    on = np.ones((T_fin*Fs), dtype=int)

    # ------ Old labeling ------ #
    # for i in range(0, (T_fin*Fs), int(T_fin*Fs/(2**n))*2):
    #     on[i:i+int(T_fin*Fs/(2**n))] = 0
    # label = on

    # ------ New labeling ------ #
    i= step = 0
    curr = np.random.randint(0,2)
    samples = T_fin*Fs
    while i < samples:
        step = np.random.randint(samples // 16, samples // 3)
        on[i:i + step] = int(curr)
        curr = not curr           # toggle
        i += step           # step increment

    label = on
    I_t = I_t*on

    save_signal(n,F,A,P,I_t,label,path)
    return I_t, label


# TODO- add new function generate multi state load
def gen_I_multi_state(path='', n=1, states=2):
    pass


# TODO- add multi states loads to sum
def gen_sum(path='', n=nn_config['num_classes']):
    S_sum , label_sum = gen_I(path)
    for i in range(2,n+1):
        temp_sum, temp_label = gen_I(path,i)
        S_sum += temp_sum
        label_sum = np.vstack((label_sum,temp_label))

    np.savetxt(path+"signal_sum_val.txt", S_sum, fmt='%.7f', delimiter='\n')
    np.savetxt(path+"signal_sum_label.txt", np.transpose(label_sum), fmt='%i', delimiter=',')


def save_signal(i,F,A,P,I,label,path):
    np.savetxt(path+"signal_{}_prop.txt".format(i), (F, A, P), fmt='%.3f', delimiter=',')
    np.savetxt(path+"signal_{}_val.txt".format(i), I, fmt='%.7f', delimiter='\n')
    np.savetxt(path+"signal_{}_label.txt".format(i), label, fmt='%i', delimiter='\n')


def load_signal(i, folder_name):
    F, A, P = np.loadtxt(folder_name + "signal_{}_prop.txt".format(i), delimiter=',')
    I = np.loadtxt(folder_name + "signal_{}_val.txt".format(i))
    return F, A, P, I


def load_sum(folder_name):
    I = np.loadtxt(folder_name + "signal_sum_val.txt")
    I_label = np.loadtxt(folder_name + "signal_sum_label.txt", delimiter=',')
    return I, I_label


def compare_input(I1,I2):
    return np.sqrt(((I1 - I2)**2).sum())


def plot_signal(s_path, index=-1,noise=False):
    label_path = s_path.replace('_val', '_label')
    sample_win = int(preproc_config['Fs']*preproc_config['sample_time'])
    I = np.loadtxt(s_path)
    I_label = np.loadtxt(label_path, delimiter=',')
    if index > (len(I)-sample_win-1):
        print('index inserted is not valid')
        return 1
    if index < 0:
        index = np.random.randint(0,(len(I)-sample_win-1))
    if not('sum' in s_path):
        for i in range(index, (len(I)-sample_win-1)):
            if I_label[i] == 0:
                continue
            else:
                break
        index=i
    sampled_I = I[index:index+sample_win]
    # sampled_label = I_label[index:index+sample_win]
    I_fft = preproc.fft(sampled_I,noise=noise)
    I_fft_amp, I_fft_phase = preproc.fft_amp_phase(I_fft)

    n = len(sampled_I)                # length of the signal
    k = np.arange(n)
    T = n/preproc_config['Fs']
    frq = k/T
    frq = frq[range(int(n/2))]

    # Plot Signal
    fig1 = plt.figure()
    p1 = fig1.add_subplot(111)
    p1.plot(range(n), sampled_I, 'b')
    p1.set_title('Current Signal')
    p1.set_xlabel('time')
    p1.set_ylabel('I(A)')

    # Plot Amplitudes and Phase
    # Amplitude
    fig2 = plt.figure()
    p2 = fig2.add_subplot(121)
    p2.plot(frq, I_fft_amp, 'r.')
    p2.vlines(frq,[0],I_fft_amp)
    p2.grid(True)
    p2.set_title('Amplitude')
    p2.set_xlabel('Freq')
    p2.set_ylabel('A')
    # Phase
    p3 = fig2.add_subplot(122)
    p3.plot(frq, I_fft_phase, 'r.')
    # p3.vlines(frq,[0],I_fft_amp)
    p3.grid(True)
    p3.set_title('Phase')
    p3.set_xlabel('Freq')
    p3.set_ylabel('deg')

    plt.show()

if __name__ == "__main__":
    path='data//'
    gen_sum(path)
    print('Data generated successfully')
